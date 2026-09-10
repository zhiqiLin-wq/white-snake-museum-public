"""Agentic Loop (v6) — 真正的 ReAct Agent，LLM 自主决定工具调用。

核心改进 (v4 → v6):
- ReAct 主循环: Think → Act → Observe → Reflect → ... → Answer
- _try_fast_answer(): 快速模型预检，80% 简单查询无需进入 ReAct
- 自动充足性评估: 每次工具调用后评估知识缺口
- Token 预算控制: 防止无限循环耗尽配额
- 流式 tool_call/tool_result 事件: 前端可展示 Agent 思考过程
"""
import asyncio
import json
import logging
import re
from typing import Optional, AsyncGenerator

from .llm.json_utils import parse_llm_json
from .llm.provider import LLMFatalError, RateLimitError, APITimeoutError
from .llm.tool_call import ToolDefinition, ToolCallRequest, ToolCallResult, GenerateWithToolsResult
from .api.events import SSEEventType, build_sse_event
from .observability.tracer import llm_call_context
from .dispatch.tool_dispatcher import ToolDispatcher

logger = logging.getLogger(__name__)

# B-084: v19 Token 预算调整 —— 长报告分段写入 + 深度研究任务需要更多轮次
MAX_REACT_ITERATIONS = 12   # v8: 7 -> 12（深度对勘/报告任务需多轮工具调用）
# B-167: 预算口径改为"纯可见输出"——provider 已剔除 reasoning_tokens（思维链），
# 30000 足够 12 轮 ReAct + 7 章 append；但深度对勘/报告任务检索轮次多，适度上调到
# 50000 留余量，避免长报告写到一半因预算耗尽被迫停止（B-167 根因之一）。
MAX_TOKENS_BUDGET = 50000  # v8: 12000 -> 30000 -> 50000（纯可见输出 tokens 口径）
DUPLICATE_TOOL_THRESHOLD = 2  # 保持不变
_GUARDRAIL_KEEP_RECENT = 2  # v16: 上下文护栏触发时保留最近 N 条工具消息完整


# ---- 回答可读性清洗：防止工具名/技术字段泄露到用户可见文本 ----
_TOOL_NAME_CN = {
    "find_shared_passages": "共享片段比对",
    "diff_passages": "逐字异文比对",
    "dynasty_compare_elements": "跨朝代元素对比",
    "search_scholar_metadata": "学术文献检索",
    "save_long_report": "完整报告",
    "verify_citation_tags": "来源校验",
    "coverage_tracker": "标注覆盖率统计",
    "annotation_knowledge_base": "标注知识库",
    "span_validator": "标注校验",
    "build_timeline": "时间线梳理",
    "save_research_snapshot": "研究快照",
    "multi_entity_cooccurrence": "实体共现分析",
    "extract_dialogues": "对话提取",
    "search_poetry": "诗词检索",
    "get_version_metadata": "版本考证",
    "compare_passages": "段落语义对比",
    "align_paragraphs": "段落对齐",
}
_TAG_LABELS = {"文献记载", "基于文献推断", "学界共识", "AI补充", "无记载", "待考", "AI基于训练知识补充"}


def _clean_source_tag(m: "re.Match") -> str:
    """清洗来源标签内部的技术字段。

    【文献记载·diff·position 3】→ 【文献记载】
    【文献记载·find_shared_passages·verdict】→ 【文献记载】
    整标签无合法中文段 → 整个删除（返回空串）
    """
    inner = m.group(1)
    parts = re.split(r"[·|／/]", inner)
    kept: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # 含 ASCII 字母/下划线 → 工具名/字段名/英文标识符 → 丢弃
        if re.search(r"[A-Za-z_]", p):
            continue
        # 纯数字段（如 "3"）→ 丢弃
        if p.isdigit():
            continue
        kept.append(p)
    if not kept:
        return ""
    return "【" + "·".join(kept) + "】"


def sanitize_answer_text(text: str) -> str:
    """清洗最终回答中的技术泄露：工具英文名、字段名、技术性来源标签。"""
    if not text:
        return text
    # ① 来源标签内的技术字段
    text = re.sub(r"【([^】]*)】", _clean_source_tag, text)
    # ② 正文中的工具英文名 → 中文名（带边界，避免误伤）
    for en, cn in _TOOL_NAME_CN.items():
        text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(en) + r"(?![A-Za-z0-9_])", cn, text)
    # ③ 工具名替换后，中文之间遗留的空格合并（"根据 共享片段比对 的结果" → "根据共享片段比对的结果"）
    for _ in range(3):
        text = re.sub(r"([一-鿿])[ \t]+([一-鿿])", r"\1\2", text)
    return text


class StreamSanitizer:
    """流式回答清洗器：按小句边界缓冲，清洗后再吐出。

    LLM 流式输出时 chunk 边界随机，技术泄露标记（如 【...·position 3】）
    可能跨 chunk。缓冲到最近的小句边界（标点/标签闭合）再清洗输出，
    延迟仅一个标点距离（中文约 20-40 字），用户无感知。
    """

    BOUNDARY = "，。；！？、\n】…"
    MAX_HOLD = 120  # 超长无标点段落强制输出上限（标签最长约 40 字，120 足够安全）

    def __init__(self):
        self._buf = ""

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []
        while True:
            last_pos = -1
            for ch in self.BOUNDARY:
                p = self._buf.rfind(ch)
                if p > last_pos:
                    last_pos = p
            if last_pos >= 0:
                ready = self._buf[: last_pos + 1]
                self._buf = self._buf[last_pos + 1:]
                out.append(sanitize_answer_text(ready))
            elif len(self._buf) >= self.MAX_HOLD:
                out.append(sanitize_answer_text(self._buf))
                self._buf = ""
            else:
                break
        return "".join(out)

    def flush(self) -> str:
        if self._buf:
            t = sanitize_answer_text(self._buf)
            self._buf = ""
            return t
        return ""


class TokenBudget:
    """Token 预算跟踪器。

    v17.1 口径: 只累计"生成输出 tokens"（completion_tokens）——
    输入上下文（system/工具schema/历史）由 CONTEXT_GUARDRAIL_TOKENS 护栏
    单独压缩管辖。此前误用 usage.total_tokens 导致输入每轮重复计费，
    预算虚胖爆表。
    """

    def __init__(self, limit: int = MAX_TOKENS_BUDGET):
        self.limit = limit
        self.used = 0

    def consume(self, tokens: int):
        self.used += tokens

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def is_exhausted(self) -> bool:
        return self.used >= self.limit

    @property
    def warning_ratio(self) -> float:
        return self.used / self.limit if self.limit > 0 else 0



class AgentLoop:
    """Agentic Loop — ReAct 模式的通用问答 Agent。

    流程:
    1. _try_fast_answer(): 快速检索 + 快模判断 → 足够则直接流式回答 (跳过 ReAct)
    2. ReAct 循环 (最多 5 轮):
       a. LLM 决策: 回答文本 或 调用工具
       b. 执行工具 → 追加结果到 messages
       c. 快速模评估: 信息是否充足? → 是则退出循环
    3. 主力模型流式生成最终答案 + 来源
    """

    def __init__(self, llm, retriever, prompt_registry=None, mcp_registry=None, node_client=None, user_id: str = "", system_context: str = ""):
        self.llm = llm
        self.retriever = retriever
        self.prompts = prompt_registry
        self.mcp = mcp_registry
        self.node_client = node_client
        self.user_id = user_id
        self._system_context = system_context
        # 统一工具调度层：收敛 user_id 注入 / tool_call_context / 调用 / 截断
        self.dispatcher = ToolDispatcher(mcp_registry=self.mcp, retriever=self.retriever)

    # ================================================================
    #  工具定义 — 从 MCP Registry 动态加载 (U02-04)
    # ================================================================

    def _load_tool_definitions(self) -> list[ToolDefinition]:
        """从 MCP Registry 动态加载工具定义。"""
        if not self.mcp:
            return []
        tools = []
        for tool_info in self.mcp.list_tools():
            schema = tool_info.get("inputSchema", {})
            # 确保 parameters 是合法的 JSON Schema
            params = {
                "type": schema.get("type", "object"),
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            }
            tools.append(ToolDefinition(
                name=tool_info["name"],
                description=tool_info.get("description", ""),
                parameters=params,
            ))
        return tools

    # ================================================================
    #  System Prompt (U02-01)
    # ================================================================

    def _build_system_prompt(self, skill: str = None) -> str:
        """组装 ReAct 模式的系统提示词。"""
        prompt = (
            "你是「白蛇传·文脉全息博物馆」的 AI 助手。"
            "你的知识基于《太平广记》《西湖三塔记》《白娘子永镇雷峰塔》《雷峰塔传奇》等历代白蛇传文本。\n\n"
            "## 工作模式 (ReAct)\n"
            "你需要自主决定是否使用工具来获取信息：\n"
            "1. 如果问题简单或你已有足够知识 → 直接回答\n"
            "2. 如果用户要求打开/查看/跳转到某章节或某段 → 必须调用 get_chapter_full_text(chapter_number=N)。这个工具会触发前端自动跳转到对应章节。如果用户指定了段落号(如'第12段')，传入 paragraph_index 参数(段落号-1，即0-based索引)\n"
            "3. 如果需要文献证据 → 调用 search_literature 或 search_paragraphs\n"
            "4. 标注任务 (最高优先级): 如果用户要求标注文本实体(人物/地点/事件/术语/母题)，且用户消息中已包含章节编号 → 直接调用 annotate_user_request(chapter_number=N, user_instruction=\"...\")，无需先检索文献。annotate_user_request 内部会自动获取章节全文。"
            "段落索引约定: 用户说的「第N段」对应 paragraph_indices=[N-1]（0-based，与前端一致）。\n"
            "4.1 批注任务: 如果用户要求写批注/旁批/评注（如「给第12段写法海形象的批注」），按以下两步执行:\n"
            "  a. 先调用 get_chapter_full_text(chapter_number=N) 获取段落原文，定位到目标段落并仔细阅读。\n"
            "  b. 基于原文内容，生成一段学术性的批注文本（100-300 字），然后调用 annotate_user_request(\n"
            "       chapter_number=N,\n"
            "       user_instruction=\"用户原话\",\n"
            "       paragraph_indices=[目标段落的0-based索引],\n"
            "       marginalia_notes=[{paragraph_index: 目标段落的0-based索引, content: \"生成的批注\"}]\n"
            "     )\n"
            "  c. 如果用户同时要求实体标注 + 批注（如「把人物标出来，再给第12段写个批注」），在一个 annotate_user_request 调用中同时传入两套参数。\n"
            "4.2 删除标注: 如果用户要求删除/清除/去掉/移除标注（如「删除第4章的地点标注」），调用 delete_annotations 工具。\n"
            "  参数: chapter_number=N（必填），categories=[\"person\",\"location\",...]（可选，指定要删除的类别），\n"
            "  delete_all=true（删除全部时使用）。例如：「删除第4章全部标注」→ delete_annotations(chapter_number=4, delete_all=true)。\n"
            "4.3 标注质量自检（P0 必走规则）:\n"
            "  a. **标注覆盖率 / 完整度查询**: 当用户问及「标注覆盖率多少」「第X章还有哪些段没标」\n"
            "  「标签完整度如何」「哪几章标注漏了」「哪些章节需要补标」，调用 coverage_tracker。\n"
            "  参数可选: chapter_numbers=[章节号]（空=全量）、categories=[类别过滤]、\n"
            "  group_by='chapter'|'category'|'dynasty'（按章节/标签/朝代分组）、include_unannotated=true。\n"
            "  返回会给出 excellent/good/fair/poor 四档质量评级，以及 needAttentionChapters（最需补标的 Top10）。\n"
            "  b. **标注规范与领域知识库查询**: 遇到以下任一情况必须先调 annotation_knowledge_base，再执行标注：\n"
            "     - 不确定「某词是 person 还是 term 还是 custom」（如雷峰塔既可是地点又可是术语）→ lookup_entity；\n"
            "     - 用户问「母题的定义是什么」「事件句子和主题句子怎么区分」「某类规则是什么」→ category_rules；\n"
            "     - 想知道某类别（如地点）已知有哪些实体词表 → get_known_entities；\n"
            "     - 识别到 LLM 返回了中文类别名（如「人物名称」）时，调用 normalize_category 转成标准 key 再写库。\n"
            "  c. **标注偏移体检**: 仅当用户明确反馈「标注标错字/标到空白处/看不到高亮词/高亮截断」\n"
            "  「帮我检查第X章的标注准确性」时，调用 span_validator。正常 annotate_user_request / annotate_passage\n"
            "  写出前系统会自动走 6 级回退校验，不需要你手工调。不要主动调此工具造成冗余。\n"
            "4.4 答案来源自检 (P1 自动后处理): 生成最终答案后，系统会自动调 verify_citation_tags\n"
            "  扫描你的答案中的 5 档来源标签和章节号引用，反查章节是否真实存在/是否在本次召回的 sources 中。\n"
            "  不一致会在答案末尾追加「⚠️ 来源校验」告警块。**你不需要手工调此工具**。\n"
            "  要点: ① 引用章节号务必来自工具召回的 sources，不要凭记忆写「第8章」如果工具只召回到第 5 章；\n"
            "  ② 每条事实陈述后附【文献记载】/【基于文献推断】/【AI补充】/【无记载】/【待考】之一。\n"
            "5. 如果用户要求统计或对比数据 → 优先调用 count_occurrences 或 compare_passages\n"
            "6. 如果需要实时信息 → 调用 web_search\n"
            "6.1 学术文献检索: 当用户问「关于白蛇传的学术研究」「方成培研究论文」「比较文学研究有哪些」\n"
            "    「有哪些学术论文讨论白蛇传」时，调用 search_scholar_metadata（CrossRef API，返回真实论文 DOI/作者/年份/引用数）。\n"
            "    引用时标注【学界共识】。不要凭记忆编造论文标题。\n"
            "6.2 时间线/事件轴: 当用户问「白蛇形象演变时间线」「按时间顺序展示」「各朝代关键事件列表」\n"
            "    「画一条演变时间轴」时，调用 build_timeline。返回结构化 JSON 含朝代/年份范围/事件/证据摘录/段落锚点。\n"
            "    与 sentiment_timeline（情感值变化）/ keyword_trend（词频变化）互补，按需选工具。\n"
            "6.3 逐字异文对比: 当用户要求「逐字对比两段」「异文对比」「一个字一个字地看差异」\n"
            "    「同一情节在不同版本怎么改的」时，调用 diff_passages。返回字级 diff 列表。\n"
            "    与 compare_passages（语义级文学分析）不同，diff_passages 是字级对齐，看改了哪几个字。\n"
            "6.4 研究结论存证: 当用户说「保存这个结论」「冻结当前分析」「我要存证」时，调用\n"
            "    save_research_snapshot(action='save', conclusion='结论文本', research_topic='标题')。\n"
            "    当用户说「对比之前的结论」「之前的结论还能复现吗」时，调用\n"
            "    save_research_snapshot(action='compare', snapshot_id='ID') 检查系统参数是否变了。\n"
            "6.5 跨朝代对比分析: 当用户要求对比不同朝代版本的元素时，调用 dynasty_compare_elements。\n"
            "    朝代→章节号映射: 唐代=1, 宋代=2, 明代=3, 清代=4(或5或6), 现当代=7。\n"
            "    用户说「清代和现当代」→ left=4, right=7; 「唐和明」→ left=1, right=3。\n"
            "    模糊表述映射: 「古时候/最早」→唐(1); 「后来/之后」→按语境选宋(2)/明(3)/清(4);\n"
            "    「现在/现代/当代」→现当代(7)。如果用户只提到一个朝代，可自行选择一个合理的对比朝代。\n"
            "6.6 内容完整性与长报告（铁律）: 你的首要职责是事实完整，不是把内容塞进对话框。\n"
            "对话框回答有生成长度上限（约 2000 字），靠压缩、概括、省略、删减素材来"
            "「挤进对话框」是错误做法——丢失的正是用户要的事实。内容确实多时，"
            "正确做法是写报告文件，对话框只给摘要。\n"
            "  【必须使用 save_long_report 写文件的客观信号】（满足任一即写文件，"
            "不要凭「感觉不长」跳过，也不要等回答被截断才想起来）:\n"
            "   - 工具返回大量条目：entity_appearances / multi_entity_cooccurrence / "
            "search_poetry / search_literature / build_timeline 等返回超过 15 条结果，"
            "或素材涉及 3 个及以上章节；\n"
            "   - 用户要求「完整 / 全部 / 逐一 / 详细 / 系统梳理 / 汇总 / 盘点」且素材跨多章节；\n"
            "   - 回答形态是穷举清单：逐处出场轨迹、逐首诗词、逐章情节、逐字对勘、"
            "多维度跨朝对比、完整时间线加分析、系统性综述；\n"
            "   - 你发现自己开始想「这里概括一下」「篇幅有限只列重点」——立刻停止，改走文件。\n"
            "  【做法】:\n"
            "  a. save_long_report(action='create', title='报告标题', topic='用户原始问题') → 得到 report_id\n"
            "  b. 多次调用 save_long_report(action='append', report_id=..., section_title='小节名', "
            "content='该节完整正文')，每次只写一节（每节 1000-2000 字，绝不超长）；"
            "把工具返回的全部素材按章节/朝代/类别完整落盘，一条都不许丢\n"
            "  c. 【封卷前自检·必做】finalize 之前逐项核对：报告是否覆盖了本次问题应有的全部范围？"
            "跨朝代问题必须七章（唐/宋/明/清戏曲/清小说/清弹词/现当代）逐章有着落，"
            "出场/共现类问题必须把工具返回的每个章节分组都写入；"
            "韵文/诗词汇编类必须以工具返回的各章命中分布为准——"
            "命中为 0 的章也要写一节说明「该章无韵文（检索依据）」，"
            "命中章数不足以成节时用 chapter 参数逐章补检后再写；"
            "只要还有已收集却没落盘的章节或条目，就继续 append 补齐，"
            "严禁提前封卷造成事实缺失。确认无遗漏后再 finalize\n"
            "  d. save_long_report(action='finalize', report_id=...) 封卷（系统自动推送下载卡片）\n"
            "  e. 对话框只给简明摘要：核心发现 5-8 条 + 「完整报告已生成（共 N 字，可点击下载）」\n"
            "  【对话框直答】仅适用于：素材少（不超过 15 条且集中在 1-2 章）、单一事实查询、"
            "闲聊、明确的操作确认。\n"
            "  注意：正文引用原文时照录古籍文字即可，不要在正文里提及任何工具名或字段名。\n"
            "7. 获取信息后 → 基于证据给出引用来源的回答\n\n"
            "## 回答原则\n"
            "- 基于文献证据回答，引用具体章节名\n"
            "- 不确定的地方明确标注「待考」\n"
            "- 使用学术但不失可读性的中文\n"
            "- 每次工具调用后评估信息是否充足\n"
            "- 标注类请求必须直接调用 annotate_user_request 工具，绝对禁止用文字描述标注结果\n"
            "- 绝对不要用 search_literature 或 search_paragraphs 来查找需要标注的章节——annotation 工具自己会读取全文\n"
            "- 【可读性铁律】回答是给人文学者看的，绝对不允许出现：工具英文名（如 diff_passages、"
            "find_shared_passages、dynasty_compare_elements 等）、技术字段名（如 position、verdict、"
            "changeType、leftText）、JSON 键名或英文标识符。来源标签只用中文档位名，可附章节名，"
            "格式如【文献记载】或【文献记载·第四章】，禁止出现任何英文字符。\n"
            "8. 如果用户要求查看特定朝代的地图标记（如「显示清代的地图」「只看唐宋的地点」"
            "「显示所有朝代」），调用 set_map_dynasty_layer 工具。"
            "dynasty 参数接受单个朝代字符串（如'清'）或多个朝代数组（如['唐','宋']）。"
            "action 参数：'set'=单选替换, 'toggle'=追加切换, 'reset'=恢复全部。默认使用'set'。"
            "例如：「显示清代的地图标记」→ set_map_dynasty_layer(dynasty='清')"
            "「只看唐宋的地图标记」→ set_map_dynasty_layer(dynasty=['唐','宋'])"
            "「显示所有朝代」→ set_map_dynasty_layer(dynasty='all')\n"
            "9. 如果用户要求定位到/飞到/聚焦/跳转到某个地点（如「定位到雷峰塔」「飞到断桥」「聚焦西湖」"
            "「跳转到金山寺」），调用 focus_map_location(location_name=\"地点名\") 工具。"
            "此工具会触发前端地图飞行动画并自动打开该地点的溯源卡片，"
            "你不需要在回答中用文字描述地图操作结果，简单确认即可。"
            "12个支持的地点：雷峰塔/西湖/金山寺/断桥/峨眉山/青城山/承天寺/卧佛寺/望江楼/龙虎山/灵隐寺/西子湖\n"
            "10. 景观-文本互构深度分析: 如果用户要求深入分析某个地点的景观与文本之间的双向影响关系"
            "（含关键词：互构、景观-文本、双向溯源、景观与文本），按以下步骤执行:\n"
            "  a. 调用 generate_mutual_construction(location_name=\"地点名\") 获取结构化的互构分析数据"
            "（包含 landscapeToText / textToLandscape 双方向分析、各朝代 descriptionStyles、关联母题 relatedMotifs）\n"
            "  b. 调用 get_location_traceability(location_names=[\"地点名\"]) 获取该地点在各朝代的详细文献记录"
            "（包含 dynasty、chapterTitle、excerpt、descriptionStyle 等字段）\n"
            "  c. 基于以上两个工具返回的数据，综合生成一份深度学术分析（500-1500字），"
            "以流式 Markdown 输出到 Agent Panel。分析需涵盖:\n"
            "    - 该地点的物理景观特征及其对文本叙事场景塑造的影响\n"
            "    - 各朝代文本中该地点描写方式的演变轨迹\n"
            "    - 景观->文本方向: 物理空间的视觉特征、建筑布局等如何影响了情节走向和叙事方式\n"
            "    - 文本->景观方向: 文学作品如何塑造了该地点的文化认知、象征意义和社会地位\n"
            "    - 与该地点关联的叙事母题及其跨朝代演化\n"
            "  d. 引用具体朝代名和章节名作为证据来源\n"
            "  e. generate_mutual_construction 的返回结果也会通过 SSE 事件自动更新地图溯源卡片，"
            "你不需要在回答中描述这个技术细节\n"
            "11. 导出标注数据: 如果用户要求导出/下载/保存标注数据（含关键词：导出、下载、保存 + 标注/CSV/JSON/HTML），"
            "调用 export_annotations 工具。\n"
            "  a. 从用户语句中解析章节编号：如「第4章」对应 chapter_numbers=[4]，"
            "「第1章和第3章」对应 [1,3]，「全部章节」对应所有章节编号（如 [1,2,3,4,5,6,7,8]）。\n"
            "  b. 解析格式：「CSV」「表格」→ csv，「JSON」→ json，「HTML」「网页」→ html。默认为 csv。\n"
            "  c. 解析来源范围：「我的标注」「用户标注」→ user，"
            "「Agent 标注」「AI 的标注」→ agent，未指定或「全部」→ all。\n"
            "  d. 调用后前端会自动生成文件并触发浏览器下载，你只需简短确认即可，"
            "无需描述文件内容。\n"
            "12. 智能标注建议: 如果不确定用户想要的标注范围，"
            "先调用 suggest_annotations 生成半透明预览标注，"
            "让用户确认后再调用 annotate_user_request 正式标注。\n"
            "13. 分栏差异标注: 如果用户在对照视图中要求标注差异，"
            "调用 compare_and_annotate 同时在左右两侧标注。\n"
            "14. 段落对齐: 如果用户需要对齐两个章节的对应段落，"
            "调用 align_paragraphs。\n"
            "15. 白话翻译: 如果用户要求解释/翻译文言文段落，"
            "调用 explain_passage 生成白话翻译和文化背景。\n"
            "16. 趋势与情感分析: 如果用户要求分析某实体的跨朝代趋势，"
            "调用 keyword_trend 获取趋势数据，"
            "调用 sentiment_timeline 获取情感变化时间线。\n"
            "17. 共现分析: 如果用户要求分析实体间的关联，"
            "调用 cooccurrence_matrix 获取共现矩阵。\n"
            "18. 出场轨迹与文本传承: 如果用户问'XX在哪些章节/段落出场''XX首次出现在哪'，"
            "调用 entity_appearances 定位每一处出场位置（结果可直接跳转原文）；"
            "如果用户问'两个版本哪些情节是直接沿用的''找出两章雷同/差异段落'，"
            "调用 find_shared_passages 获取段落级相似证据；"
            "如果用户问'第X章讲了什么''快速了解某版本剧情'，"
            "调用 chapter_summary 获取情节骨架。\n"
            "⚠️ 数据口径（转述给用户时必须如实说明）: "
            "keyword_trend/sentiment_timeline/cooccurrence_matrix 基于『语义检索召回的相关段落』"
            "统计，数字表示相关程度分布，不是字面出现次数；"
            "需要全文逐字的真实频次时使用 count_occurrences（字面计数，实体未出现会返回 0 并提示别名）。\n"
            "19. 诗词曲牌检索: 如果用户要求查找诗词/唱词/曲牌/韵文内容"
            "（如'找出所有写西湖的诗词''列出租师唱词''某版本的诗赞'），"
            "调用 search_poetry(query='关键词', form_type='诗|词|曲|唱词|全部')。"
            "该工具通过文体标记词（诗曰/词曰/曲曰/有诗为证 等）定位韵文段落。\n"
            "20. 多实体关联出场: 如果用户问'XX和YY的关联出场''XX和YY在哪些段落同框'"
            "'金山寺和雷峰塔的同时出现'，调用 multi_entity_cooccurrence(entities=['实体1','实体2'])。"
            "返回多个实体在同一段落中共现的位置和摘录。至少 2 个实体，最多 4 个。\n"
            "21. 角色对话提取: 如果用户问'白娘子说了什么''法海的台词''许仙的对话'，"
            "调用 extract_dialogues(character='角色名', chapter_number='章节号')。"
            "该工具通过对话标记模式（X道/X说/X曰 等）定位角色发言，自动识别常见别名。"
            "支持跨版本对比同一角色的发言风格。\n"
            "22. 版本元数据: 如果用户问'白蛇传有哪些版本''方成培是什么朝代的'"
            "'雷峰塔传奇的作者是谁''冯梦龙是什么人'，调用 get_version_metadata(chapter_number='章节号')。"
            "返回作者/朝代/体裁/文献来源等事实性信息，确保不编造。\n"
        )
        if skill:
            prompt += f"\n当前激活技能：{skill}"
        # 注入记忆上下文（摘要 + Core Memory + 被动注入），由 ContextManager 提供 (P-fix)
        if self._system_context:
            prompt += f"\n\n## 记忆与上下文\n{self._system_context}"
        return prompt

    # ================================================================
    #  工具调用
    # ================================================================

    async def _post_verify_citations(self, answer: str, sources: list) -> str:
        """P1-1: 答案生成后自动调 verify_citation_tags 后处理。

        扫描 5 档来源标签和章节号引用，反查章节是否真实存在/在 sources 中。
        有可疑时在 answer 末尾追加「⚠️ 来源校验」告警块。无问题原样返回。
        异常时静默降级（不阻塞答案输出）。
        """
        import time as _time
        _t0 = _time.perf_counter()
        try:
            from .mcp.tools.verify_citation_tags import handler as vct_handler
            result = await vct_handler(
                answer=answer, sources=sources,
                node_client=self.node_client, retriever=self.retriever,
                level="lenient",
            )
            _elapsed = (_time.perf_counter() - _t0) * 1000
            if result.get("verified"):
                logger.info(f"[P1-1] verify_citation_tags 通过: {result.get('stats')} ({_elapsed:.0f}ms)")
                return answer
            stats = result.get("stats", {})
            warnings = result.get("warnings", [])
            logger.info(
                f"[P1-1] verify_citation_tags 发现 {len(warnings)} 处可疑: "
                f"tags={stats.get('total_tags', 0)}, refs={stats.get('total_refs', 0)}, "
                f"quotes={stats.get('quotes_total', 0)}(verified={stats.get('quotes_verified', 0)},"
                f"suspicious={stats.get('quotes_suspicious', 0)}) ({_elapsed:.0f}ms)"
            )
            return answer + result.get("append_hint", "")
        except Exception as ex:
            logger.warning(f"[P1-1] verify_citation_tags 后处理异常（降级跳过）: {ex}")
            return answer

    async def _verify_report_citations(self, user_id: str, report_id: str,
                                       report_content: str, sources: list,
                                       truncated: bool = False) -> dict:
        """v21/P1-2: 报告封卷后校验 md 文档中的引文是否真实命中原文。

        与 _post_verify_citations 不同——本方法校验的是**写入 md 文件的完整报告内容**，
        而非对话框中的简短答案。校验结果追加为报告末尾的「来源校验」节。

        设计原则:
        - 绝不修改/删除报告正文（保证文档完整性）
        - 只在末尾追加「来源校验」节，列出可疑引文
        - 性能：并行检索 + 上限 8 条 + fast path（sources excerpt）
        - truncated=True 时额外标注"报告可能因 token 限制截断"

        Returns:
            {"stats": {..., "elapsed_ms"}, "append_hint": str}
        """
        import time as _time
        from .mcp.tools.verify_citation_tags import (
            QUOTE_EXTRACT_RE, _norm_quote_content, _quote_verbatim_in,
            MAX_QUOTE_LOOKUPS,
        )
        _t0 = _time.perf_counter()
        stats = {
            "quotes_total": 0, "quotes_verified": 0, "quotes_suspicious": 0,
            "truncated": truncated, "elapsed_ms": 0.0,
        }

        if not report_content:
            stats["elapsed_ms"] = (_time.perf_counter() - _t0) * 1000
            return {"stats": stats, "append_hint": ""}

        # 1. 合并 sources excerpt（fast path）
        sources_corpus = ""
        for s in (sources or []):
            exc = s.get("excerpt", "") or s.get("excerptFull", "")
            if exc:
                sources_corpus += exc

        # 2. 提取引文（与 verify_citation_tags 同构，跳过 <8 字短片段）
        quotes_found = QUOTE_EXTRACT_RE.findall(report_content)
        stats["quotes_total"] = len(quotes_found)
        unique_quotes: list[str] = []
        seen_norm: set[str] = set()
        for _, q, _ in quotes_found:
            nq = _norm_quote_content(q)
            if len(nq) >= 8 and nq not in seen_norm:
                seen_norm.add(nq)
                unique_quotes.append(q)
        quotes_to_lookup = unique_quotes[:MAX_QUOTE_LOOKUPS]

        # 3. 快速路径 + 并行慢速路径（与 verify_citation_tags v20.1 同构）
        warnings: list[dict] = []
        quotes_need_retrieval: list[str] = []
        for qt in quotes_to_lookup:
            if _quote_verbatim_in(sources_corpus, qt):
                stats["quotes_verified"] += 1
            else:
                quotes_need_retrieval.append(qt)

        if self.retriever is not None and quotes_need_retrieval:
            import asyncio as _aio
            async def _lookup_one(qt: str) -> tuple[str, bool]:
                try:
                    hits = await self.retriever.retrieve(qt, top_k=3)
                    if isinstance(hits, list):
                        for h in hits:
                            content = h.get("content", "") if isinstance(h, dict) else ""
                            if content and _quote_verbatim_in(content, qt):
                                return qt, True
                except Exception:
                    pass
                return qt, False
            results = await _aio.gather(*[_lookup_one(q) for q in quotes_need_retrieval])
            for qt, ok in results:
                if ok:
                    stats["quotes_verified"] += 1
                else:
                    stats["quotes_suspicious"] += 1
                    preview = qt[:50] + ("..." if len(qt) > 50 else "")
                    warnings.append({
                        "type": "quote_not_in_source",
                        "severity": "high",
                        "message": f"引文「{preview}」未在文献原文中逐字命中，可能为模型训练数据或编造。",
                        "quote": preview,
                    })
        elif quotes_need_retrieval:
            for qt in quotes_need_retrieval:
                stats["quotes_suspicious"] += 1
                preview = qt[:50] + ("..." if len(qt) > 50 else "")
                warnings.append({
                    "type": "quote_not_in_source",
                    "severity": "high",
                    "message": f"引文「{preview}」未在文献原文中逐字命中，可能为模型训练数据或编造。",
                    "quote": preview,
                })

        stats["elapsed_ms"] = (_time.perf_counter() - _t0) * 1000

        # 4. 有可疑或截断 → 在 md 文件末尾追加「来源校验」节
        if not warnings and not truncated:
            return {"stats": stats, "append_hint": ""}

        lines = ["\n\n---\n\n## 来源校验\n"]
        if truncated:
            lines.append(
                "> ⚠️ **完整性提示**：本报告因 Token 预算耗尽或循环达上限被迫封卷，"
                "部分章节可能未写完。如需完整内容请重新提问或缩小范围。\n\n"
            )
        if warnings:
            lines.append(f"> 引文逐字反查发现 {len(warnings)} 处可疑（共扫描 {stats['quotes_total']} 条引文）：\n")
            for w in warnings:
                lines.append(f"> - {w['message']}\n")
        else:
            lines.append(f"> 引文逐字反查全部通过（共 {stats['quotes_verified']} 条）。\n")

        append_hint = "\n".join(lines)
        # 写入 md 文件（追加，不修改已有正文）
        if report_id:
            try:
                from .mcp.tools.save_long_report import _md_path, _safe_id
                p = _md_path(_safe_id(user_id), report_id)
                if p.exists():
                    with p.open("a", encoding="utf-8") as f:
                        f.write(append_hint)
                    logger.debug(f"[P1-2] 来源校验节已追加到 {report_id}")
            except Exception as ex:
                logger.debug(f"[P1-2] 追加校验节到 md 文件失败: {ex}")

        return {"stats": stats, "append_hint": append_hint}

    async def _call_tool(self, tool_name: str, params: dict,
                         on_progress=None) -> ToolCallResult:
        """B-082: 调用 MCP 工具并返回结构化结果（委托 ToolDispatcher 统一调度）。

        user_id 注入 / tool_call_context / MCP 调用 / 结果截断统一收敛到 ToolDispatcher。
        summary 与 sse_events 是本类特有的工作区事件格式化（依赖 node_client），
        通过回调注入 ToolDispatcher，仍基于完整 raw 生成（截断前）。
        on_progress: 可选中途进度回调，透传到 handler（长耗时工具实时推送进度）。
        """
        try:
            return await self.dispatcher.dispatch(
                tool_name, params, caller="agent_loop",
                purpose=tool_name, user_id=self.user_id,
                upstream="react_loop", downstream="mcp_call_tool",
                summarizer=self._summarize_tool_result,
                sse_builder=self._build_tool_sse_events,
                on_progress=on_progress,
            )
        except Exception as e:
            logger.error(f"Agent Loop 工具调用失败 [{tool_name}]: {e}")
            raise  # fast fail

    async def _build_tool_sse_events(self, tool_name: str, result) -> list[dict]:
        """B-082: 为工作区驱动工具构建额外的 SSE 事件。

        如果工具返回了 error 字段，不生成任何工作区事件（fast fail）。
        """
        if isinstance(result, dict) and "error" in result:
            return []
        events = []
        workspace_event_map = {
            "annotate_passage": SSEEventType.ANNOTATION_DATA,
            "annotate_user_request": SSEEventType.ANNOTATION_DATA,
            "suggest_annotations": SSEEventType.ANNOTATION_SUGGEST,
            "align_paragraphs": SSEEventType.PARAGRAPH_ALIGNMENT,
            "count_occurrences": SSEEventType.STATS_DATA,
            "compare_passages": SSEEventType.COMPARISON_DATA,
            "compare_and_annotate": SSEEventType.COMPARISON_DATA,
            "search_literature": SSEEventType.SEARCH_RESULTS,
            "search_paragraphs": SSEEventType.SEARCH_RESULTS,
            # v19: 出场轨迹复用搜索结果列表的跳转按钮
            "entity_appearances": SSEEventType.SEARCH_RESULTS,
            # v17: 诗词检索和多实体共现复用搜索结果列表
            "search_poetry": SSEEventType.SEARCH_RESULTS,
            "multi_entity_cooccurrence": SSEEventType.SEARCH_RESULTS,
            "get_chapter_full_text": SSEEventType.TEXT_NAVIGATE,
            # v10 新增: 跨朝代对比 + 地图溯源工具 SSE 路由
            "dynasty_compare_elements": SSEEventType.DYNASTY_COMPARE,
            "get_location_traceability": SSEEventType.LOCATION_TRACEABILITY,
            "generate_mutual_construction": SSEEventType.MUTUAL_CONSTRUCTION,
            "set_map_dynasty_layer": SSEEventType.MAP_LAYER_UPDATE,
            # v11: 地图聚焦工具 SSE 路由
            "focus_map_location": SSEEventType.MAP_HIGHLIGHT,
            # U-16: 导出标注 SSE 路由
            "export_annotations": SSEEventType.EXPORT_READY,
        }
        event_type = workspace_event_map.get(tool_name)
        if event_type:
            formatted = await self._format_workspace_event(tool_name, event_type, result, node_client=self.node_client)
            if formatted is not None:
                events.append(formatted)

        # C-08: 如果 annotate_user_request 结果包含 marginalia，额外发送独立 MARGINALIA_DATA 事件
        if tool_name == "annotate_user_request" and isinstance(result, dict) and result.get("marginalia"):
            target = result.get("target", {})
            events.append({
                "type": SSEEventType.MARGINALIA_DATA,
                "data": {
                    "chapterNumber": target.get("chapter_number", 0),
                    "marginalia": result["marginalia"],
                }
            })

        return events

    async def _format_workspace_event(self, tool_name: str, event_type: str, result, node_client=None) -> dict:
        """将 MCP 工具输出转为前端 SSE EventBus 期望的数据格式。"""
        if event_type == SSEEventType.TEXT_NAVIGATE:
            return self._format_text_navigate(result)
        if event_type == SSEEventType.ANNOTATION_DATA:
            return await self._format_annotation_data(result, node_client=node_client)

        # v10: 需要展开结果到顶层的工具 (dynasty_compare_elements / generate_mutual_construction / set_map_dynasty_layer / focus_map_location)
        if tool_name in ("dynasty_compare_elements", "generate_mutual_construction", "set_map_dynasty_layer", "focus_map_location"):
            if isinstance(result, dict):
                return build_sse_event(event_type, **result)
            return build_sse_event(event_type, data=result)

        # U-16: 导出标注 — 将工具返回的 {chapter_numbers, format, source} 展开到 SSE 事件顶层
        if tool_name == "export_annotations":
            if isinstance(result, dict):
                return build_sse_event(event_type, **result)
            return build_sse_event(event_type, data=result)

        # v10: 需要将结果包装在 data 中的工具 (get_location_traceability)
        if tool_name == "get_location_traceability":
            return build_sse_event(event_type, data=result)

        # P0-fix: align_paragraphs — 展开返回字段到顶层，确保 camelCase 一致
        if tool_name == "align_paragraphs":
            if isinstance(result, dict):
                return build_sse_event(event_type, **result)
            return build_sse_event(event_type, data=result)

        # P0-fix: suggest_annotations — 展开 passageAnnotations 分组到顶层
        if tool_name == "suggest_annotations":
            if isinstance(result, dict) and "passageAnnotations" in result:
                return build_sse_event(event_type,
                    passageAnnotations=result["passageAnnotations"])
            return build_sse_event(event_type, data=result)

        # P1-fix: compare_and_annotate / compare_passages → comparison_data
        if event_type == SSEEventType.COMPARISON_DATA:
            if isinstance(result, dict):
                return build_sse_event(event_type, **result)
            return build_sse_event(event_type, data=result)

        # v19: entity_appearances — chapter_groups 摊平为前端 SearchResultItem 结构，
        # 复用搜索结果列表的分页 + 跳转按钮（出场位置可点击跳转原文）
        if tool_name == "entity_appearances" and isinstance(result, dict):
            flat = []
            rank = 0
            for g in result.get("chapter_groups", []):
                for app in g.get("appearances", []):
                    rank += 1
                    flat.append({
                        "rank": rank,
                        "chapterNumber": g.get("chapter_number", ""),
                        "chapterTitle": (
                            f"{g.get('chapter_title', '')}"
                            f"（{g.get('dynasty') or '研究文献'}）"
                        ),
                        "paragraphIndex": app.get("paragraph_index", 0),
                        "excerpt": app.get("excerpt", ""),
                        "highlightRanges": [],
                        "paragraphMatchPositions": [],
                        "relevanceScore": 1.0,
                        "isResearch": not g.get("chapter_number"),
                    })
            return build_sse_event(SSEEventType.SEARCH_RESULTS,
                query=f"『{result.get('entity', '')}』出场轨迹",
                totalMatches=result.get("total_occurrences", len(flat)),
                results=flat,
                sortOptions=["按章节"],
            )

        # v17: search_poetry — 韵文检索结果摊平为 SearchResultItem
        if tool_name == "search_poetry" and isinstance(result, dict):
            flat = []
            for rank, item in enumerate(result.get("results", []), 1):
                ch = item.get("chapter_number", "")
                flat.append({
                    "rank": rank,
                    "chapterNumber": ch,
                    "chapterTitle": (
                        f"{item.get('chapter_title', '')}"
                        f"（{item.get('dynasty') or '研究文献'}）"
                    ),
                    "paragraphIndex": item.get("paragraph_index", 0),
                    "excerpt": item.get("excerpt", ""),
                    "highlightRanges": [],
                    "paragraphMatchPositions": [],
                    "relevanceScore": 1.0,
                    "isResearch": not ch,
                })
            forms_str = result.get("form_type", "全部")
            return build_sse_event(SSEEventType.SEARCH_RESULTS,
                query=f"韵文检索{'·' + forms_str if forms_str != '全部' else ''}：{result.get('query', '')}",
                totalMatches=result.get("total", len(flat)),
                results=flat,
                sortOptions=["按章节"],
            )

        # v17: multi_entity_cooccurrence — 多实体共现摊平为 SearchResultItem
        if tool_name == "multi_entity_cooccurrence" and isinstance(result, dict):
            flat = []
            for rank, co in enumerate(result.get("cooccurrences", []), 1):
                ch = co.get("chapter_number", "")
                # excerpt 优先用合并的多实体上下文
                excerpt_parts = []
                for entity, hit in co.get("entity_hits", {}).items():
                    excerpt_parts.append(f"【{entity}×{hit.get('count', 0)}】{hit.get('excerpt', '')[:80]}")
                excerpt = " | ".join(excerpt_parts) if excerpt_parts else co.get("excerpt", "")
                flat.append({
                    "rank": rank,
                    "chapterNumber": ch,
                    "chapterTitle": (
                        f"{co.get('chapter_title', '')}"
                        f"（{co.get('dynasty') or '研究文献'}）"
                    ),
                    "paragraphIndex": co.get("paragraph_index", 0),
                    "excerpt": excerpt,
                    "highlightRanges": [],
                    "paragraphMatchPositions": [],
                    "relevanceScore": 1.0,
                    "isResearch": not ch,
                })
            entities_label = " + ".join(result.get("entities", []))
            return build_sse_event(SSEEventType.SEARCH_RESULTS,
                query=f"『{entities_label}』共现段落",
                totalMatches=result.get("total_cooccurrences", len(flat)),
                results=flat,
                sortOptions=["按章节"],
            )

        return {"type": event_type, "data": result}

    def _format_text_navigate(self, result) -> dict:
        """将 get_chapter_full_text 输出转为 text_navigate SSE 事件。

        工具返回: {chapter_number, title, dynasty, paragraphs, paragraph_count}
        前端期望: {chapterNumber, chapterTitle, paragraphIndex}

        如果 result 不含 chapter_number，返回 None，由调用方跳过。
        """
        if not isinstance(result, dict):
            return None
        ch_num = result.get("chapter_number")
        if ch_num is None:
            return None
        return {
            "type": SSEEventType.TEXT_NAVIGATE,
            "data": {
                "chapterNumber": ch_num,
                "chapterTitle": result.get("title", ""),
                "paragraphIndex": result.get("paragraph_index"),
            }
        }

    async def _format_annotation_data(self, result, node_client=None) -> dict:
        """将 annotate_passage / annotate_user_request 的输出转为 annotation_data SSE 事件格式。

        CU-01: 段落级拆分 — 将拼接全文坐标系下的标注偏移映射回各段落独立坐标系。
        MCP 输出: {target: {chapter_number, paragraph_range}, annotations: [{entity, start_char, end_char, ...}]}
        输出: {passageAnnotations: [{passageKey, paragraphIndex, annotations, insights}, ...]}
        """
        target = result.get("target", {})
        chapter_number = target.get("chapter_number", 0)
        para_range = target.get("paragraph_range", [0, 0])
        if isinstance(para_range, list) and len(para_range) >= 2:
            para_start, para_end = para_range[0], para_range[1]
        else:
            para_start = para_end = 0

        now_ts = int(__import__('time').time() * 1000)
        annotations_raw = result.get("annotations", [])

        # CU-01 Step 1: 获取段落文本，构建边界表
        # 使用与前端 sourceTreeStore.splitParagraphs 一致的切分规则（空行分隔）
        boundaries = []  # [{paraIdx, start, end}, ...] 相对拼接全文的偏移
        if node_client is not None:
            try:
                ch = await node_client.get_literature_by_chapter(chapter_number)
                if ch:
                    content = ch.get("content", "")
                    content = re.sub(r'\r\n|\r', '\n', content)
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    offset = 0
                    for para_idx in range(para_start, min(para_end + 1, len(paragraphs))):
                        para_text = paragraphs[para_idx]
                        boundaries.append({
                            "paraIdx": para_idx,
                            "start": offset,
                            "end": offset + len(para_text),
                        })
                        offset += len(para_text) + 2  # +2 for \n\n separator
            except Exception as e:
                logger.warning(f"_format_annotation_data: 无法获取段落文本用于边界计算: {e}")

        # CU-01 Step 2: 将每个标注映射到归属段落并计算相对偏移
        def resolve_paragraph(ann_start_char):
            for b in boundaries:
                if b["start"] <= ann_start_char < b["end"]:
                    return b["paraIdx"], ann_start_char - b["start"]
            # 边界不匹配: 尝试找到最近的段落（annotations 不应落在段落分隔符上）
            for b in boundaries:
                if b["start"] <= ann_start_char <= b["end"]:
                    rel = max(0, ann_start_char - b["start"])
                    logger.warning(
                        f"_format_annotation_data: annotation at {ann_start_char} "
                        f"lands on boundary of para {b['paraIdx']}, "
                        f"clamping to relative offset {rel}"
                    )
                    return b["paraIdx"], rel
            # Fallback: 范围第一段
            logger.warning(
                f"_format_annotation_data: annotation at {ann_start_char} "
                f"outside all paragraph boundaries [{boundaries[0]['start'] if boundaries else '?'}, "
                f"{boundaries[-1]['end'] if boundaries else '?'}), falling back to para {para_start}"
            )
            return para_start, ann_start_char

        # CU-01 Step 3: 按段落分组
        grouped: dict[int, dict] = {}
        for i, ann in enumerate(annotations_raw):
            start_char = ann.get("start_char", ann.get("span", {}).get("startChar", 0))
            end_char = ann.get("end_char", ann.get("span", {}).get("endChar", 0))
            entity = ann.get("entity", ann.get("text", ""))

            para_idx, rel_start = resolve_paragraph(start_char)
            rel_end = rel_start + (end_char - start_char) if end_char > start_char else rel_start + len(entity)

            frontend_ann = {
                "id": ann.get("id", f"agent_ann_{chapter_number}_{para_idx}_{i}"),
                "category": ann.get("category", "custom"),
                "label": entity,
                "span": {"startChar": rel_start, "endChar": rel_end},
                "explanation": ann.get("explanation", ""),
                "confidence": ann.get("confidence", "high"),
                "color": ann.get("color", "#F1C40F"),
                "source": ann.get("source", "agent"),
                "chapterNumber": chapter_number,
                "paragraphIndex": para_idx,
                "text": entity,
                "createdAt": now_ts,
                "updatedAt": now_ts,
            }

            if para_idx not in grouped:
                grouped[para_idx] = {"annotations": [], "insights": []}
            grouped[para_idx]["annotations"].append(frontend_ann)

        # CU-01 Step 4: 同样将 insights 归属到范围第一段（insight 是段落级以上的宏观分析）
        for i, ins in enumerate(result.get("insights", [])):
            if isinstance(ins, str):
                grouped.setdefault(para_start, {"annotations": [], "insights": []})
                grouped[para_start]["insights"].append({
                    "id": f"ins_{chapter_number}_{para_start}_{i}",
                    "text": ins,
                    "category": "motif",
                    "label": ins[:30],
                })
            elif isinstance(ins, dict):
                grouped.setdefault(para_start, {"annotations": [], "insights": []})
                grouped[para_start]["insights"].append({
                    "id": ins.get("id", f"ins_{chapter_number}_{para_start}_{i}"),
                    "text": ins.get("text", ins.get("explanation", "")),
                    "category": ins.get("category", "motif"),
                    "label": ins.get("label", str(ins.get("text", ""))[:30]),
                    "motifRef": ins.get("motifRef"),
                })

        # CU-01 Step 5: 构建 passageAnnotations 数组（替代旧平面结构）
        split_side = result.get("split_side", "")
        passage_key_suffix = f":{split_side}" if split_side else ""
        passage_annotations = []
        for para_idx in sorted(grouped.keys()):
            passage_annotations.append({
                "passageKey": f"{chapter_number}:{para_idx}{passage_key_suffix}",
                "paragraphIndex": para_idx,
                "annotations": grouped[para_idx]["annotations"],
                "insights": grouped[para_idx].get("insights", []),
            })

        return {
            "type": SSEEventType.ANNOTATION_DATA,
            "data": {
                "chapterNumber": chapter_number,
                "chapterTitle": target.get("chapter_title", ""),
                "paragraphRange": [para_start, para_end],
                "passageAnnotations": passage_annotations,
                "marginalia": result.get("marginalia", []),
                "splitSide": result.get("split_side"),
            }
        }

    def _summarize_tool_result(self, tool_name: str, result) -> str:
        """生成工具结果的简短摘要（degraded 显式标注，供 LLM 感知降级状态）。"""
        if isinstance(result, list):
            return f"找到 {len(result)} 条结果"
        elif isinstance(result, dict):
            if "error" in result:
                if result.get("degraded"):
                    return (f"部分降级错误（工具仍在运行但结果可能不完整，"
                            f"请向用户说明）: {result['error']}")
                return f"错误: {result['error']}"
            keys = list(result.keys())[:5]
            base = f"返回 {len(result)} 个字段: {', '.join(keys)}"
            if result.get("degraded"):
                base = f"[降级结果，可能不完整] {base}"
            return base
        return str(result)[:100]

    # ================================================================
    #  快路径: 快速模型预检 (U02-03)
    # ================================================================

    async def _try_fast_answer(self, query: str, budget: TokenBudget = None) -> Optional[dict]:
        """B-081: 快速检索 + 快模判断 + 新模式匹配。

        扩展快路径的匹配模式:
        - 统计类查询 -> 直接调用 count_occurrences
        - 对比类查询 -> 直接调用 compare_passages
        - 标注类查询 -> 直接调用 annotate_passage

        Args:
            budget: 可选 TokenBudget —— 快模调用也纳入预算记账（v16），不传则只执行不记账

        Returns:
            {"docs": [...], "sufficient": bool, "fast_action": str, "fast_tool": str} 或 None
        """
        if not self.retriever:
            return None

        try:
            # B-081: 快速模式匹配（规则 + 快模）
            from .config import settings as app_settings
            fast_model = app_settings.effective_fast_model

            # 意图快速检测
            intent_system = "你是查询意图分类器。只返回JSON: {\"intent\":\"statistics|comparison|annotation|general\"}"
            intent_user = f"查询: {query}"
            try:
                # thinking_disabled=True: 小 max_tokens 下思考会吃光预算致 content 为空
                with llm_call_context("agent_loop", "fast_intent",
                                      upstream="user_query", downstream="intent_rules"):
                    resp = await self.llm.generate(intent_system, intent_user, model=fast_model, max_tokens=64, temperature=0.1, thinking_disabled=True)
                if budget:
                    budget.consume(resp.completion_tokens)  # v16: 快模调用也纳入账本
                intent_data = parse_llm_json(resp.content)
                fast_intent = intent_data.get("intent", "general")
            except (json.JSONDecodeError, KeyError, AttributeError, ValueError):
                # 意图快速检测失败（纯优化路径，失败时回退到 general）
                fast_intent = "general"

            # 规则兜底: 如果查询包含系统上下文注入(chapter_number=) + 标注关键词
            # LLM 可能误判为 general，用规则强制纠正
            if fast_intent == "general":
                # 地图控制关键词（必须在标注检查之前，因为"地图标记"含"标记"）
                map_control_keywords = [
                    r"地图标记", r"地图图层", r"地图朝代", r"显示.*朝",
                    r"只看.*朝", r"查看.*朝.*标记", r"切换图层",
                    r"显示所有朝代", r"全部朝代.*标记", r"过滤.*朝代",
                    r"朝代.*地图", r"地图.*过滤",
                ]
                has_map_kw = any(re.search(kw, query) for kw in map_control_keywords)
                if has_map_kw:
                    fast_intent = "map_control"

            if fast_intent == "general":
                # 导出关键词必须在标注检查之前——"导出标注"含"标注"但语义是导出
                export_keywords = ["导出", "下载", "保存.*标注", "标注.*导出", "标注.*下载"]
                if any(re.search(kw, query) for kw in export_keywords):
                    fast_intent = "export"

            if fast_intent == "general":
                annotation_keywords = ["标注", "划线", "高亮", "标记", "批注", "标出来", "标一下"]
                has_annotation_kw = any(kw in query for kw in annotation_keywords)
                has_chapter_ctx = "chapter_number=" in query
                if has_annotation_kw and has_chapter_ctx:
                    fast_intent = "annotation"
                elif has_annotation_kw:
                    fast_intent = "annotation"

            if fast_intent == "general":
                mutual_construction_keywords = ["互构", "景观-文本", "景观与文本", "双向溯源"]
                if any(kw in query for kw in mutual_construction_keywords):
                    fast_intent = "mutual_construction"

            docs = await self.retriever.retrieve(query)
            if not docs:
                return {"docs": [], "sufficient": False, "fast_intent": fast_intent}

            # 用快速模型判断充足性
            docs_preview = "\n".join(
                f"[{d.get('metadata', {}).get('chapter_title', '')}] {d.get('content', '')[:200]}"
                for d in docs[:3]
            )
            system = "你是信息充足性评估助手。只返回 JSON: {\"sufficient\": true/false}"
            user = f"用户问题：{query}\n\n检索到的文献：\n{docs_preview}\n\n这些文献是否足以充分回答用户问题？"

            # thinking_disabled=True: 同上，防止思考吃光小预算致空响应
            with llm_call_context("agent_loop", "assess_docs_sufficient",
                                  upstream="retriever.search", downstream="_try_fast_answer"):
                resp = await self.llm.generate(system, user, model=fast_model, max_tokens=64, temperature=0.1, thinking_disabled=True)
            if budget:
                budget.consume(resp.completion_tokens)  # v16: 快模调用也纳入账本
            data = parse_llm_json(resp.content)
            return {
                "docs": docs,
                "sufficient": data.get("sufficient", False),
                "fast_intent": fast_intent,
            }
        except (json.JSONDecodeError, KeyError) as e:
            # LLM 返回格式异常（非关键——快路径失败不影响 ReAct 主循环）
            logger.warning(f"_try_fast_answer 降级（非关键）: {e}")
            return None
        except (ValueError, ConnectionError, RuntimeError) as e:
            # retriever 不可达（非关键）
            logger.warning(f"_try_fast_answer 检索失败（非关键）: {e}")
            return None

    # ================================================================
    #  充足性评估 (U02-05)
    # ================================================================

    async def _assess_sufficiency(self, query: str, messages: list[dict], docs_found: int) -> tuple[bool, str]:
        """B-083: 快速模型评估——扩展维度。

        新增评估维度:
        - 是否有足够证据回答统计问题？
        - 对比分析是否覆盖了所有指定维度？
        """
        from .config import settings as app_settings
        fast_model = app_settings.effective_fast_model

        conversation = "\n".join(
            f"[{m['role']}]: {str(m.get('content', ''))[:200]}" for m in messages[-6:]
        )
        system = (
            "你是信息充足性评估助手。评估以下维度:\n"
            "1. 是否收集了足够的文献证据？\n"
            "2. 如果是统计问题，是否有定量数据？\n"
            "3. 如果是对比问题，是否覆盖了所有维度？\n"
            "只返回 JSON: {\"sufficient\": true/false, \"follow_up\": \"如果不足的建议检索查询\", "
            "\"missing_dimensions\": [\"缺失维度\"]}"
        )
        user = f"用户原问题：{query}\n\n已收集信息 ({docs_found}条):\n{conversation}\n\n信息是否足以回答用户问题？"

        try:
            # max_tokens=256 (原128): system prompt 要求输出 sufficient+follow_up+missing_dimensions，
            # 128 会被截断产生非法 JSON（曾出现 "Unterminated string" 降级）。关思考防预算被吃。
            with llm_call_context("agent_loop", "assess_sufficiency",
                                  upstream="react_loop", downstream="_should_continue"):
                resp = await self.llm.generate(system, user, model=fast_model, max_tokens=256, temperature=0.1, thinking_disabled=True)
            data = parse_llm_json(resp.content)
            return data.get("sufficient", True), data.get("follow_up", "")
        except (json.JSONDecodeError, KeyError) as e:
            # LLM 返回格式异常（非关键——评估失败假设充足，避免无限循环）
            logger.warning(f"_assess_sufficiency 降级: {e}")
            return True, ""

    # ================================================================
    #  上下文护栏 (v16) — 控制输入上下文膨胀
    # ================================================================

    def _apply_context_guardrail(self, messages: list[dict]) -> int:
        """messages 总 token 超阈值时，把早期工具结果压缩为一行占位摘要。

        只统计 LLM 输出 token 的旧预算管不住输入膨胀（输入才是费用与首字延迟的大头），
        此护栏按 tiktoken 近似估算 messages 总量，超限时仅保留最近
        _GUARDRAIL_KEEP_RECENT 条工具消息完整，其余替换为占位摘要。
        Returns: 被压缩的消息条数。
        """
        from .config import settings as app_settings
        from .context.token_counter import count_messages_tokens

        threshold = app_settings.context_guardrail_tokens
        if count_messages_tokens(messages) <= threshold:
            return 0

        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        keep = set(tool_idx[-_GUARDRAIL_KEEP_RECENT:])
        # 关键工具结果不压缩——它们的输出是后续分析的核心素材
        PROTECTED_TOOLS = {"dynasty_compare_elements", "get_chapter_full_text", "find_shared_passages"}
        compressed = 0
        for i in tool_idx:
            if i in keep:
                continue
            m = messages[i]
            # 检查是否为受保护工具的结果
            tool_name = m.get("tool_name", "")
            if tool_name in PROTECTED_TOOLS:
                continue
            content = m.get("content", "")
            original_len = len(content)
            # 智能截断：保留前 800 字符 + 后 200 字符，中间用省略号连接
            # 比完全丢弃好得多——LLM 仍能看到工具结果的开头（通常含摘要/结论）和结尾
            if original_len > 1000:
                head = content[:800]
                tail = content[-200:]
                messages[i] = {
                    **m,
                    "content": f"{head}\n...[已截断: 原 {original_len} 字符]...\n{tail}",
                }
                compressed += 1
            # 短于 1000 字符的不压缩
        if compressed:
            logger.info(f"[ContextGuardrail] 压缩 {compressed} 条早期工具结果 (threshold={threshold})")
        return compressed

    # ================================================================
    #  ReAct 主循环 — 非流式 (U02-02)
    # ================================================================

    async def run(self, query: str, history: list = None,
                  skill: str = None, thread_id: str = "default") -> dict:
        """非流式 ReAct 执行。"""
        from .config import settings as app_settings
        budget = TokenBudget()
        system = self._build_system_prompt(skill)
        tools = self._load_tool_definitions()

        # 构建初始 messages
        messages = []
        if history:
            for h in history:
                role = h.get("role", "user")
                messages.append({"role": role, "content": h.get("content", "")})
        messages.append({"role": "user", "content": query})

        all_sources = []
        tool_calls_record = []

        # Step 1: 尝试快路径
        fast_result = await self._try_fast_answer(query)
        fast_intent = fast_result.get("fast_intent", "general") if fast_result else "general"

        # CU-annotation: 标注/统计/对比意图必须进入 ReAct 循环以调用工具产生可视化结果
        # 快路径只生成文本回答，无法产生前端可见的标注/图表/对比视图
        ACTION_INTENTS = {"annotation", "comparison", "statistics", "map_control", "mutual_construction", "export"}

        if fast_result and fast_result["sufficient"] and fast_intent not in ACTION_INTENTS:
            docs = fast_result["docs"]
            docs_text = "\n\n".join(
                f"[{d.get('metadata', {}).get('chapter_title', '')}] {d.get('content', '')[:500]}"
                for d in docs
            )
            user_prompt = f"用户问题：{query}\n\n相关文献：\n{docs_text}\n\n请基于文献简洁回答，引用具体章节。"
            try:
                with llm_call_context("agent_loop", "generate_answer",
                                      upstream="retriever.docs", downstream="final_answer"):
                    resp = await self.llm.generate(system, user_prompt,
                                                   max_tokens=app_settings.max_output_tokens, temperature=0.5)
                budget.consume(resp.completion_tokens)
                # 长度截断告警
                if resp.finish_reason == "length":
                    logger.warning(f"[AgentLoop] 快路径最终回答因长度上限被截断 (max_tokens={app_settings.max_output_tokens}, completion={resp.completion_tokens})")
                    resp._truncated_hint = True
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                logger.error(f"Agent Loop 快路径 LLM 失败: {e}")
                raise

            for d in docs[:5]:
                meta = d.get("metadata", {})
                all_sources.append({
                    "chapter_number": meta.get("chapter_number", ""),
                    "chapter_title": meta.get("chapter_title", ""),
                    "excerpt": d.get("content", "")[:200],
                    "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
                })
            return {
                "answer": resp.content,
                "sources": all_sources,
                "workflow": "agent_loop",
                "tool_calls": tool_calls_record,
            }

        # Step 2: ReAct 循环
        iteration = 0
        last_tool_calls = []  # 用于检测重复调用

        while iteration < MAX_REACT_ITERATIONS and not budget.is_exhausted:
            iteration += 1
            try:
                with llm_call_context("agent_loop", "react_tool_round",
                                      upstream="react_loop", downstream="tool_exec"):
                    result = await self.llm.generate_with_tools(
                        system=system, messages=messages, tools=tools,
                        max_tokens=app_settings.max_output_tokens, temperature=0.5,
                        thinking_disabled=False,  # v17: ReAct 决策轮保留推理（唯一显式开思考的入口）
                    )
                # B-167: 排除 thinking 模式的 reasoning_tokens（思维链）
                budget.consume(result.completion_tokens - getattr(result, "reasoning_tokens", 0))
                if result.finish_reason == "length":
                    logger.warning(
                        f"[AgentLoop] ReAct 决策轮因长度上限被截断 "
                        f"(max_tokens={app_settings.max_output_tokens}, completion={result.completion_tokens})"
                    )
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                logger.error(f"Agent Loop ReAct LLM 失败 (iter {iteration}): {e}")
                raise

            # 情况 A: LLM 选择直接回答
            if not result.tool_calls and result.text:
                for d in fast_result.get("docs", [])[:5] if fast_result else []:
                    meta = d.get("metadata", {})
                    all_sources.append({
                        "chapter_number": meta.get("chapter_number", ""),
                        "chapter_title": meta.get("chapter_title", ""),
                        "excerpt": d.get("content", "")[:200],
                        "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
                    })
                # v19: 先清洗技术泄露，再做 P1-1 来源校验
                clean_answer = sanitize_answer_text(result.text)
                verified_answer = await self._post_verify_citations(clean_answer, all_sources)
                return {
                    "answer": verified_answer,
                    "sources": all_sources,
                    "workflow": "agent_loop",
                    "tool_calls": tool_calls_record,
                }

            # 情况 B: LLM 选择调用工具
            # 协议: 先追加一条包含本轮全部 tool_calls 的 assistant 消息，
            # 随后每个工具结果各追加一条 tool 消息（顺序不可颠倒）
            messages.append(result.to_assistant_message())
            for tc in result.tool_calls:
                # 检测重复调用
                dup_key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
                last_tool_calls.append(dup_key)
                recent_dups = last_tool_calls[-DUPLICATE_TOOL_THRESHOLD:]
                if len(recent_dups) >= DUPLICATE_TOOL_THRESHOLD and len(set(recent_dups)) == 1:
                    logger.warning(f"检测到重复工具调用 ({tc.name})，强制结束 ReAct 循环")
                    iteration = MAX_REACT_ITERATIONS  # 强制退出
                    break

                tool_result = await self._call_tool(tc.name, tc.arguments)
                tool_calls_record.append({
                    "tool": tc.name,
                    "input": tc.arguments,
                    "status": "completed" if tool_result.success else "error",
                    "result_preview": tool_result.summary,
                })

                # 每个工具结果一条 tool 消息（assistant 消息已在循环前统一追加）
                # tool_name 字段供 ContextGuardrail 识别受保护工具，不影响 LLM API 调用
                tool_msg = {"role": "tool", "tool_call_id": tc.id,
                           "tool_name": tc.name,
                           "content": json.dumps(tool_result.result, ensure_ascii=False) if tool_result.success else tool_result.error}
                messages.append(tool_msg)

            # 充足性评估
            total_docs = sum(1 for m in messages if m.get("role") == "tool")
            sufficient, follow_up = await self._assess_sufficiency(query, messages, total_docs)
            if sufficient:
                break
            if follow_up:
                messages.append({"role": "user", "content": f"请进一步检索: {follow_up}"})

        # Step 3: 综合所有信息，生成最终答案
        messages.append({"role": "user", "content": "请基于以上所有收集到的信息，给出完整、有引用来源的回答。"})
        try:
            with llm_call_context("agent_loop", "generate_answer",
                                  upstream="react_loop", downstream="final_answer"):
                resp = await self.llm.generate(system,
                                               "\n".join(f"[{m['role']}]: {str(m.get('content', ''))[:2000]}" for m in messages),
                                               max_tokens=app_settings.max_output_tokens, temperature=0.5)
            budget.consume(resp.completion_tokens)
        except (LLMFatalError, RateLimitError, APITimeoutError) as e:
            logger.error(f"Agent Loop 最终生成失败: {e}")
            raise

        # 收集来源
        for d in (fast_result.get("docs", []) if fast_result else [])[:5]:
            meta = d.get("metadata", {})
            all_sources.append({
                "chapter_number": meta.get("chapter_number", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
            })

        # v19: 先清洗技术泄露，再做 P1-1 来源校验
        clean_answer = sanitize_answer_text(resp.content)
        verified_answer = await self._post_verify_citations(clean_answer, all_sources)
        return {
            "answer": verified_answer,
            "sources": all_sources,
            "workflow": "agent_loop",
            "tool_calls": tool_calls_record,
            "iterations": iteration,
        }

    # ================================================================
    #  ReAct 主循环 — 流式 (U02-02, U02-06)
    # ================================================================

    async def run_stream(self, query: str, history: list = None,
                         skill: str = None, thread_id: str = "stream") -> AsyncGenerator[dict, None]:
        """流式 ReAct 执行（公共入口）——兜底包装：任何未处理异常都转为 ERROR 事件，
        绝不允许流静默中断（2026-09-02 事故：底层异常逃逸导致用户侧无回答）。"""
        try:
            async for event in self._run_stream_inner(query, history, skill, thread_id):
                yield event
        except Exception as e:
            logger.exception(f"[AgentLoop] run_stream 未捕获异常: {e}")
            # B-165: 异常消息为空时补类型名，避免前端弹窗显示空括号"()"
            err_detail = str(e)[:120] or type(e).__name__
            yield {"type": SSEEventType.ERROR,
                   "data": {"message": f"分析过程意外中断（{err_detail}）。请重试一次，或换个更具体的问法。"}}
            yield {"type": SSEEventType.DONE,
                   "data": {"workflow": "agent_loop", "thread_id": thread_id, "interrupted": True}}

    async def _run_stream_inner(self, query: str, history: list = None,
                         skill: str = None, thread_id: str = "stream") -> AsyncGenerator[dict, None]:
        """流式 ReAct 执行 — 产出 tool_call/tool_result/chunk/source/done 事件。"""
        from .config import settings as app_settings
        budget = TokenBudget()
        system = self._build_system_prompt(skill)
        tools = self._load_tool_definitions()

        # 构建初始 messages
        messages = []
        if history:
            for h in history:
                role = h.get("role", "user")
                messages.append({"role": role, "content": h.get("content", "")})
        messages.append({"role": "user", "content": query})

        all_sources = []

        # Step 1: 尝试快路径
        yield {"type": SSEEventType.STATUS, "data": "正在检索文献..."}
        fast_result = await self._try_fast_answer(query, budget=budget)
        fast_intent = fast_result.get("fast_intent", "general") if fast_result else "general"

        # CU-annotation: 标注/统计/对比意图必须进入 ReAct 循环以调用工具产生可视化结果
        ACTION_INTENTS = {"annotation", "comparison", "statistics", "map_control", "mutual_construction", "export"}

        if fast_result and fast_result["sufficient"] and fast_intent not in ACTION_INTENTS:
            yield {"type": SSEEventType.STATUS, "data": f"快速检索完成：{len(fast_result['docs'])} 段文献已足够回答"}
            docs = fast_result["docs"]
            docs_text = "\n\n".join(
                f"[{d.get('metadata', {}).get('chapter_title', '')}] {d.get('content', '')[:500]}"
                for d in docs
            )
            user_prompt = f"用户问题：{query}\n\n相关文献：\n{docs_text}\n\n请基于文献简洁回答，引用具体章节。"
            try:
                with llm_call_context("agent_loop", "generate_answer",
                                      upstream="retriever.docs", downstream="final_answer"):
                    fast_chunks = 0
                    stream = await self.llm.generate_stream(
                        system, user_prompt, max_tokens=app_settings.max_output_tokens,
                        temperature=0.5, thinking_disabled=True,
                    )
                    async for text in stream:
                        if text and text.strip():
                            fast_chunks += 1
                        yield {"type": SSEEventType.CHUNK, "data": text}
                    # v17.2: finish_reason == length → 回答末尾追加可见提示
                    if stream.finish_reason == "length":
                        hint = (
                            "\n\n> **注**：本回答内容较多，已达生成长度上限被截断。"
                            "回复「生成完整报告」，我会把全部内容完整写入报告文件供您下载（不压缩、不省略）。"
                        )
                        logger.warning(
                            f"[AgentLoop] 快路径流式最终回答因长度上限被截断 "
                            f"(max_tokens={app_settings.max_output_tokens})"
                        )
                        yield {"type": SSEEventType.CHUNK, "data": hint}
                    if fast_chunks == 0:
                        yield {"type": SSEEventType.CHUNK, "data": "抱歉，回答生成失败（模型返回空内容），请稍后重试。"}
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                logger.error(f"Agent Loop 快路径流式 LLM 失败: {e}")
                yield {"type": SSEEventType.ERROR, "data": {"message": str(e)}}
                return

            for d in docs[:5]:
                meta = d.get("metadata", {})
                yield {"type": SSEEventType.SOURCE, "data": {
                    "chapter_number": meta.get("chapter_number", ""),
                    "chapter_title": meta.get("chapter_title", ""),
                    "excerpt": d.get("content", "")[:200],
                    "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
                }}
            yield {"type": SSEEventType.DONE, "data": {"workflow": "agent_loop", "thread_id": thread_id}}
            return

        # Step 2: ReAct 循环
        if fast_intent in ACTION_INTENTS:
            action_label = {"annotation": "标注", "comparison": "对比", "statistics": "统计", "map_control": "地图控制", "mutual_construction": "景观互构分析", "export": "导出"}.get(fast_intent, fast_intent)
            yield {"type": SSEEventType.STATUS, "data": f"检测到{action_label}意图，启动 ReAct 工具调用 (最多 {MAX_REACT_ITERATIONS} 轮)..."}
        else:
            yield {"type": SSEEventType.STATUS, "data": f"快检索不足，启动 ReAct 深度检索 (最多 {MAX_REACT_ITERATIONS} 轮)..."}
        iteration = 0
        last_tool_calls = []
        full_answer = ""  # v19: 累积流式最终答案全文，供来源校验后处理使用
        # B-167: 追踪本会话 create 但未 finalize 的 report_id，
        # 循环结束时兜底封卷（LLM 偶尔会漏调 finalize）
        pending_report_ids: list[str] = []
        # v21: 收集 ReAct 检索工具返回的 sources，供报告引文反查快速路径使用
        react_sources: list[dict] = []

        while iteration < MAX_REACT_ITERATIONS and not budget.is_exhausted:
            iteration += 1

            if budget.warning_ratio > 0.8:
                yield {"type": SSEEventType.STATUS, "data": f"Token 预算已用 {budget.used}/{budget.limit}，加速收敛..."}

            # v17: 流式 function calling — 决策轮思考增量经队列桥接实时透传，
            # 前端显示"思考中..."折叠区（用户不再觉得卡住）
            thinking_queue: asyncio.Queue = asyncio.Queue()

            async def _on_reasoning(delta: str):
                await thinking_queue.put(delta)

            try:
                with llm_call_context("agent_loop", "react_tool_round",
                                      upstream="react_loop", downstream="tool_exec"):
                    llm_task = asyncio.create_task(self.llm.generate_with_tools_stream(
                        system=system, messages=messages, tools=tools,
                        max_tokens=app_settings.max_output_tokens, temperature=0.5,
                        thinking_disabled=False,  # v17: ReAct 决策轮保留推理（唯一显式开思考的入口）
                        on_reasoning=_on_reasoning,
                    ))
                    # 桥接: LLM 思考期间持续把 reasoning delta 转为 SSE thinking_delta
                    while not llm_task.done():
                        try:
                            r_delta = await asyncio.wait_for(thinking_queue.get(), timeout=0.05)
                            yield {"type": SSEEventType.THINKING_DELTA, "data": {"delta": r_delta}}
                        except asyncio.TimeoutError:
                            continue
                    while not thinking_queue.empty():
                        r_delta = thinking_queue.get_nowait()
                        yield {"type": SSEEventType.THINKING_DELTA, "data": {"delta": r_delta}}
                    result = llm_task.result()
                # B-167: 排除 thinking 模式的 reasoning_tokens（思维链），
                # 只把"可见输出"计入预算——否则多轮 ReAct 决策的思考会凭空榨干
                # MAX_TOKENS_BUDGET，导致 LLM 被迫提前停止写报告。
                budget.consume(result.completion_tokens - getattr(result, "reasoning_tokens", 0))
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                logger.error(f"Agent Loop ReAct 流式 LLM 失败 (iter {iteration}): {e}")
                yield {"type": SSEEventType.ERROR, "data": {"message": str(e)}}
                return

            # 情况 A: LLM 选择直接回答
            if not result.tool_calls and result.text:
                yield {"type": SSEEventType.STATUS, "data": f"ReAct 第{iteration}轮: AI 判断信息充足，开始回答"}
                clean_text = sanitize_answer_text(result.text)
                yield {"type": SSEEventType.CHUNK, "data": clean_text}
                # v17.2: 决策轮 finish_reason == length → 用户可见提示
                if getattr(result, "finish_reason", None) == "length":
                    hint = (
                        "\n\n> **注**：本回答内容较多，已达生成长度上限被截断。"
                        "回复「生成完整报告」，我会把全部内容完整写入报告文件供您下载（不压缩、不省略）。"
                    )
                    logger.warning(
                        f"[AgentLoop] ReAct 流式决策轮最终回答因长度上限被截断 "
                        f"(max_tokens={app_settings.max_output_tokens})"
                    )
                    yield {"type": SSEEventType.CHUNK, "data": hint}
                break

            # 情况 B: LLM 调用工具
            # 协议: 先追加一条包含本轮全部 tool_calls 的 assistant 消息，
            # 随后每个工具结果各追加一条 tool 消息（顺序不可颠倒）
            messages.append(result.to_assistant_message())
            for tc in result.tool_calls:
                # 检测重复调用
                dup_key = (tc.name, json.dumps(tc.arguments, sort_keys=True))
                last_tool_calls.append(dup_key)
                recent_dups = last_tool_calls[-DUPLICATE_TOOL_THRESHOLD:]
                if len(recent_dups) >= DUPLICATE_TOOL_THRESHOLD and len(set(recent_dups)) == 1:
                    logger.warning(f"检测到重复工具调用 ({tc.name})，强制结束 ReAct")
                    yield {"type": SSEEventType.STATUS, "data": f"检测到重复调用 {tc.name}，跳过并进入回答阶段"}
                    iteration = MAX_REACT_ITERATIONS
                    break

                yield {"type": SSEEventType.TOOL_CALL, "data": {
                    "tool": tc.name,
                    "arguments": tc.arguments,
                }}

                # v18: 长耗时工具中途进度推送 — 复用 v17 thinking_queue 桥接模式，
                # 工具内部 progress_callback 产出的消息经队列实时转为 SSE status 事件
                progress_queue: asyncio.Queue = asyncio.Queue()

                async def _on_progress(message: str):
                    await progress_queue.put(message)

                tool_task = asyncio.create_task(
                    self._call_tool(tc.name, tc.arguments, on_progress=_on_progress)
                )
                while not tool_task.done():
                    try:
                        p_msg = await asyncio.wait_for(progress_queue.get(), timeout=0.05)
                        yield {"type": SSEEventType.STATUS, "data": p_msg}
                    except asyncio.TimeoutError:
                        continue
                while not progress_queue.empty():
                    p_msg = progress_queue.get_nowait()
                    yield {"type": SSEEventType.STATUS, "data": p_msg}
                tool_result = tool_task.result()
                yield {"type": SSEEventType.TOOL_RESULT, "data": {
                    "tool": tc.name,
                    "summary": tool_result.summary,
                    "success": tool_result.success,
                }}

                # B-066: 产出工作区驱动 SSE 事件 (annotation_data / stats_data / comparison_data / search_results)
                for sse_event in tool_result.sse_events:
                    yield sse_event

                # v19: 长报告 finalize → 下发报告卡片事件（前端渲染下载卡片）
                # v19.2: 必须用 full_result（未截断）—— result 是 LLM context 副本，
                # content 被截断会导致前端下载/持久化的报告内容缺失
                if tc.name == "save_long_report" and isinstance(tool_result.result, dict):
                    r = tool_result.result
                    # B-167: 追踪 create 产生的 report_id（finalize 兜底用）
                    if r.get("reportId") and tc.arguments.get("action") == "create":
                        pending_report_ids.append(r["reportId"])
                    # finalize 成功 → 下发报告卡片，并从待封卷列表移除
                    if r.get("reportReady"):
                        rid = r.get("reportId")
                        if rid and rid in pending_report_ids:
                            pending_report_ids.remove(rid)
                        payload = tool_result.full_result \
                            if isinstance(tool_result.full_result, dict) else r
                        yield {"type": SSEEventType.REPORT_READY, "data": payload}
                        # v21: 报告封卷后异步校验引文（不阻塞 SSE 流）
                        if isinstance(payload, dict) and payload.get("content"):
                            try:
                                vfy = await self._verify_report_citations(
                                    user_id, payload.get("reportId", ""),
                                    payload["content"], react_sources,
                                )
                                if vfy.get("append_hint"):
                                    logger.info(
                                        f"[P1-2] report_citation 校验: "
                                        f"quotes={vfy['stats']['quotes_total']}"
                                        f"(verified={vfy['stats']['quotes_verified']},"
                                        f"suspicious={vfy['stats']['quotes_suspicious']}) "
                                        f"({vfy['stats']['elapsed_ms']:.0f}ms)"
                                    )
                            except Exception as ex:
                                logger.debug(f"[P1-2] report_citation 校验异常: {ex}")

                # v21: 收集 search 类工具返回的 sources（引文反查快速路径用）
                if tc.name in ("search_paragraphs", "search_literature", "search_poetry") \
                        and isinstance(tool_result.result, dict):
                    for r in (tool_result.result.get("results") or tool_result.result.get("sources") or [])[:8]:
                        if isinstance(r, dict):
                            react_sources.append({
                                "chapter_number": r.get("chapterNumber") or r.get("chapter_number", ""),
                                "chapter_title": r.get("chapterTitle") or r.get("chapter_title", ""),
                                "excerpt": r.get("excerpt", ""),
                            })

                # 每个工具结果一条 tool 消息（assistant 消息已在循环前统一追加）
                # tool_name 字段供 ContextGuardrail 识别受保护工具，不影响 LLM API 调用
                tool_msg = {"role": "tool", "tool_call_id": tc.id,
                           "tool_name": tc.name,
                           "content": json.dumps(tool_result.result, ensure_ascii=False) if tool_result.success else tool_result.error}
                messages.append(tool_msg)

            # v16 上下文护栏: 工具结果累积后估算 messages 总量，超阈值压缩早期工具结果
            self._apply_context_guardrail(messages)

            # v16: 充足性判断收敛给 LLM（system prompt 已要求"信息足够即直接回答"），
            # 不再用外部快模评估器每轮复核 —— 消除双头决策与每轮一次的隐形快模调用
            if budget.is_exhausted:
                yield {"type": SSEEventType.STATUS, "data": "Token 预算耗尽，基于已有信息回答"}
                break

        # B-167: 兜底封卷——LLM 写完所有节后偶尔会漏调 finalize 直接进入回答，
        # 此处扫描 pending_report_ids，对未封卷的报告自动调 finalize 并下发卡片
        # v21: 兜底封卷意味着 LLM 因 token 预算耗尽或循环达上限被迫停止，
        # 报告可能不完整 → 校验时标注"可能因 token 限制截断"
        budget_exhausted = budget.is_exhausted
        for rid in list(pending_report_ids):
            try:
                fr = await self._call_tool("save_long_report", {
                    "action": "finalize", "report_id": rid,
                })
                if isinstance(fr.result, dict) and fr.result.get("reportReady"):
                    payload = fr.full_result if isinstance(fr.full_result, dict) else fr.result
                    yield {"type": SSEEventType.REPORT_READY, "data": payload}
                    logger.info(f"[AgentLoop] 兜底封卷成功: {rid} ({fr.result.get('sectionCount', '?')} 节)")
                    # v21: 兜底封卷的报告可能不完整，校验并标注截断风险
                    if isinstance(payload, dict) and payload.get("content"):
                        try:
                            vfy = await self._verify_report_citations(
                                user_id, rid, payload["content"], react_sources,
                                truncated=budget_exhausted,
                            )
                            if vfy.get("append_hint"):
                                logger.info(
                                    f"[P1-2] report_citation(兜底) 校验: "
                                    f"quotes={vfy['stats']['quotes_total']}"
                                    f"(verified={vfy['stats']['quotes_verified']},"
                                    f"suspicious={vfy['stats']['quotes_suspicious']}) "
                                    f"({vfy['stats']['elapsed_ms']:.0f}ms)"
                                )
                        except Exception as ex:
                            logger.debug(f"[P1-2] report_citation(兜底) 校验异常: {ex}")
            except Exception as e:
                logger.warning(f"[AgentLoop] 兜底封卷失败 {rid}: {e}")

        # Step 3: 流式生成最终答案
        # 如果 ReAct 循环中已经直接回答了 (情况 A)，跳过
        if not result or result.tool_calls:
            yield {"type": SSEEventType.STATUS, "data": "综合所有信息，生成最终答案..."}
            messages.append({"role": "user", "content": "请基于以上所有收集到的信息，给出完整、有引用来源的回答。"})
            conversation = "\n".join(
                f"[{m['role']}]: {str(m.get('content', ''))[:2000]}" for m in messages
            )
            try:
                with llm_call_context("agent_loop", "generate_answer",
                                      upstream="react_loop", downstream="final_answer"):
                    answer_chunks = 0
                    answer_sanitizer = StreamSanitizer()  # v19: 流式清洗技术泄露
                    full_answer_parts: list[str] = []
                    stream = await self.llm.generate_stream(
                        system, conversation,
                        max_tokens=app_settings.max_output_tokens, temperature=0.5, thinking_disabled=True,
                    )
                    async for text in stream:
                        if text and text.strip():
                            answer_chunks += 1
                        if text:
                            full_answer_parts.append(text)
                            cleaned = answer_sanitizer.feed(text)
                            if cleaned:
                                yield {"type": SSEEventType.CHUNK, "data": cleaned}
                    flush_text = answer_sanitizer.flush()
                    if flush_text:
                        yield {"type": SSEEventType.CHUNK, "data": flush_text}
                    full_answer = "".join(full_answer_parts)
                    # v17.2: finish_reason == length → 用户可见提示
                    if stream.finish_reason == "length":
                        hint = (
                            "\n\n> **注**：本回答内容较多，已达生成长度上限被截断。"
                            "回复「生成完整报告」，我会把全部内容完整写入报告文件供您下载（不压缩、不省略）。"
                        )
                        logger.warning(
                            f"[AgentLoop] 流式最终答案因长度上限被截断 "
                            f"(max_tokens={app_settings.max_output_tokens})"
                        )
                        yield {"type": SSEEventType.CHUNK, "data": hint}
                    if answer_chunks == 0:
                        # LLM 返回空回答：不允许沉默，给出基于工具执行情况的兜底说明
                        logger.warning("[AgentLoop] 最终答案生成返回空文本，使用兜底回答")
                        yield {"type": SSEEventType.CHUNK, "data": (
                            "抱歉，最终回答生成失败（模型返回了空内容）。"
                            f"本次分析共执行了 {iteration} 轮工具调用。"
                            "请稍后重试，或基于上方工具执行结果中的数据自行查看；"
                            "若问题涉及文献对比，也可直接在跨朝代对读面板中查看结构化结果。"
                        )}
            except (LLMFatalError, RateLimitError, APITimeoutError) as e:
                logger.error(f"Agent Loop 流式最终生成失败: {e}")
                yield {"type": SSEEventType.ERROR, "data": {"message": str(e)}}
                return

        # P1-1: 答案来源自动校验后处理（流式版：有告警则追加 CHUNK）
        # v19: 流式答案已累积到 full_answer / result.text，校验后追加告警块
        collected_sources_for_verify = []
        for d in (fast_result.get("docs", []) if fast_result else [])[:5]:
            meta = d.get("metadata", {})
            collected_sources_for_verify.append({
                "chapter_number": meta.get("chapter_number", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
            })
        answer_for_verify = full_answer if full_answer else (result.text if result and result.text else "")
        if answer_for_verify:
            verified = await self._post_verify_citations(answer_for_verify, collected_sources_for_verify)
            if len(verified) > len(answer_for_verify):
                yield {"type": SSEEventType.CHUNK, "data": verified[len(answer_for_verify):]}

        # 来源
        for d in (fast_result.get("docs", []) if fast_result else [])[:5]:
            meta = d.get("metadata", {})
            yield {"type": SSEEventType.SOURCE, "data": {
                "chapter_number": meta.get("chapter_number", ""),
                "chapter_title": meta.get("chapter_title", ""),
                "excerpt": d.get("content", "")[:200],
                "relevance_score": round(d.get("relevance_score", d.get("score", 0)), 2),
            }}

        yield {"type": SSEEventType.DONE, "data": {
            "workflow": "agent_loop",
            "thread_id": thread_id,
            "iterations": iteration,
        }}
