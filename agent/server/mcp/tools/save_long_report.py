"""v19: save_long_report — 超长回答的完整报告分段写入工具。

解决"输出 token 上限截断长回答"问题:
- LLM 单次生成最多 ~8000 tokens（约 5000 汉字），逐字对勘报告/系统性综述会被截断
- 本工具让 LLM 把完整报告分节写入磁盘 Markdown 文件（每节一次工具调用，不受单次生成长度限制）
- finalize 时返回完整内容，agent_loop 下发 report_ready SSE 事件
- 前端渲染报告卡片，用户可下载 .md；报告卡片随消息持久化到用户数据库

存储路径: agent/reports/{user_id}/{report_id}.md + .meta.json
"""
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "reports"

MAX_SECTION_CHARS = 6000  # 单节正文上限（约 3000 汉字，留足余量）


def _safe_id(raw: str) -> str:
    """清洗 user_id / report_id，防止路径穿越。"""
    return "".join(c for c in (raw or "") if c.isalnum() or c in "-_") or "default"


def _user_dir(user_id: str) -> Path:
    d = REPORTS_DIR / _safe_id(user_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _meta_path(user_id: str, report_id: str) -> Path:
    return _user_dir(user_id) / f"{_safe_id(report_id)}.meta.json"


def _md_path(user_id: str, report_id: str) -> Path:
    return _user_dir(user_id) / f"{_safe_id(report_id)}.md"


def _load_meta(user_id: str, report_id: str) -> dict | None:
    p = _meta_path(user_id, report_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _save_meta(user_id: str, meta: dict) -> None:
    _meta_path(user_id, meta["reportId"]).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


TOOL_DEF = {
    "name": "save_long_report",
    "description": (
        "把内容丰富的回答完整写入报告文件（逐字对勘报告、多章节详尽对比、完整时间线+分析、"
        "系统综述、逐处出场轨迹、逐首诗词汇编等穷举清单）。"
        "触发信号（满足任一就应使用，不要凭感觉跳过）：工具返回超过 15 条结果或涉及 3 个以上章节；"
        "用户要求「完整/全部/逐一/详细/系统梳理/汇总」；你开始想「概括一下/只列重点」。"
        "用法：先 create 创建报告拿到 report_id，再多次 append 每次写一节（每节 1000-2000 字，"
        "全部素材一条不丢），全部写完后 finalize 封卷（系统自动把完整报告推送给用户下载）。"
        "对话框里只给简明摘要，完整内容走报告文件——严禁靠压缩、省略素材来挤进对话框。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "append", "finalize", "list", "get"],
                "description": "create=创建报告; append=追加一节; finalize=封卷并返回完整内容; list=历史报告列表; get=读取某报告",
            },
            "title": {"type": "string", "description": "报告标题（create 时必填）"},
            "topic": {"type": "string", "description": "研究主题/用户原始问题（create 时填）"},
            "report_id": {"type": "string", "description": "报告 ID（append/finalize/get 时必填，create 返回值）"},
            "section_title": {"type": "string", "description": "小节标题（append 时必填）"},
            "content": {"type": "string", "description": "小节完整正文 Markdown（append 时必填，单次不超过 6000 字符）"},
        },
        "required": ["action"],
    },
}


async def handler(
    action: str = "",
    user_id: str = "default",
    title: str = "",
    topic: str = "",
    report_id: str = "",
    section_title: str = "",
    content: str = "",
) -> dict:
    """生成长报告工具 handler。"""
    # action 缺失/非法时返回友好提示而非抛 TypeError，便于 LLM 立即修正重试
    if action not in ("create", "append", "finalize", "list", "get"):
        return {
            "error": (
                f"缺少或非法的 action 参数（收到: {action!r}）。"
                "请从 create / append / finalize / list / get 中选择后重试。"
                "典型流程：create 拿 report_id → 多次 append → finalize 封卷。"
            )
        }
    uid = _safe_id(user_id)

    # ---------- create ----------
    if action == "create":
        rid = f"rpt_{time.strftime('%Y%m%d_%H%M%S')}"
        meta = {
            "reportId": rid,
            "title": (title or "研究报告").strip()[:120],
            "topic": (topic or "").strip()[:500],
            "sections": [],
            "createdAt": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finalized": False,
        }
        _save_meta(uid, meta)
        header = (
            f"# {meta['title']}\n\n"
            f"> 研究主题：{meta['topic'] or '（未填写）'}\n"
            f"> 生成时间：{meta['createdAt']}\n"
            f"> 由白蛇传文脉全息 Agent 生成\n\n---\n\n"
        )
        _md_path(uid, rid).write_text(header, encoding="utf-8")
        return {
            "reportId": rid,
            "title": meta["title"],
            "note": (
                f"报告已创建（report_id={rid}）。请立即开始 append 分节写入正文，"
                f"每节 1000-2000 字，写完所有节后调用 finalize 封卷。"
            ),
        }

    # ---------- append ----------
    if action == "append":
        if not report_id:
            return {"error": "append 需要 report_id 参数（先调用 create 获取）"}
        meta = _load_meta(uid, report_id)
        if meta is None:
            return {"error": f"报告不存在: {report_id}，请先 create"}
        if meta.get("finalized"):
            return {"error": f"报告 {report_id} 已封卷，不能再追加"}
        if not section_title or not content:
            return {"error": "append 需要 section_title 和 content"}
        section_content = content.strip()
        if len(section_content) > MAX_SECTION_CHARS:
            # 不拒绝，截断保护 + 提示 LLM 该节应拆成两节
            section_content = section_content[:MAX_SECTION_CHARS]
            oversize_note = f"（本节超出 {MAX_SECTION_CHARS} 字符已截断；请把剩余内容作为新小节 append）"
        else:
            oversize_note = ""

        idx = len(meta["sections"]) + 1
        # 剥离 LLM 在 content 开头重复书写的标题行（工具已自动生成
        # "## {idx}. {section_title}"，重复会造成双标题）。
        # 匹配 content 首个非空行，若为 markdown 标题且文本与 section_title
        # 高度相似（去空白后互相包含），则删除该行。
        lines = section_content.split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("#"):
                heading_text = stripped.lstrip("#").strip()
                t = section_title.strip()
                if heading_text and t and (
                    heading_text == t or t in heading_text or heading_text in t
                ):
                    lines = lines[i + 1:]
                    section_content = "\n".join(lines).strip("\n")
            break
        block = f"## {idx}. {section_title.strip()}\n\n{section_content}\n\n"
        with _md_path(uid, report_id).open("a", encoding="utf-8") as f:
            f.write(block)
        meta["sections"].append({"index": idx, "title": section_title.strip(), "chars": len(section_content)})
        _save_meta(uid, meta)
        total_chars = len(_md_path(uid, report_id).read_text(encoding="utf-8"))
        return {
            "reportId": report_id,
            "sectionIndex": idx,
            "sectionTitle": section_title.strip(),
            "sectionChars": len(section_content),
            "totalChars": total_chars,
            "sectionCount": len(meta["sections"]),
            "note": f"第 {idx} 节已写入。" + oversize_note + (
                f"目前共 {len(meta['sections'])} 节。如全部写完请调用 finalize。"
            ),
        }

    # ---------- finalize ----------
    if action == "finalize":
        if not report_id:
            return {"error": "finalize 需要 report_id 参数"}
        meta = _load_meta(uid, report_id)
        if meta is None:
            return {"error": f"报告不存在: {report_id}"}
        md_file = _md_path(uid, report_id)
        if not meta.get("finalized"):
            meta["finalized"] = True
            meta["finalizedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
            _save_meta(uid, meta)
        full_content = md_file.read_text(encoding="utf-8") if md_file.exists() else ""
        return {
            "reportReady": True,  # agent_loop 识别此标记 → 下发 report_ready SSE
            "reportId": report_id,
            "title": meta["title"],
            "topic": meta.get("topic", ""),
            "totalChars": len(full_content),
            "sectionCount": len(meta["sections"]),
            "sections": [s["title"] for s in meta["sections"]],
            "createdAt": meta["createdAt"],
            "finalizedAt": meta.get("finalizedAt", ""),
            "content": full_content,
            "note": "报告已封卷，系统已推送给用户。对话框中给出简明摘要即可。",
        }

    # ---------- list ----------
    if action == "list":
        items = []
        d = _user_dir(uid)
        for mp in sorted(d.glob("*.meta.json"), reverse=True):
            try:
                meta = json.loads(mp.read_text(encoding="utf-8"))
                items.append({
                    "reportId": meta.get("reportId"),
                    "title": meta.get("title"),
                    "topic": meta.get("topic", ""),
                    "sectionCount": len(meta.get("sections", [])),
                    "createdAt": meta.get("createdAt"),
                    "finalized": meta.get("finalized", False),
                })
            except (json.JSONDecodeError, OSError):
                continue
        return {"reports": items, "count": len(items)}

    # ---------- get ----------
    if action == "get":
        if not report_id:
            return {"error": "get 需要 report_id 参数"}
        meta = _load_meta(uid, report_id)
        if meta is None:
            return {"error": f"报告不存在: {report_id}"}
        md_file = _md_path(uid, report_id)
        return {
            **meta,
            "content": md_file.read_text(encoding="utf-8") if md_file.exists() else "",
        }

    return {"error": f"未知 action: {action}"}
