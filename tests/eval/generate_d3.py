"""D3 长对话记忆保持集生成器（v2：30 条关键信息 + 短/中/长 3 段对话）。

30 条用户关键信息（身份/偏好/观点/约束，第一人称，均 >= 20 字以通过长度过滤），
拆成 3 段对话（短 20 轮 / 中 50 轮 / 长 100 轮），每段埋 10 条关键信息，
中间用白蛇传客观闲聊填充。评测长对话后关键信息的保持率（含长度维度）。
输出 tests/eval/data/d3_retention.json
"""

import asyncio
import json
import re
import sys
from pathlib import Path

_eval_dir = Path(__file__).resolve().parent
_project_root = _eval_dir.parent.parent
sys.path.insert(0, str(_project_root))
sys.path.insert(0, str(_eval_dir))

from common import write_json
from agent.server.llm.deepseek_provider import DeepSeekProvider
from agent.server.config import settings

MODEL = settings.memory_extraction_model or settings.effective_fast_model

# 30 条关键信息（第一人称，均 >= 20 字），分 3 组，每组 10 条
SEG1_FACTS = [
    "我叫小研，是一名专门研究白蛇传文献的学者。",
    "我的研究方向是宋代江南民间信仰与文学互动。",
    "在所有文学人物里，我最喜欢的是法海这个角色。",
    "我不喜欢收到长篇大论的回复，简短一点就行。",
    "我偏好把文本细读和宏观历史分析结合起来研究。",
    "我特别关注《西湖三塔记》中的三怪母题。",
    "我认为《白娘子永镇雷峰塔》是白蛇传定型的标志。",
    "我研究时要求引用文献原文并标注清楚出处。",
    "我习惯用母题流变的视角来分析不同版本之间的演变。",
    "我关注雷峰塔意象从镇妖到镇人的历史变迁。",
]

SEG2_FACTS = [
    "我觉得法海这个人物有卫道者和悲剧推动者的双重性。",
    "我认为白娘子从《西湖三塔记》到《白娘子永镇雷峰塔》经历了从妖性到人性的转变。",
    "我不太喜欢把白蛇传和现代恋爱剧混为一谈，这两者差别很大。",
    "我研究版本演变时，倾向于先考订版本年代再讨论内容。",
    "我认为田汉的京剧改编强化了反封建的主题。",
    "我偏好直接引用原文，而不是转述别人的二手解读。",
    "我觉得小青从青鱼精变成青蛇反映了民间想象的演变。",
    "我比较反感过度解读文献、硬套理论的做法。",
    "我习惯回答问题的时候先给结论，然后再慢慢展开分析。",
    "我认为雷峰塔倒塌这件事在民间记忆里有特殊意义。",
]

SEG3_FACTS = [
    "我不喜欢看那种堆砌版本罗列、没有分析的整理。",
    "我特别在意文献引用的准确性，不能凭记忆转述。",
    "我觉得《白娘子永镇雷峰塔》的悲剧结局比大团圆结局更有文学价值。",
    "我研究版本演变的时候，一定会按照朝代顺序来梳理。",
    "我认为许仙这个人物在不同版本里形象差异很大，值得专门研究。",
    "我不太喜欢别人用现代价值观去评判古代作品。",
    "我偏好先读原始文献本身，然后再去看二手研究。",
    "我觉得白蛇传和梁祝、孟姜女并称的提法值得商榷。",
    "我习惯把问题放到具体的历史语境里去理解，而不是孤立地看。",
    "我不喜欢回复里出现英文术语，能用中文就说中文。",
]

# 3 段对话配置：段 id、标签、埋入的关键信息、闲聊组数（每组 = 问 + 答 2 轮）
# n_mid = 前后两批关键信息之间的闲聊组数，n_tail = 后 5 条关键信息之后的闲聊组数
SEGMENTS = [
    {"id": "short", "label": "短对话 20 轮", "facts": SEG1_FACTS, "n_mid": 3, "n_tail": 2},
    {"id": "mid", "label": "中对话 50 轮", "facts": SEG2_FACTS, "n_mid": 15, "n_tail": 5},
    {"id": "long", "label": "长对话 100 轮", "facts": SEG3_FACTS, "n_mid": 35, "n_tail": 10},
]

CHITCHAT_USER = (
    "请生成 45 条关于白蛇传的学术闲聊问答（一问一答），内容与用户画像无关，"
    "只涉及白蛇传的版本、人物、情节、地理等客观话题。\n"
    "只输出 JSON 数组，每个元素形如 {{\"q\": \"问题\", \"a\": \"回答\"}}。"
)

GENERATOR_SYSTEM = "你是一个严谨的数据集生成器，只输出合法 JSON。"


def _build_conversation(facts, chitchat, n_mid, n_tail):
    """前 5 条关键信息 -> n_mid 组闲聊 -> 后 5 条关键信息 -> n_tail 组闲聊。

    后 5 条关键信息后面补 n_tail 组闲聊，把末尾关键信息"顶"出 FIFO 触发提取，
    避免关键信息因埋在对话末尾、未触发压缩弹出而永远留在缓冲区没进记忆库。
    """
    conversation = []
    for f in facts[:5]:
        conversation.append({"role": "user", "content": f})
    for c in chitchat[:n_mid]:
        conversation.append({"role": "user", "content": c["q"]})
        conversation.append({"role": "assistant", "content": c["a"]})
    for f in facts[5:]:
        conversation.append({"role": "user", "content": f})
    for c in chitchat[n_mid:n_mid + n_tail]:
        conversation.append({"role": "user", "content": c["q"]})
        conversation.append({"role": "assistant", "content": c["a"]})
    return conversation


async def main():
    llm = DeepSeekProvider()
    chitchat = []
    for _ in range(5):
        try:
            resp = await llm.generate(system=GENERATOR_SYSTEM, user=CHITCHAT_USER, model=MODEL,
                                      max_tokens=8192, temperature=0.8)
            txt = resp.content.strip()
            m = re.search(r"\[[\s\S]*\]", txt)
            data = json.loads(m.group(0)) if m else json.loads(txt)
            chitchat = [c for c in data if isinstance(c, dict) and c.get("q") and c.get("a")]
            if chitchat:
                break
        except Exception:
            chitchat = []
            await asyncio.sleep(3)

    if len(chitchat) < 45:
        print(f"  [警告] 只生成 {len(chitchat)} 组闲聊，长对话可能不足 100 轮")

    segments = []
    for seg in SEGMENTS:
        conversation = _build_conversation(seg["facts"], chitchat, seg["n_mid"], seg["n_tail"])
        segments.append({
            "id": seg["id"],
            "label": seg["label"],
            "key_facts": seg["facts"],
            "conversation": conversation,
        })
        print(f"  [{seg['id']}] {seg['label']}: 埋 {len(seg['facts'])} 条关键信息, "
              f"{len(conversation)} 轮对话")

    out = {
        "meta": {
            "name": "D3 长对话记忆保持集",
            "segments": len(segments),
            "total_key_facts": sum(len(s["facts"]) for s in SEGMENTS),
        },
        "segments": segments,
    }
    path = write_json(out, "d3_retention.json")
    print(f"\n完成：{len(segments)} 段对话，共 {out['meta']['total_key_facts']} 条关键信息。写入 {path}")


if __name__ == "__main__":
    asyncio.run(main())
