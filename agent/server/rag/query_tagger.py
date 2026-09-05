"""query 标签解析：在线 LLM 打标（YAML 输出），从词表选 canonical。

复用 rag_eval 的 query 打标 prompt + 全量词表。
"""
import json
import re
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
VOCAB_PATH = _PROJECT_ROOT / "rag_eval" / "results" / "merged_vocab_full.json"

GENRE_HINT = (
    "classical_chinese=唐文言小说《李黄》/ huaben=宋话本《西湖三塔记》/ "
    "imitation_huaben=明拟话本《白娘子永镇雷峰塔》/ opera_libretto=清戏曲方成培《雷峰塔传奇》/ "
    "chapter_vernacular=清白话神魔小说玉山堂主人《雷峰塔传奇》/ tanci=清苏州弹词《义妖传》/ "
    "script=现当代京剧田汉《白蛇传》"
)


def _format_vocab(vocab, kind, label):
    entries = [e for e in vocab.get(kind, []) if e.get("canonical") != "待删除"]
    lines = [f"{label}词表:"]
    for e in entries:
        raw = "、".join(e.get("raw", []))
        lines.append(f"- {e['canonical']} <- {raw}")
    return "\n".join(lines)


def _build_vocab_text():
    vocab = json.load(open(VOCAB_PATH, encoding="utf-8"))
    return "\n\n".join([
        _format_vocab(vocab, "persons", "persons(人物)"),
        _format_vocab(vocab, "locations", "locations(地点)"),
        _format_vocab(vocab, "plot_units", "plot_unit(情节单元)"),
        _format_vocab(vocab, "topics", "分析主题"),
    ])


_VOCAB_TEXT = _build_vocab_text()

QUERY_SYSTEM = """你是白蛇传文献研究专家。请解析下面这个问题的结构化要素。

版本 genre 映射: {genre_hint}

下面是从语料中归纳的受控词表:

{vocab_text}

输出 YAML:
query_type: 类型
persons:
  - canonical: 标准名
    raw: 原文称呼
locations:
  - canonical: 标准名
    raw: 原文称呼
plot_unit:
  - canonical: 标准名
    raw: 原文称呼
plot_detail: 细情节描述
分析主题:
  - canonical: 标准名
    raw: 原文称呼
对比维度: {{}}
summary: 一句话
genre: genre代码或空

字段说明:
- query_type: 严格从 情节题/人物题/地点题/comparison题/研究分析题 五类中选，不要用"其他"。
  判断标准:
  - 情节题: 问某个情节的具体内容、细节或文本（含唱词、台词、对白、发生的事情）
  - 人物题: 问某个人物的身份、关系或行为
  - 地点题: 问某个地点
  - comparison题: 跨版本或跨文本的比较、承袭、源流
  - 研究分析题: 问学术分析（形象流变、景观互构、主题演变、版本谱系等）
- persons/locations/plot_unit/分析主题: canonical 必须从上面词表的 canonical 里选，raw 用问题原文里实际出现的称呼
- plot_detail: 问题问的具体情节，一句话，无则留空
- 对比维度: 仅 comparison 题填。字段名严格用 朝代/文本/体裁/作者/称谓:
  - 朝代: 唐/宋/明/清/现当代
  - 文本: 李黄/西湖三塔记/白娘子永镇雷峰塔/雷峰塔传奇/义妖传/白蛇传
  - 体裁: 文言小说/话本/拟话本/戏曲/白话神魔小说/苏州弹词/京剧
  - 作者: 佚名/冯梦龙/方成培/玉山堂主人/陈遇乾/田汉
  - 称谓: 具体称呼列表，如 ["汉文","许宣"]
  - 无具体值的维度省略
- 分析主题: 仅研究分析题或 comparison 题填
- summary: 这个问题问的是什么事，一句话
- genre: 问题明确指向的版本 genre 代码（从上面映射里选），不明确则留空

只输出 YAML，不要其他文字。"""


def _parse_yaml(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:]) if len(lines) > 1 else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        obj = yaml.safe_load(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    return None


async def tag_query(provider, query_text, sem=None):
    """对 query 打标，返回标签 dict（或 {"_error": ...}）。"""
    from ..observability.tracer import llm_call_context  # B-152: 消除 trace 中 purpose=unknown 盲区

    system = QUERY_SYSTEM.format(genre_hint=GENRE_HINT, vocab_text=_VOCAB_TEXT)
    user = f"问题是: {query_text}"

    async def _call():
        for attempt in range(3):
            try:
                with llm_call_context("rag:query_tagger", "query_tagging",
                                      upstream="retriever", downstream="retriever"):
                    resp = await provider.generate(
                        system=system, user=user, model=None,
                        max_tokens=4096, temperature=0.1,
                    )
            except Exception as e:
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                return {"_error": f"{type(e).__name__}: {e}"}
            parsed = _parse_yaml(resp.content)
            if isinstance(parsed, dict):
                return parsed
            import asyncio
            await asyncio.sleep(1.0)
        return {"_error": "parse_failed"}

    if sem is not None:
        async with sem:
            return await _call()
    return await _call()
