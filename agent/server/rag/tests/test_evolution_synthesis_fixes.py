"""演化分析综合叙述/E7 检索修复的回归测试（测试先行，2026-09-09）。

覆盖两个线上实锤 bug：

Bug A — 综合叙述被"掏空"（日志实锤：removed 67 fabricated tagged quotes）：
  LLM 实际输出 171 个直引号(0x22)、0 个弯引号，直引号同时用于强调/专名
  （如"异类报恩"）。旧校验先扫描引号对再向后找标签，强调引号的闭引号被
  当成下一段引文的开引号，配对错位后把【真实引文+真实标签】整段吞删。
  修复：validate_narrative_citations() 改为以 [标签] 为锚点反向提取
  最近引文（与前端 citeRegex 同构），直/弯/直角引号统一支持。

Bug B — E7 首轮证据检索全灭（日志实锤：retrieve_for_claim[i] 异常:
  too many values to unpack (expected 2)，6/6 条主张零证据全部"证据不足"）：
  retrieve_for_claim 中 asyncio.gather 有 3 个 awaitable 却解包成 2 个变量，
  异常被 gather(return_exceptions=True) 静默吞掉。

运行: python -m pytest server/rag/tests/test_evolution_synthesis_fixes.py -v
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    import pytest
except ImportError:
    class _SkipTest(Exception):
        pass

    class _PytestShim:
        @staticmethod
        def skip(msg=""):
            raise _SkipTest(msg)

    pytest = _PytestShim()

from server.langgraph.evolution_workbench import (
    EvolutionWorkbench,
    validate_narrative_citations,
)


# ---------------------------------------------------------------------------
# Bug A: 标签锚定式引文校验
# ---------------------------------------------------------------------------

# 模拟 E7 检索回来的 chunk 原文（excerptFull）
_EV_TANG = "贞元中，李黄归家，才及家，便觉脑疼，斯须益甚，至辰巳间，脑裂而卒。" \
           "其家仆于昨夜所止之处复验之，但见枯槐树中，有大蛇蟠屈之迹。"
_EV_RESEARCH = "《白蛇传》相关叙事从唐代发端，历经宋、明、清各代不断发展，" \
               "文体从文言小说逐渐衍生出话本、戏曲、弹词等多种大众文艺形式。"
_EV_QING = "我想那白蛇呵，可惜他千载焚修也那一旦抛，多也波姣本是妖。这的是人妖宿有苗。"


def _refs():
    return {
        "唐-ch1:0": {"excerptFull": _EV_TANG, "excerpt": _EV_TANG[:80]},
        "src-ch0:67": {"excerptFull": _EV_RESEARCH, "excerpt": _EV_RESEARCH[:80]},
        "清-ch4:0": {"excerptFull": _EV_QING, "excerpt": _EV_QING[:80]},
    }


def test_straight_quote_real_citations_are_kept():
    """直引号真实引文 + 强调引号专名交错 → 真实引文与标签全部保留，专名不被标注。"""
    narrative = (
        '追踪目标"白素贞"的母题矩阵显示，各母题在唐代几乎空白。'
        '这一演化轨迹与文献所载高度吻合：'
        '"《白蛇传》相关叙事从唐代发端，历经宋、明、清各代不断发展，文体从文言小说逐渐衍生出话本、戏曲、弹词等多种大众文艺形式"[src-ch0:67]。\n\n'
        '文本记载李黄归家后"才及家，便觉脑疼，斯须益甚，至辰巳间，脑裂而卒"[唐-ch1:0]，'
        '可见唐代白蛇仍属害人模式。"异类报恩"母题在唐代仅1次、清代20次。'
    )
    out, stats = validate_narrative_citations(narrative, _refs())

    # 真实标签全部保留
    assert "[src-ch0:67]" in out
    assert "[唐-ch1:0]" in out
    # 真实引文保留
    assert "才及家，便觉脑疼" in out
    assert "《白蛇传》相关叙事从唐代发端" in out
    # 强调/专名直引号原样保留，且不被误标训练数据
    assert '"异类报恩"' in out
    assert "（模型训练数据）" not in out
    assert '"白素贞"' in out
    assert stats["removed_fabricated"] == 0
    assert stats["removed_hallucinated_anchors"] == 0


def test_fabricated_quote_with_valid_label_has_tag_stripped():
    """标签真实但引文不在任何证据原文中 → 摘除标签（防张冠李戴），原文保留不掏空。

    真实日志实锤：LLM 常引用训练数据中的真实小说/戏曲原句，而检索回来的
    chunk 是标题级碎片，整句删除会把综合报告掏空（旧代码一次删 67 条）。
    新规则：非证据原文摘掉徽章标签、正文保留（弯引号另由 Pass2 标注来源）。
    """
    narrative = (
        "唐代白蛇温良和善，从不害人"  # 非证据原文
        '，法海奉旨收妖"[唐-ch1:0]。'
        "后续叙述应当保留。"
    )
    out, stats = validate_narrative_citations(narrative, _refs())

    # 错误标签摘除（徽章不再跳到不含此句的证据）
    assert "[唐-ch1:0]" not in out
    # 正文完整保留，不掏空
    assert "法海奉旨收妖" in out
    assert "唐代白蛇温良和善" in out
    assert "后续叙述应当保留" in out
    assert stats["removed_fabricated"] == 1


def test_hallucinated_anchor_is_removed():
    """LLM 编造不存在的标签 → 摘除假标签（前端不出现假徽章），原文保留。"""
    narrative = '某段编造原文"[明-ch9:99]。其余叙述保留。'
    out, stats = validate_narrative_citations(narrative, _refs())

    assert "[明-ch9:99]" not in out
    assert "其余叙述保留" in out
    assert "某段编造原文" in out  # 正文不删
    assert stats["removed_hallucinated_anchors"] == 1


def test_curly_fabricated_quote_without_tag_is_marked():
    """无标签弯引号引文不在证据语料中 → 标注（模型训练数据）；直引号专名不动。"""
    narrative = '如《警世通言》所载“西湖水干江潮不起雷峰塔倒白蛇出世”乃后世流传；而"盗仙草救夫"母题清代成熟。'
    out, stats = validate_narrative_citations(narrative, _refs())

    assert "（模型训练数据）" in out
    # 直引号专名不被误标
    assert '"盗仙草救夫"' in out
    assert stats["marked_memory"] == 1


def test_valid_curly_quote_without_tag_kept_unmarked():
    """无标签弯引号引文逐字命中证据语料 → 保留且不标注。"""
    narrative = "原文写道“但见枯槐树中，有大蛇蟠屈之迹”，可资佐证。"
    out, stats = validate_narrative_citations(narrative, _refs())

    assert "但见枯槐树中，有大蛇蟠屈之迹" in out
    assert "（模型训练数据）" not in out
    assert stats["marked_memory"] == 0


def test_bare_label_without_quote_is_kept():
    """标签前无引文（裸徽章）→ 保留标签供前端点击。"""
    narrative = "详见该段记载[清-ch4:0]，此为清代戏曲证据。"
    out, stats = validate_narrative_citations(narrative, _refs())

    assert "[清-ch4:0]" in out
    assert "此为清代戏曲证据" in out


def test_real_quote_near_wrong_label_gets_relabeled():
    """引文真实但挂错标签（标签证据中无此句，另一条证据有）→ 改写为正确标签。

    标签槽位 :N 由顺延分配，LLM 可能挂错槽号；改写后徽章跳转到真正
    包含该句的 chunk，既不张冠李戴也不损失真实引文。
    """
    # 唐代原文的引文挂到了清代标签下
    narrative = '记载云"才及家，便觉脑疼，斯须益甚，至辰巳间，脑裂而卒"[清-ch4:0]。'
    out, stats = validate_narrative_citations(narrative, _refs())

    assert "[清-ch4:0]" not in out
    assert "[唐-ch1:0]" in out  # 改写为正确标签
    assert "才及家，便觉脑疼" in out  # 引文保留
    assert stats["relabeled"] == 1
    assert stats["removed_fabricated"] == 0


# ---------------------------------------------------------------------------
# Bug B: E7 首轮检索 asyncio.gather 解包
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_test_claims_retrieve_populates_evidence_pool():
    """retrieve_for_claim 不得抛 'too many values to unpack'，证据池必须填充。"""
    wb = EvolutionWorkbench.__new__(EvolutionWorkbench)
    wb._live_queue = None  # _emit_live 安全降级

    async def fake_filtered_retrieve(query, filters=None, top_k=None, skip_tagging=False):
        # 各朝代/批次返回不同内容，模拟真实检索（内容按朝代不同，
        # 否则 retrieve_for_claim 的 content[:80] 去重会把 4 批次折叠成 1 条）
        _dyn = (filters or {}).get("dynasty", "明")
        _stype = (filters or {}).get("source_type", "primary_literature")
        return [{
            "content": f"【{_dyn}-{_stype}】白娘子主动追求许宣，报恩情结在{_dyn}文本中初步显现。" * 3,
            "chunk_id": f"chunk-{_dyn}-{_stype}",
            "metadata": {
                "chapter_number": 3,
                "chapter_title": "明·白娘子永镇雷峰塔",
                "paragraph_index": -1,
                "char_start": 100,
                "dynasty": _dyn,
                "source_type": _stype,
            },
        }]

    async def fake_opposing_query(statement, target):
        return ""

    wb._filtered_retrieve = fake_filtered_retrieve
    wb._generate_opposing_query = fake_opposing_query

    state = {"claims": [
        {"statement": "白蛇形象在明代文本中趋于人性化", "verdict": "pending"},
        {"statement": "清代戏曲强化了白蛇的伦理内涵", "verdict": "pending"},
    ]}

    result = await wb._test_claims_retrieve(state)
    claims = result["claims"]

    assert len(claims) == 2
    for c in claims:
        pool = c.get("evidence_pool")
        assert pool, "evidence_pool 必须被填充（gather 解包 bug 回归）"
        assert len(pool) >= 4  # 4 朝代并行检索 + extra
        ev0 = pool[0]
        assert ev0.get("excerptFull")
        assert ev0.get("chapterNumber") == 3
        assert "chunkId" in ev0 or "chunk_id" in ev0


if __name__ == "__main__":
    # 无 pytest 环境时的最小运行器
    import inspect

    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            if inspect.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
            print(f"PASS {name}")
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    if failures:
        sys.exit(1)
    print("ALL PASS")
