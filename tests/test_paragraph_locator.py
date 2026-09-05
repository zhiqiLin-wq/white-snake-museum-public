"""段落定位器（paragraph_locator）测试 — 先于实现编写。

背景（B-163）：语料 79% 的 chunk paragraph_index=-1（明/清等章节源文件用单换行
分段，\\n\\n 切不出段落），导致出场轨迹等工具返回的段落号无效 → 前端
"跳转到原文"定位失败且高亮为空。

修复方案：复刻 Node API（server/src/services/literature.service.ts parseLiterature）
的章节切分逻辑 + 前端 splitParagraphs 的段落切分逻辑，运行时用 chunk content
在章节段落中子串定位，恢复真实段落号。

运行: python tests/test_paragraph_locator.py
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.server.rag.paragraph_locator import (  # noqa: E402
    ParagraphLocator,
    clean_text_for_locate,
)

# ---- 与 Node parseLiterature / 前端 splitParagraphs 行为一致的测试语料 ----

SAMPLE_TEXT = """白蛇传文献选集

一、萌芽期 唐：太平广记·李黄
唐贞元中，有盐铁使逊之侄子。~~~~段1正文。

二、萌芽期 宋：西湖三塔记
宋话本正文第一段。
宋话本正文第二段。

三、成熟期 明：白娘子永镇雷峰塔
话说西湖景致，山林树木。

第一回　许仙游湖。

雨过天晴，湖上遇蛇仙。

四、成熟期 清：雷峰塔传奇
清传奇正文。
"""


def _make_locator() -> ParagraphLocator:
    """用内存样本构造（绕过文件加载）。"""
    loc = ParagraphLocator(source_path='')
    loc._load_from_text(SAMPLE_TEXT)
    return loc


class TestCleanText(unittest.TestCase):

    def test_normalize_crlf(self):
        self.assertEqual(clean_text_for_locate('a\r\nb\rc'), 'a\nb\nc')

    def test_strip_bom_and_collapse(self):
        self.assertEqual(clean_text_for_locate('\ufeffa\n\n\n\nb'), 'a\n\nb')


class TestChapterSplit(unittest.TestCase):
    """章节切分必须与 Node parseLiterature 行为一致（标题正则 + 区间截取）。"""

    def test_split_chapters(self):
        loc = _make_locator()
        # 中文数字按 Unicode 码点排序非字典序（一<三<二<四），用集合比较
        self.assertEqual(set(loc.chapters().keys()), {'一', '二', '三', '四'})

    def test_chapter_paragraph_split(self):
        loc = _make_locator()
        paras = loc.paragraphs('三')
        # 明章节 content 按空行切段（不含标题行本身）
        self.assertEqual(len(paras), 3)
        self.assertTrue(paras[0].startswith('话说西湖景致'))
        self.assertTrue(paras[2].startswith('雨过天晴'))

    def test_title_line_not_mixed_into_paragraphs(self):
        # 回归：^\s* 吞换行导致 m.start() 落在标题前空行 → 标题行混入段落
        loc = _make_locator()
        for num, paras in loc.chapters().items():
            for p in paras:
                self.assertFalse(
                    p.startswith(('一、', '二、', '三、', '四、', '五、', '六、', '七、')),
                    f'章节{num} 段落混入标题行: {p[:30]!r}'
                )

    def test_chinese_number_chapters_kept_as_key(self):
        loc = _make_locator()
        # chapter key 用中文数字（与 chunk metadata chapter_number 一致）
        self.assertIn('三', loc.chapters())


class TestLocateParagraph(unittest.TestCase):
    """chunk content → 章节内段落号。多级降级：整串 → 中段 → 前缀。"""

    def setUp(self):
        self.loc = _make_locator()

    def test_exact_full_match(self):
        # chunk = 段落全文（含跨段？不含——chunk 属于单段时整串命中）
        pno = self.loc.locate('三', '雨过天晴，湖上遇蛇仙。')
        self.assertEqual(pno, 2)

    def test_partial_match_middle_snippet(self):
        # chunk 是段落的一部分（句级切分）；片段需 >= _MIN_SNIPPET_LEN(6)
        pno = self.loc.locate('三', '湖上遇蛇仙。')
        self.assertEqual(pno, 2)

    def test_min_length_guard(self):
        # 过短内容不定位（误匹配风险高），返回 None
        self.assertIsNone(self.loc.locate('三', '雨过'))
        self.assertIsNone(self.loc.locate('三', ''))

    def test_unknown_chapter_returns_none(self):
        self.assertIsNone(self.loc.locate('九', '很长很长很长的内容片段'))
        self.assertIsNone(self.loc.locate('', '很长很长很长的内容片段'))

    def test_not_found_returns_none(self):
        self.assertIsNone(self.loc.locate('三', '这段文字不存在于样本中呀呀呀'))

    def test_content_with_title_prefix_stripped(self):
        # ch七 类 chunk 带 "七、流变期… · " 标题装饰 → 剥掉后定位
        pno = self.loc.locate('三', '三、成熟期 明：白娘子永镇雷峰塔 · 雨过天晴，湖上遇蛇仙')
        self.assertEqual(pno, 2)

    def test_whitespace_normalized_match(self):
        # B-163+: chunk 内含 \n\n / 换行空白（跨段片段、诗曰体、装饰后缀）时，
        # 原样子串匹配必败；匹配需对空白归一化（段落号定位不需要字符位置）
        pno = self.loc.locate('三', '雨过天晴，\n\n湖上遇蛇仙。')
        self.assertEqual(pno, 2)
        # 前导空白 + 内部换行
        pno2 = self.loc.locate('三', '\n 话说西湖景致，\n山林树木。')
        self.assertEqual(pno2, 0)


class TestLocateInRealCorpus(unittest.TestCase):
    """真实语料冒烟测试：要求 excel_data 源文件存在时才跑。"""

    SRC = Path(r'e:\Flow\white-snake-museum-public\excel_data\白蛇传文献选集.txt')

    def setUp(self):
        if not self.SRC.exists():
            self.skipTest('真实语料文件不存在')

    def test_real_corpus_chapters_and_paragraphs(self):
        loc = ParagraphLocator(source_path=str(self.SRC))
        chs = loc.chapters()
        self.assertTrue({'一', '二', '三', '四', '五', '六', '七'} <= set(chs.keys()))
        # 章节段落量级：明·白娘子永镇雷峰塔应有可观段落数（旧逻辑只有 5 段）
        # 注意：段落切分仍按 \n\n（与前端 DOM 一致），段落数可能仍少——
        # 这里只验证章节切分本身完整
        self.assertGreaterEqual(len(chs), 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)
