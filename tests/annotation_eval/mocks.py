"""标注评测 mock 依赖。

- MockLLM: 可编程 LLM。核心能力：从 user prompt 里提取原文，对已知实体
  做 str.find 生成偏移，可选择性地把偏移"污染"（故意偏移）以测偏移校验器。
  并提供 empty_discovery / empty_annotate 开关。
- MockNodeClient: 只实现 get_literature_by_chapter。
- MockPromptRegistry: 只有 render(template, vars) 方法，供流水线用。

全部 mock 都是确定性、无网络的，保证评测可复现。
"""

import json
import logging

from .fixtures import build_chapter, GOLDEN_ENTITIES

logger = logging.getLogger("annotation_eval.mocks")


class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content


class MockLLM:
    """从原文里找已知实体返回标注的确定性 LLM。

    corrupt_offset_delta: 非 0 时把每个 start 故意偏移 delta 字符，
        用于测 annotate_passage._resolve_offset 的窗口/全文回退。
    return_empty: True 时 annotate 调用返回空 annotations（测空批重试路径）。
    """

    def __init__(self, corrupt_offset_delta: int = 0, return_empty: bool = False):
        self.corrupt_offset_delta = corrupt_offset_delta
        self.return_empty = return_empty
        self.calls = []  # 记录每次调用 (system_head, user_head, model, max_tokens)

    async def generate(self, system: str, user: str, model: str = "",
                       max_tokens: int = 2048, temperature: float = 0.1,
                       thinking_disabled: bool = False):
        self.calls.append({
            "system_head": system[:60],
            "user_head": user[:60],
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        logger.debug(
            "[mock-llm] generate model=%s max_tokens=%s temp=%s system=%r user=%r",
            model, max_tokens, temperature, system[:40], user[:40],
        )

        # 意图解析（annotate_user_request 阶段2）→ 固定返回全部类别
        if "意图解析器" in system:
            return MockLLMResponse(json.dumps({
                "is_annotation": True,
                "action": "annotate",
                "categories": ["person", "location", "event", "term", "motif"],
                "color_overrides": {},
                "confidence": 0.9,
            }, ensure_ascii=False))

        # Discovery（流水线 Pass1）→ 从段落列表里扫已知实体
        if "段落列表" in user or "discoveries" in system:
            return MockLLMResponse(self._discovery_response(user))

        # Precise（流水线 Pass2）→ 从"段落文本:"里定位实体
        if "段落文本" in user or "精确定位" in system:
            return MockLLMResponse(self._precise_response(user))

        # annotate_passage / annotate_user_request 的实体标注调用
        return MockLLMResponse(self._annotate_response(user))

    def _extract_annotate_text(self, user: str) -> str:
        """从 annotate 的 user prompt 里抠出原文块。"""
        marker = "原文:"
        idx = user.find(marker)
        if idx < 0:
            return ""
        rest = user[idx + len(marker):]
        end_marker = "请标注"
        eidx = rest.find(end_marker)
        if eidx >= 0:
            rest = rest[:eidx]
        return rest.strip().strip("\n").strip()

    def _scan_entities(self, text: str):
        """对 text 扫 GOLDEN_ENTITIES，返回 [{entity, category, start, end}]。

        长实体优先，且不与已匹配区间重叠（与规则引擎一致）。
        """
        entities = sorted(GOLDEN_ENTITIES, key=lambda x: -len(x[0]))
        matched_ranges = []
        out = []
        for entity, category in entities:
            search_start = 0
            while True:
                pos = text.find(entity, search_start)
                if pos < 0:
                    break
                end = pos + len(entity)
                overlaps = any(pos < m_end and end > m_start for m_start, m_end in matched_ranges)
                if not overlaps:
                    matched_ranges.append((pos, end))
                    out.append({
                        "entity": entity,
                        "category": category,
                        "start_char": pos,
                        "end_char": end,
                    })
                search_start = pos + 1
        return out

    def _annotate_response(self, user: str) -> str:
        if self.return_empty:
            return ""

        text = self._extract_annotate_text(user)
        # 行式 tuple 输出：实体<|#|>类别<|#|>概括词<|#|>解释<|COMPLETE|>
        # 坐标不再由 LLM 返回（corrupt_offset_delta 已无作用，坐标由代码定位）。
        lines = []
        for item in self._scan_entities(text):
            cat = item["category"]
            summary = item["entity"] if cat in ("event", "motif") else ""
            lines.append(
                f"{item['entity']}<|#|>{cat}<|#|>{summary}<|#|>mock-{cat}<|COMPLETE|>"
            )
        return "\n".join(lines)

    def _discovery_response(self, user: str) -> str:
        """解析 user 里的 '--- 段落 N ---' 块，返回 discoveries。

        用正则按段落标记切块，避免多段落窗口时把下一个段落标记当成正文。
        """
        import re
        discoveries = []
        # 匹配 "--- 段落 N ---" 及其后正文（到下一个标记或"请"为止）
        pattern = re.compile(r"--- 段落 (\d+) ---\n(.*?)(?=\n--- 段落 |\n请|$)", re.S)
        for m in pattern.finditer(user):
            idx = int(m.group(1))
            text = m.group(2).strip()
            for item in self._scan_entities(text):
                discoveries.append({
                    "entity": item["entity"],
                    "category": item["category"],
                    "paragraph_index": idx,
                    "context_hint": text[max(0, item["start_char"] - 5):item["end_char"] + 5],
                    "explanation": f"mock-{item['category']}",
                })
        return json.dumps({"discoveries": discoveries}, ensure_ascii=False)

    def _precise_response(self, user: str) -> str:
        """解析 user 里的 '段落文本: ...'，返回 annotations。"""
        marker = "段落文本:"
        idx = user.find(marker)
        text = user[idx + len(marker):].strip() if idx >= 0 else ""
        annotations = []
        for item in self._scan_entities(text):
            annotations.append({
                "entity": item["entity"],
                "category": item["category"],
                "start_char": item["start_char"],
                "end_char": item["end_char"],
                "explanation": f"mock-{item['category']}",
            })
        return json.dumps({"annotations": annotations}, ensure_ascii=False)


class MockNodeClient:
    """只实现 get_literature_by_chapter，返回固定章节。"""

    def __init__(self, chapter: dict | None = None):
        self._chapter = chapter if chapter is not None else build_chapter()
        self.saved = []  # 记录 save 调用

    async def get_literature_by_chapter(self, chapter_number: int):
        if self._chapter.get("chapterNumber") == chapter_number:
            return self._chapter
        return None


class MockPromptRegistry:
    """流水线需要的 prompt_registry，只有 render(template, vars)。"""

    def render(self, template: str, variables: dict):
        if template == "annotation_discovery":
            return self._render_discovery(variables)
        if template == "annotation_precise":
            return self._render_precise(variables)
        raise ValueError(f"unknown template: {template}")

    @staticmethod
    def _render_discovery(variables: dict):
        system = "你是文本标注专家。discoveries"
        paragraphs = variables.get("paragraphs", [])
        blocks = []
        for p in paragraphs:
            blocks.append(f"--- 段落 {p['index']} ---\n{p['text']}")
        user = "段落列表:\n" + "\n".join(blocks) + "\n请扫描"
        return system, user

    @staticmethod
    def _render_precise(variables: dict):
        system = "你是文本标注专家。精确定位"
        user = "段落文本: " + variables.get("paragraph_text", "")
        return system, user
