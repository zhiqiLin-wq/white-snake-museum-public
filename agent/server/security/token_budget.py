"""LLM Token 预算控制 — 防止单次请求或恶意输入消耗过多 Token。"""
import logging
from ..config import settings

logger = logging.getLogger(__name__)

MAX_INPUT_TOKENS = 8_000
MAX_OUTPUT_TOKENS = 2_000
MAX_CONVERSATION_TOKENS = 50_000


class TokenBudget:
    """Token 预算管理。"""

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """粗略估算：中文 ~1.5 字符/token，英文 ~4 字符/token。"""
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '一' <= c <= '鿿')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)

    @staticmethod
    def truncate_context(docs: list, max_tokens: int | None = None) -> list:
        """按 Token 预算截断检索结果，保留高相关度文档。"""
        if max_tokens is None:
            max_tokens = MAX_INPUT_TOKENS

        # 按相关度降序
        sorted_docs = sorted(docs, key=lambda d: getattr(d, 'score', getattr(d, 'relevance_score', 0)), reverse=True)

        selected = []
        used = 0
        for doc in sorted_docs:
            content = getattr(doc, 'content', '') or getattr(doc, 'page_content', '') or ''
            est = TokenBudget.estimate_tokens(content)
            if used + est > max_tokens:
                break
            selected.append(doc)
            used += est
        return selected

    @staticmethod
    def check_input_budget(text: str) -> bool:
        """检查输入是否超过单次上限。"""
        return TokenBudget.estimate_tokens(text) <= MAX_INPUT_TOKENS

    @staticmethod
    def get_max_output_tokens() -> int:
        return settings.max_output_tokens or MAX_OUTPUT_TOKENS
