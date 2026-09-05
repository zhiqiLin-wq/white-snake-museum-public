"""tiktoken cl100k_base 封装，提供统一的 token 计数接口。

使用 OpenAI cl100k_base 编码器作为近似计数。注意：该编码器面向英文优化，
对中文的 token 计数偏差可能明显大于 10%（DeepSeek 中文 tokenizer 通常更省 token）。
此处仅作为 FIFO 容量判断的近似值，不是精确 token 计数。
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_ENCODING: Optional["tiktoken.Encoding"] = None


def get_encoding() -> "tiktoken.Encoding":
    """延迟加载 cl100k_base encoding（单例）。"""
    global _ENCODING
    if _ENCODING is None:
        try:
            import tiktoken
            _ENCODING = tiktoken.get_encoding("cl100k_base")
            logger.info("tiktoken cl100k_base encoding loaded")
        except ImportError:
            raise ImportError(
                "tiktoken is required for token counting. "
                "Install it with: pip install tiktoken"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load tiktoken cl100k_base encoding: {e}")
    return _ENCODING


def count_tokens(text: str) -> int:
    """计算文本 token 数。"""
    if not text:
        return 0
    return len(get_encoding().encode(text))


def count_message_tokens(message: dict) -> int:
    """计算单条消息的 token 数。

    包含 role 标记开销（约 4 tokens/条）和 content + tool_calls 的 token。
    """
    enc = get_encoding()
    total = 4
    content = str(message.get("content", ""))
    if content:
        total += len(enc.encode(content))
    tool_calls = message.get("tool_calls")
    if tool_calls:
        total += len(enc.encode(json.dumps(tool_calls, ensure_ascii=False, default=str)))
    return total


def count_messages_tokens(messages: list[dict]) -> int:
    """计算消息列表的总 token 数。"""
    return sum(count_message_tokens(m) for m in messages)
