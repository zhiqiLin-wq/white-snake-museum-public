"""DeepSeek LLM 提供者 — OpenAI 兼容 API，基于 GenericOpenAIProvider。"""
import logging
from typing import Optional

from .openai_compatible_provider import GenericOpenAIProvider
from ..config import settings

logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(GenericOpenAIProvider):
    """DeepSeek LLM 提供者 (OpenAI 兼容协议)。

    继承自 GenericOpenAIProvider，默认使用 DeepSeek 的 API 地址和密钥。
    """

    def __init__(self):
        super().__init__(
            api_key=settings.deepseek_api_key,
            base_url=DEEPSEEK_BASE_URL,
            default_model=settings.deepseek_model,
            fast_model=settings.deepseek_fast_model,
        )
