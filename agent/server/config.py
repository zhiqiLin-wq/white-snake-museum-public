"""集中配置管理，从环境变量加载所有设置。"""
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

from .rag.config import RAGConfig

# agent/ 目录，用于解析 data_dir / chroma_persist_dir 等相对路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# .env 文件位于项目根目录（white-snake-museum -public/），而非 agent/ 下
ENV_FILE = PROJECT_ROOT.parent / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


@dataclass
class Settings:
    # ---- 服务 ----
    agent_port: int = int(os.getenv("AGENT_PORT", "8000"))
    agent_host: str = os.getenv("AGENT_HOST", "127.0.0.1")
    node_api_url: str = os.getenv("NODE_API_URL", "http://127.0.0.1:3000/api")

    # ---- LLM ----
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    anthropic_fast_model: str = os.getenv("ANTHROPIC_FAST_MODEL", "claude-haiku-3-5-20241022")

    # DeepSeek (OpenAI 兼容)
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_fast_model: str = os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash")

    # ---- RAG 管线（委托给 RAGConfig） ----
    rag: RAGConfig = field(default_factory=RAGConfig)

    # ---- 安全 ----
    agent_api_key: str = os.getenv("AGENT_API_KEY", "")
    rate_limit_per_second: float = float(os.getenv("RATE_LIMIT_PER_SECOND", "10.0"))
    rate_limit_burst: int = int(os.getenv("RATE_LIMIT_BURST", "20"))
    max_input_length: int = int(os.getenv("MAX_INPUT_LENGTH", "2000"))
    # 最终报告/长回答的 max_tokens 上限（DeepSeek 官方上限 8192）。
    # 只控制"上限"不会影响短回答的预算（短回答自然 stop），长分析不再被截断。
    # 通过 env var 可调，子图/agent_loop/provider 统一读此值，不再散点硬编码。
    max_output_tokens: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))

    # ---- 上下文管理 (v14) ----
    # FIFO 上下文队列 token 上限。向后兼容: 也检查旧环境变量名
    fifo_max_tokens: int = int(
        os.getenv("FIFO_MAX_TOKENS",
                  os.getenv("CONTEXT_MAX_TOKENS", "90000"))
    )
    # 记忆提取 + 摘要更新使用的模型（为空时 fallback 到 effective_fast_model）
    memory_extraction_model: str = os.getenv(
        "MEMORY_EXTRACTION_MODEL",
        os.getenv("CONTEXT_SUMMARY_MODEL", ""),
    )
    # 是否启用长期记忆提取
    memory_extraction_enabled: bool = (
        os.getenv("MEMORY_EXTRACTION_ENABLED", "true").lower() == "true"
    )
    # 是否启用 Recall Storage
    recall_storage_enabled: bool = (
        os.getenv("RECALL_STORAGE_ENABLED", "true").lower() == "true"
    )
    # 单条 tool_result 最大字符数（写入 LLM context 的副本；超出时按字段感知精简，
    # 保留元信息与列表前 N 条。get_chapter_full_text 等整章文本工具另有独立预算覆盖）
    context_tool_result_max_chars: int = int(os.getenv("CONTEXT_TOOL_RESULT_MAX_CHARS", "4000"))
    # ReAct 上下文护栏: messages 总 token 超过此值时压缩早期工具结果（v16）
    context_guardrail_tokens: int = int(os.getenv("CONTEXT_GUARDRAIL_TOKENS", "32000"))

    # ---- 记忆系统 (v15) ----
    # 模型上下文窗口总上限（所有注入内容加总：system prompt + Core Memory + 摘要 + FIFO + 被动注入）
    context_window_tokens: int = int(
        os.getenv("CONTEXT_WINDOW_TOKENS", "120000")
    )
    # 被动注入 token 预算（每次请求注入的记忆内容上限，作为动态预算的上限）
    memory_injection_token_budget: int = int(
        os.getenv("MEMORY_INJECTION_TOKEN_BUDGET", "12000")
    )
    # 是否启用被动注入（每次请求自动检索并注入相关记忆）
    memory_passive_injection_enabled: bool = (
        os.getenv("MEMORY_PASSIVE_INJECTION_ENABLED", "true").lower() == "true"
    )
    # 是否启用 Core Memory (persona/human 块)
    core_memory_enabled: bool = (
        os.getenv("CORE_MEMORY_ENABLED", "true").lower() == "true"
    )
    # Core Memory persona 配置文件路径（相对于 agent/server/）
    core_memory_persona_path: str = os.getenv(
        "CORE_MEMORY_PERSONA_PATH",
        str(PROJECT_ROOT / "server" / "core_memory" / "persona.yaml"),
    )
    # 是否启用后台记忆维护任务（合并/衰减/补偿）
    background_memory_tasks_enabled: bool = (
        os.getenv("BACKGROUND_MEMORY_TASKS_ENABLED", "true").lower() == "true"
    )
    # 当前用户 ID（用于 MemoryDB 路径解析，测试时可覆盖）
    memory_user_id: str = os.getenv("MEMORY_USER_ID", "")

    # ---- 地图 API ----
    amap_api_key: str = os.getenv("AMAP_API_KEY", "")

    # ---- 标注优化 (v11) ----
    annotation_rule_prescan_enabled: bool = (
        os.getenv("ANNOTATION_RULE_PRESCAN_ENABLED", "true").lower() == "true"
    )
    annotation_batch_size: int = int(os.getenv("ANNOTATION_BATCH_SIZE", "5"))
    annotation_batch_max_chars: int = int(os.getenv("ANNOTATION_BATCH_MAX_CHARS", "8000"))
    annotation_empty_retry_enabled: bool = (
        os.getenv("ANNOTATION_EMPTY_RETRY_ENABLED", "true").lower() == "true"
    )
    annotation_prefilter_enabled: bool = (
        os.getenv("ANNOTATION_PREFILTER_ENABLED", "true").lower() == "true"
    )
    annotation_prefilter_min_chars: int = int(
        os.getenv("ANNOTATION_PREFILTER_MIN_CHARS", "15")
    )
    annotation_quality_metrics_enabled: bool = (
        os.getenv("ANNOTATION_QUALITY_METRICS_ENABLED", "true").lower() == "true"
    )
    annotation_gleaning_enabled: bool = (
        os.getenv("ANNOTATION_GLEANING_ENABLED", "true").lower() == "true"
    )
    """标注后追加一轮查漏(gleaning)，追问 LLM 找出漏掉的实体，提升召回。"""

    # ---- 新标注流水线配置 (阶段 0 — CFG-01) ----

    # Pass 1 Discovery 配置
    annotation_discovery_window_size: int = int(
        os.getenv("ANNOTATION_DISCOVERY_WINDOW_SIZE", "3")
    )
    """Discovery 滑动窗口大小（段落数）。"""

    annotation_discovery_overlap: int = int(
        os.getenv("ANNOTATION_DISCOVERY_OVERLAP", "1")
    )
    """Discovery 窗口间重叠段落数。"""

    annotation_discovery_max_tokens: int = int(
        os.getenv("ANNOTATION_DISCOVERY_MAX_TOKENS", "4096")
    )
    """Discovery LLM 调用的 max_tokens。"""

    annotation_discovery_concurrency: int = int(
        os.getenv("ANNOTATION_DISCOVERY_CONCURRENCY", "5")
    )
    """Pass 1 窗口级并发数。"""

    # Pass 2 Precise Resolution 配置
    annotation_precise_concurrency: int = int(
        os.getenv("ANNOTATION_PRECISE_CONCURRENCY", "8")
    )
    """Pass 2 段落级并发数。"""

    # 质量保证配置
    annotation_quality_coverage_excellent: float = float(
        os.getenv("ANNOTATION_QUALITY_COVERAGE_EXCELLENT", "0.90")
    )
    annotation_quality_coverage_good: float = float(
        os.getenv("ANNOTATION_QUALITY_COVERAGE_GOOD", "0.75")
    )
    annotation_quality_coverage_fair: float = float(
        os.getenv("ANNOTATION_QUALITY_COVERAGE_FAIR", "0.50")
    )

    # 批注配置
    annotation_marginalia_max_count: int = int(
        os.getenv("ANNOTATION_MARGINALIA_MAX_COUNT", "50")
    )
    """单次批注生成的最大实体数。超过时提示用户可继续。"""

    annotation_marginalia_batch_size: int = int(
        os.getenv("ANNOTATION_MARGINALIA_BATCH_SIZE", "8")
    )
    """批注流式生成的每批实体数。"""

    annotation_marginalia_concurrency: int = int(
        os.getenv("ANNOTATION_MARGINALIA_CONCURRENCY", "5")
    )
    """批注生成的并发数。"""

    @property
    def is_llm_configured(self) -> bool:
        if self.llm_provider == "deepseek":
            return bool(self.deepseek_api_key and self.deepseek_api_key != "sk-...")
        return bool(self.anthropic_api_key and self.anthropic_api_key != "sk-ant-...")

    @property
    def effective_model(self) -> str:
        """根据当前 provider 返回正确的主模型名。"""
        if self.llm_provider == "deepseek":
            return self.deepseek_model
        return self.anthropic_model

    @property
    def effective_fast_model(self) -> str:
        """根据当前 provider 返回正确的快速模型名。"""
        if self.llm_provider == "deepseek":
            return self.deepseek_fast_model
        return self.anthropic_fast_model

    def __getattr__(self, name):
        """向后兼容：已迁移到 RAGConfig 的字段通过 self.rag 委托访问。"""
        if name != 'rag' and hasattr(self.rag, name):
            return getattr(self.rag, name)
        raise AttributeError(f"'Settings' object has no attribute '{name}'")


settings = Settings()
