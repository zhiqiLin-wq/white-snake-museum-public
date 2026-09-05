"""Persona 配置加载与不可变区保护。

从 persona.yaml 加载结构化模板。
提供 replace/append 方法，自动保护不可变区字段。

不可变区: core_identity, knowledge_boundary
可变区: communication_style, user_preferences, interaction_patterns

对于 replace 操作:
- 新内容中的 不可变区字段 -> 被忽略（保留模板原始值）
- 新内容中的 可变区字段 -> 正常覆盖

对于 append 操作:
- 内容追加到指定可变区字段
- 不可变区字段 -> 拒绝操作
"""

import copy
import json
import logging
from pathlib import Path
from typing import Optional

from ..llm.json_utils import parse_llm_json

logger = logging.getLogger(__name__)

# 不可变区字段名（persona 块）
_PERSONA_IMMUTABLE_FIELDS = {"core_identity", "knowledge_boundary"}

# 可变区字段名（persona 块）
_PERSONA_MUTABLE_FIELDS = {
    "communication_style", "user_preferences", "interaction_patterns", "_notes",
}

# human 块全部可变
_HUMAN_MUTABLE_FIELDS = {
    "personal_info", "research_interests", "interaction_style",
    "preferences", "constraints", "_notes",
}

# 最大 token 限制（每块 ~2000 tokens）
MAX_TOKENS_PER_BLOCK = 2000


class PersonaLoader:
    """Persona 模板加载与合并引擎。

    核心职责:
    1. 加载 persona.yaml 模板（包含不可变区锚定值）
    2. 在与 Core Memory 当前值合并时，保护不可变区不被覆盖
    3. 提供 replace/append 接口供 MCP 工具调用
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Args:
            config_path: persona.yaml 的路径。
                         默认: agent/server/config/persona.yaml
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "persona.yaml"
        self._config_path = config_path
        self._template = self._load_template()

    def _load_template(self) -> dict:
        """加载 persona.yaml 模板。"""
        try:
            import yaml
            with open(self._config_path, "r", encoding="utf-8") as f:
                template = yaml.safe_load(f)
            logger.info("persona template loaded: %s", self._config_path)
            return template
        except ImportError:
            logger.warning("yaml not available, using default empty template")
            return {"persona": {}, "human": {}}
        except Exception as e:
            logger.error("failed to load persona template: %s", e)
            return {"persona": {}, "human": {}}

    @property
    def persona_template(self) -> dict:
        """返回 persona 块的模板内容。"""
        return copy.deepcopy(self._template.get("persona", {}))

    @property
    def human_template(self) -> dict:
        """返回 human 块的模板内容。"""
        return copy.deepcopy(self._template.get("human", {}))

    # ================================================================
    #  Replace: 替换整个块（保护不可变区）
    # ================================================================

    def replace(self, block: str, new_content: str, current_db_content: str = "{}") -> str:
        """替换 Core Memory 块内容。

        规则:
        - persona 块: 不可变区字段保留模板值，可变区字段用新内容覆盖
        - human 块: 全部可变，直接替换

        Args:
            block: "persona" | "human"
            new_content: 新内容（JSON 字符串）
            current_db_content: 数据库中当前的内容（JSON 字符串），用于合并

        Returns:
            合并后的 JSON 字符串
        """
        try:
            new_obj = json.loads(new_content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("core_memory_replace: invalid JSON for %s, treating as raw text", block)
            new_obj = {"_notes": [new_content]}

        try:
            current_obj = json.loads(current_db_content)
        except (json.JSONDecodeError, TypeError):
            current_obj = {}

        if block == "persona":
            result = self._replace_persona(new_obj, current_obj)
        elif block == "human":
            result = self._replace_human(new_obj, current_obj)
        else:
            raise ValueError(f"Unknown block: {block}")

        # Token 检查（fast fail：超限直接拒绝，不写入）
        result_str = json.dumps(result, ensure_ascii=False, indent=2)
        estimated_tokens = len(result_str) // 2  # 粗略: 中文 ~2 chars/token
        if estimated_tokens > MAX_TOKENS_PER_BLOCK:
            raise ValueError(
                f"core_memory {block}: estimated {estimated_tokens} tokens "
                f"exceeds limit {MAX_TOKENS_PER_BLOCK}"
            )

        return result_str

    def _replace_persona(self, new_obj: dict, current_obj: dict) -> dict:
        """替换 persona 块：不可变区受保护，未覆盖的可变字段保留当前值。

        以 current_obj（数据库当前内容）为基，只覆盖 new_obj 中出现的可变字段，
        避免 replace 把之前通过 append / save_to_memory 累积的字段清空。
        """
        template = self._template.get("persona", {})
        result = copy.deepcopy(current_obj) if current_obj else {}
        # 补齐模板中缺失的字段（例如首次 replace 时 current 为空）
        for field in template:
            if field not in result:
                result[field] = copy.deepcopy(template[field])

        # 不可变区：强制使用模板值（忽略 new 与 current 中的不可变区字段）
        for field in _PERSONA_IMMUTABLE_FIELDS:
            if field in template:
                result[field] = copy.deepcopy(template[field])
            if field in new_obj:
                logger.info(
                    "core_memory persona: immutable field '%s' blocked from modification",
                    field,
                )

        # 可变区：用新内容覆盖（仅覆盖 new_obj 中出现的字段）
        for field in _PERSONA_MUTABLE_FIELDS:
            if field in new_obj:
                result[field] = new_obj[field]

        return result

    def _replace_human(self, new_obj: dict, current_obj: dict) -> dict:
        """替换 human 块：全部可变，未覆盖字段保留当前值。"""
        template = self._template.get("human", {})
        result = copy.deepcopy(current_obj) if current_obj else {}
        for field in template:
            if field not in result:
                result[field] = copy.deepcopy(template[field])
        for field in _HUMAN_MUTABLE_FIELDS:
            if field in new_obj:
                result[field] = new_obj[field]
        return result

    # ================================================================
    #  Append: 向指定字段追加内容
    # ================================================================

    def append(self, block: str, field: str, content: str,
               current_db_content: str = "{}") -> str:
        """向 Core Memory 块的可变区字段追加内容。

        Args:
            block: "persona" | "human"
            field: 目标字段名（如 "user_preferences"、"constraints"）
            content: 要追加的内容（文本）
            current_db_content: 数据库中当前的内容（JSON 字符串）

        Returns:
            合并后的 JSON 字符串
        """
        # 检查不可变区
        if block == "persona" and field in _PERSONA_IMMUTABLE_FIELDS:
            raise ValueError(
                f"Cannot append to immutable field '{field}' in persona block"
            )

        try:
            current_obj = json.loads(current_db_content)
        except (json.JSONDecodeError, TypeError):
            current_obj = {}

        # 初始化模板
        if block == "persona":
            template = self._template.get("persona", {})
        else:
            template = self._template.get("human", {})

        result = copy.deepcopy(template)
        # 合并当前数据库中的可变字段
        for k, v in current_obj.items():
            if k in template:
                result[k] = v

        # 追加内容
        if field not in result:
            result[field] = {}

        if isinstance(result[field], dict):
            # 尝试解析 content 为 JSON
            try:
                snippet = parse_llm_json(content)
                if isinstance(snippet, dict):
                    result[field].update(snippet)
                elif isinstance(snippet, list):
                    if "_items" not in result[field]:
                        result[field]["_items"] = []
                    result[field]["_items"].extend(snippet)
                else:
                    if "_notes" not in result:
                        result["_notes"] = []
                    result["_notes"].append(content)
            except (json.JSONDecodeError, TypeError):
                if "_notes" not in result:
                    result["_notes"] = []
                result["_notes"].append(content)
        elif isinstance(result[field], list):
            result[field].append(content)
        else:
            if "_notes" not in result:
                result["_notes"] = []
            result["_notes"].append(
                json.dumps({field: content}, ensure_ascii=False)
            )

        result_str = json.dumps(result, ensure_ascii=False, indent=2)
        estimated_tokens = len(result_str) // 2
        if estimated_tokens > MAX_TOKENS_PER_BLOCK:
            raise ValueError(
                f"core_memory {block}.{field}: estimated {estimated_tokens} tokens "
                f"exceeds limit {MAX_TOKENS_PER_BLOCK}"
            )

        return result_str

    # ================================================================
    #  序列化（注入上下文前格式化）
    # ================================================================

    def format_for_context(self, block_content: str, block_name: str) -> str:
        """将 Core Memory 块内容格式化为可注入 LLM 上下文的文本。

        Args:
            block_content: JSON 字符串
            block_name: "persona" | "human"

        Returns:
            自然语言文本
        """
        try:
            obj = json.loads(block_content)
        except (json.JSONDecodeError, TypeError):
            return block_content

        if block_name == "persona":
            return self._format_persona(obj)
        else:
            return self._format_human(obj)

    def _format_persona(self, obj: dict) -> str:
        """将 persona JSON 格式化为自然语言。"""
        parts = []
        ci = obj.get("core_identity", {})
        if ci:
            parts.append(f"I am {ci.get('name', 'an AI assistant')}. "
                        f"My role is: {ci.get('role', '')}.")

        kb = obj.get("knowledge_boundary", {})
        if kb.get("domains"):
            parts.append(f"My expertise covers: {', '.join(kb['domains'])}.")
        if kb.get("exclusions"):
            parts.append(f"I do not: {'; '.join(kb['exclusions'])}.")

        cs = obj.get("communication_style", {})
        if cs:
            parts.append(f"I communicate in {cs.get('language', 'Chinese')} "
                        f"with a {cs.get('tone', 'professional')} tone.")

        up = obj.get("user_preferences", {})
        if up:
            pref_str = json.dumps(up, ensure_ascii=False)
            parts.append(f"User preferences: {pref_str}")

        return " ".join(parts)

    def _format_human(self, obj: dict) -> str:
        """将 human JSON 格式化为自然语言。"""
        parts = []
        pi = obj.get("personal_info", {})
        if pi:
            parts.append(f"User info: {json.dumps(pi, ensure_ascii=False)}")

        ri = obj.get("research_interests", {})
        if ri:
            parts.append(f"Research interests: {json.dumps(ri, ensure_ascii=False)}")

        pref = obj.get("preferences", {})
        if pref:
            parts.append(f"Preferences: {json.dumps(pref, ensure_ascii=False)}")

        constraints = obj.get("constraints", {})
        if constraints:
            parts.append(f"Constraints: {json.dumps(constraints, ensure_ascii=False)}")

        style = obj.get("interaction_style", {})
        if style:
            parts.append(f"Interaction style: {json.dumps(style, ensure_ascii=False)}")

        return " ".join(parts)

    def estimate_tokens(self, content: str) -> int:
        """粗略估算 token 数。"""
        return len(content) // 2
