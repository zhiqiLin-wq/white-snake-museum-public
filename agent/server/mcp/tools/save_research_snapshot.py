"""P3-11: save_research_snapshot — 研究结论快照工具。

保存当前研究结论的可复现性快照到磁盘文件（JSON）。
包含: 结论文本 + 模型版本 + 温度 + max_tokens + chunk 版本 + prompt 模板版本
+ 当时的标注覆盖率统计 + 工具调用记录。

解决用户"chunk 变了之后之前的结论怎么审计"的诉求:
- 每次保存快照时冻结当时的系统参数和标注状态
- 下次 load 快照可以对比"当时的条件"和"现在的条件"是否一致
- 如果 chunk 版本或标注数变了 → 提示用户结论可能需要复核

存储路径: agent/snapshots/{user_id}/{timestamp}.json
"""
import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

SNAPSHOTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "snapshots"


def _ensure_dir(user_id: str) -> Path:
    """确保用户快照目录存在。"""
    safe_uid = "".join(c for c in user_id if c.isalnum() or c in "-_") or "default"
    d = SNAPSHOTS_DIR / safe_uid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_system_metadata() -> dict:
    """收集当前系统参数快照。"""
    from ..config import settings as app_settings
    import importlib.metadata

    # prompt 模板版本（文件 mtime 作为版本标识）
    prompts_dir = Path(__file__).resolve().parent.parent.parent / "prompts" / "templates"
    prompt_versions = {}
    if prompts_dir.exists():
        for f in sorted(prompts_dir.glob("*.yaml")):
            prompt_versions[f.name] = {
                "mtime": int(f.stat().st_mtime),
                "size": f.stat().st_size,
            }

    return {
        "model": app_settings.llm_model,
        "fastModel": app_settings.effective_fast_model,
        "maxOutputTokens": app_settings.max_output_tokens,
        "temperature": getattr(app_settings, "llm_temperature", 0.5),
        "embeddingModel": getattr(app_settings, "embedding_model", "unknown"),
        "chunkVersion": getattr(app_settings, "chunk_version", "unknown"),
        "promptTemplateVersions": prompt_versions,
        "pythonVersion": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "savedAt": time.time(),
    }


TOOL_DEF = {
    "name": "save_research_snapshot",
    "description": (
        "保存研究结论快照到磁盘。冻结当时的系统参数（模型/温度/chunk版本/prompt模板版本）"
        "和标注状态（覆盖率统计），供未来审计复现性。"
        "典型场景：用户说「保存这个结论」「冻结当前分析结果」「我要存证」时调用。"
        "也可用于 list（列出历史快照）和 get（加载某条快照对比当前系统状态）。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "list", "get", "compare"],
                "description": (
                    "save=保存新快照; "
                    "list=列出历史快照; "
                    "get=加载指定快照详情; "
                    "compare=对比指定快照与当前系统状态（标注是否变了/chunk是否变了）"
                ),
            },
            "conclusion": {
                "type": "string",
                "description": "save 模式: 要冻结的结论文本（必填）",
            },
            "snapshot_id": {
                "type": "string",
                "description": "get/compare 模式: 快照 ID（timestamp 文件名，如 1709123456）",
            },
            "research_topic": {
                "type": "string",
                "description": "save 模式: 研究主题/标题（便于 list 时识别）",
            },
            "tool_calls_record": {
                "type": "array",
                "description": "save 模式: 工具调用记录（来自 agent_loop 返回的 tool_calls 字段）",
                "items": {"type": "object"},
            },
        },
        "required": ["action"],
    },
}


async def handler(action: str = "save",
                  conclusion: str = "",
                  snapshot_id: str = "",
                  research_topic: str = "",
                  tool_calls_record: list = None,
                  user_id: str = "",
                  node_client=None) -> dict:
    """MCP handler: 研究结论快照管理。"""
    action = (action or "save").strip()

    if action == "save":
        if not conclusion or not conclusion.strip():
            return {"error": "save 模式需要 conclusion 文本"}
        uid = user_id or "default"
        d = _ensure_dir(uid)
        ts = int(time.time())
        sys_meta = _get_system_metadata()

        # 如果有 node_client，拉当前标注覆盖率统计
        ann_stats = None
        if node_client:
            try:
                from .coverage_tracker import handler as cov_handler
                from functools import partial
                cov = await cov_handler(node_client=node_client)
                ann_stats = cov.get("summary")
            except Exception as ex:
                logger.warning(f"快照: 拉标注覆盖率失败（降级）: {ex}")

        snapshot = {
            "id": str(ts),
            "timestamp": ts,
            "timestampHuman": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "userId": uid,
            "researchTopic": research_topic or "(未命名)",
            "conclusion": conclusion,
            "systemMetadata": sys_meta,
            "annotationStats": ann_stats,
            "toolCallsRecord": tool_calls_record or [],
        }

        filepath = d / f"{ts}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

        logger.info(f"P3-11 快照已保存: {filepath} (topic={research_topic[:40]})")

        return {
            "result": f"快照已保存: ID={ts}, 文件={filepath.name}",
            "snapshotId": str(ts),
            "filepath": str(filepath),
            "researchTopic": research_topic or "(未命名)",
            "savedAt": snapshot["timestampHuman"],
        }

    elif action == "list":
        uid = user_id or "default"
        d = _ensure_dir(uid)
        snapshots = []
        for f in sorted(d.glob("*.json"), reverse=True):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                snapshots.append({
                    "id": data.get("id", f.stem),
                    "timestamp": data.get("timestampHuman", ""),
                    "researchTopic": data.get("researchTopic", ""),
                    "conclusionPreview": (data.get("conclusion", "") or "")[:100],
                    "model": data.get("systemMetadata", {}).get("model", ""),
                    "chunkVersion": data.get("systemMetadata", {}).get("chunkVersion", ""),
                    "annotationCoverage": (data.get("annotationStats", {}) or {}).get("coverageRate"),
                })
            except (json.JSONDecodeError, OSError) as ex:
                logger.warning(f"读取快照 {f.name} 失败: {ex}")

        return {
            "snapshots": snapshots,
            "total": len(snapshots),
            "directory": str(d),
        }

    elif action == "get":
        if not snapshot_id:
            return {"error": "get 模式需要 snapshot_id"}
        uid = user_id or "default"
        d = _ensure_dir(uid)
        filepath = d / f"{snapshot_id}.json"
        if not filepath.exists():
            return {"error": f"快照 {snapshot_id} 不存在"}
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    elif action == "compare":
        if not snapshot_id:
            return {"error": "compare 模式需要 snapshot_id"}
        uid = user_id or "default"
        d = _ensure_dir(uid)
        filepath = d / f"{snapshot_id}.json"
        if not filepath.exists():
            return {"error": f"快照 {snapshot_id} 不存在"}
        with open(filepath, "r", encoding="utf-8") as f:
            old = json.load(f)

        current_meta = _get_system_metadata()
        old_meta = old.get("systemMetadata", {})

        changes = []

        # 模型变化
        if old_meta.get("model") != current_meta.get("model"):
            changes.append({
                "field": "model",
                "oldValue": old_meta.get("model"),
                "newValue": current_meta.get("model"),
                "severity": "high",
                "message": f"模型已变更: {old_meta.get('model')} → {current_meta.get('model')}。结论可能需要用新模型重新验证。",
            })

        # chunk 版本变化
        if old_meta.get("chunkVersion") != current_meta.get("chunkVersion"):
            changes.append({
                "field": "chunkVersion",
                "oldValue": old_meta.get("chunkVersion"),
                "newValue": current_meta.get("chunkVersion"),
                "severity": "critical",
                "message": f"Chunk 版本已变更: {old_meta.get('chunkVersion')} → {current_meta.get('chunkVersion')}。文献切分结果不同，引用的章节/段落可能已偏移。",
            })

        # max_tokens 变化
        if old_meta.get("maxOutputTokens") != current_meta.get("maxOutputTokens"):
            changes.append({
                "field": "maxOutputTokens",
                "oldValue": old_meta.get("maxOutputTokens"),
                "newValue": current_meta.get("maxOutputTokens"),
                "severity": "low",
                "message": f"生成长度上限变化: {old_meta.get('maxOutputTokens')} → {current_meta.get('maxOutputTokens')}。",
            })

        # prompt 模板变化
        old_prompts = old_meta.get("promptTemplateVersions", {})
        current_prompts = current_meta.get("promptTemplateVersions", {})
        for name in sorted(set(old_prompts.keys()) | set(current_prompts.keys())):
            old_v = old_prompts.get(name, {})
            cur_v = current_prompts.get(name, {})
            if old_v.get("mtime") != cur_v.get("mtime"):
                changes.append({
                    "field": f"prompt:{name}",
                    "oldMtime": old_v.get("mtime"),
                    "newMtime": cur_v.get("mtime"),
                    "severity": "medium",
                    "message": f"Prompt 模板 {name} 已修改。系统提示规则可能不同。",
                })

        # 标注覆盖率变化
        old_cov = (old.get("annotationStats", {}) or {}).get("coverageRate")
        if old_cov is not None and node_client:
            try:
                from .coverage_tracker import handler as cov_handler
                cov = await cov_handler(node_client=node_client)
                new_cov = cov.get("summary", {}).get("coverageRate")
                if new_cov is not None and abs(new_cov - old_cov) > 0.01:
                    changes.append({
                        "field": "annotationCoverage",
                        "oldValue": old_cov,
                        "newValue": new_cov,
                        "severity": "medium",
                        "message": f"标注覆盖率从 {old_cov:.1%} 变为 {new_cov:.1%}。标注数量变化可能影响结论。",
                    })
            except Exception:
                pass

        return {
            "snapshotId": snapshot_id,
            "snapshotTopic": old.get("researchTopic", ""),
            "snapshotDate": old.get("timestampHuman", ""),
            "changes": changes,
            "changeCount": len(changes),
            "isReproducible": len(changes) == 0,
            "summary": ("✅ 系统状态与快照时一致，结论可复现。" if len(changes) == 0
                        else f"⚠️ 检测到 {len(changes)} 处变化，结论可能需要复核。"),
        }

    return {"error": f"未知 action={action!r}"}
