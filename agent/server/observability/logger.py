"""结构化日志（控制台 + 按天切分的文件）。"""
import logging
import os
import sys
import json
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler


def setup_logging(level: int = logging.INFO):
    """配置 JSON 行式日志（仅控制台 handler）。"""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PlainFormatter())
    root = logging.getLogger()
    root.setLevel(level)
    # 清除已有 handler，避免重复
    root.handlers = [handler]
    return root


def setup_file_logging(log_dir: str = None, level: int = logging.INFO):
    """追加文件日志 handler（按天切分，保留 30 天）。

    控制台输出用易读的 PlainFormatter，文件日志用 JSON 行式（方便检索）。
    不存在的目录会自动创建（对应 Experience: 787300 的坑）。
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    log_dir = os.path.abspath(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "agent.log")

    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",          # 每天 0 点切分
        interval=1,
        backupCount=30,            # 保留最近 30 天
        encoding="utf-8",
        delay=False,
    )
    file_handler.suffix = "%Y%m%d"   # 切分后文件名：agent.log.20260829
    file_handler.setFormatter(_JsonFormatter())
    file_handler.setLevel(level)

    root = logging.getLogger()
    root.addHandler(file_handler)
    root.info(f"[LOG] 文件日志已启用: {log_file} (保留 30 天，按天切分)")
    return log_dir


# ---- LLM 调用独立日志（单独文件，便于单独检索）----

# LLM 专用 logger 的名字（固定，tracer.py 用这个来获取）
LLM_LOGGER_NAME = "llm_calls"


def get_llm_logger() -> logging.Logger:
    """获取 LLM 调用专用 logger。"""
    return logging.getLogger(LLM_LOGGER_NAME)


def setup_llm_logging(log_dir: str = None, level: int = logging.INFO) -> str:
    """设置 LLM 调用专用日志文件。

    与 agent.log 完全隔离：
    - 目录：<log_dir>/llm/  子目录（一眼看出是 LLM 相关）
    - 文件名：llm_calls.log（按天切分，切分后 llm_calls.log.YYYYMMDD）
    - 保留 60 天（比业务日志长，便于事后审计/比对 prompt 和响应）
    - 为避免 LLM 调用日志也被根 logger 的 agent.log 捕获，LLM logger 设为 propagate=False
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    base_log_dir = os.path.abspath(log_dir)
    llm_log_dir = os.path.join(base_log_dir, "llm")   # 单独子目录：logs/llm/
    os.makedirs(llm_log_dir, exist_ok=True)

    llm_log_file = os.path.join(llm_log_dir, "llm_calls.log")

    file_handler = TimedRotatingFileHandler(
        filename=llm_log_file,
        when="midnight",
        interval=1,
        backupCount=60,              # LLM 调用保留更久：60 天
        encoding="utf-8",
        delay=False,
    )
    file_handler.suffix = "%Y%m%d"   # llm_calls.log.20260829
    file_handler.setFormatter(_LLMJsonFormatter())
    file_handler.setLevel(level)

    logger = logging.getLogger(LLM_LOGGER_NAME)
    logger.handlers = []             # 清除旧 handlers
    logger.addHandler(file_handler)
    logger.setLevel(level)
    logger.propagate = False         # 关键：不向上传到根 logger，避免 agent.log 里也出现一份

    # 启动信息（同时打到根 logger 和 llm logger）
    msg = f"[LLM_LOG] LLM 调用日志已启用: {llm_log_file} (logs/llm/ 子目录, 保留 60 天, 按天切分)"
    logging.getLogger().info(msg)
    logger.info(msg)
    return llm_log_dir


class _LLMJsonFormatter(logging.Formatter):
    """LLM 日志专用 JSON 格式化（自带 [LLM_CALL] 前缀，grep 更方便）。"""
    def format(self, record):
        # record.getMessage() 就是 tracer.py 里拼好的 JSON 字符串（带全量 prompt）
        raw = record.getMessage()
        ts = datetime.now(timezone.utc).isoformat()
        # 为了 grep 起来舒服，每一行前缀加显式标识
        return f"[LLM_CALL] {ts} {raw}"


# ---- 工具调用独立日志（单独文件，便于单独检索）----

# 工具调用专用 logger 的名字（固定，tracer.py 用这个来获取）
TOOL_LOGGER_NAME = "tool_calls"


def get_tool_logger() -> logging.Logger:
    """获取工具调用专用 logger。"""
    return logging.getLogger(TOOL_LOGGER_NAME)


def setup_tool_logging(log_dir: str = None, level: int = logging.INFO) -> str:
    """设置工具调用专用日志文件。

    与 agent.log 完全隔离（与 LLM 日志同构）：
    - 目录：<log_dir>/tool/  子目录（一眼看出是工具调用相关）
    - 文件名：tool_calls.log（按天切分，切分后 tool_calls.log.YYYYMMDD）
    - 保留 60 天（与 LLM 日志一致，便于事后审计/比对工具输入输出）
    - propagate=False，避免 agent.log 里也出现一份
    - 每行带 [TOOL_CALL] 前缀，grep 友好
    """
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    base_log_dir = os.path.abspath(log_dir)
    tool_log_dir = os.path.join(base_log_dir, "tool")   # 单独子目录：logs/tool/
    os.makedirs(tool_log_dir, exist_ok=True)

    tool_log_file = os.path.join(tool_log_dir, "tool_calls.log")

    file_handler = TimedRotatingFileHandler(
        filename=tool_log_file,
        when="midnight",
        interval=1,
        backupCount=60,              # 与 LLM 日志一致：60 天
        encoding="utf-8",
        delay=False,
    )
    file_handler.suffix = "%Y%m%d"   # tool_calls.log.20260829
    file_handler.setFormatter(_ToolJsonFormatter())
    file_handler.setLevel(level)

    tl = logging.getLogger(TOOL_LOGGER_NAME)
    tl.handlers = []                # 清除旧 handlers
    tl.addHandler(file_handler)
    tl.setLevel(level)
    tl.propagate = False            # 关键：不向上传到根 logger，避免 agent.log 里也出现一份

    msg = f"[TOOL_LOG] 工具调用日志已启用: {tool_log_file} (logs/tool/ 子目录, 保留 60 天, 按天切分)"
    logging.getLogger().info(msg)
    tl.info(msg)
    return tool_log_dir


class _ToolJsonFormatter(logging.Formatter):
    """工具调用日志专用 JSON 格式化（自带 [TOOL_CALL] 前缀，grep 更方便）。"""
    def format(self, record):
        # record.getMessage() 就是 tracer.py 里拼好的 JSON 字符串（带全量输入输出）
        raw = record.getMessage()
        ts = datetime.now(timezone.utc).isoformat()
        return f"[TOOL_CALL] {ts} {raw}"


class _PlainFormatter(logging.Formatter):
    """控制台用：人类可读格式（和原 main.py basicConfig 一致）。"""
    def format(self, record):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{ts} [{record.levelname}] {record.name}: {record.getMessage()}"


class _JsonFormatter(logging.Formatter):
    """业务日志文件用：JSON 行式（方便 grep / 工具检索）。"""
    def format(self, record):
        obj = {
            "event": record.getMessage(),
            "level": record.levelname,
            "logger": record.name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if record.exc_info and record.exc_info[1]:
            obj["error"] = str(record.exc_info[1])
        return json.dumps(obj, ensure_ascii=False)


# 初始化（仅控制台 handler；文件 handler 由 main.py 在启动时追加）
setup_logging()
