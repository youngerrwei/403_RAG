"""
统一日志模块 - LAB 403 RAG 系统
支持同时输出到控制台和日志文件，按日期自动轮转
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

# ======== 日志配置（可通过 .env 覆盖）========
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
LOG_FILE_PREFIX = os.getenv("LOG_FILE_PREFIX", "rag")
LOG_MAX_DAYS = int(os.getenv("LOG_MAX_DAYS", "7"))

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str = "rag") -> logging.Logger:
    """
    获取指定名称的 Logger 实例
    - 同时输出到控制台和日志文件
    - 日志文件按天轮转，保留 LOG_MAX_DAYS 天
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # 日志格式
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    logger.addHandler(console_handler)

    # 文件 Handler（按天轮转）
    log_file = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}_{name}.log")
    file_handler = TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=LOG_MAX_DAYS,
        encoding="utf-8"
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
    logger.addHandler(file_handler)

    return logger
