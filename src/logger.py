# 日志模块：统一记录每次运行的时间、参数与结果
# 作用：论文写作时能翻日志确认“某个数字是哪次跑出来的”
# 设计：整个程序只开一个日志文件（每次运行一个），各模块共用，避免刷出一堆小文件

import logging      # 标准库日志
import sys          # 用于同时输出到控制台
from datetime import datetime  # 生成时间戳文件名

from . import config  # 读取日志目录配置

_CONFIGURED = False   # 标记：本次运行是否已经配置过日志（保证只配一次）
_RUN_FILE = None      # 本次运行的日志文件路径（供日志里打印）


def get_logger(name: str = "waterbloom", level: int = logging.INFO) -> logging.Logger:
    """返回模块专属的 logger，但所有模块共用同一个日志文件。

    原理：把 handler 挂在“根日志器”上，子日志器默认向上传播，
         所以无论哪个模块打日志，都会写进同一个文件。

    参数：
        name  : 日志器名称，一般用模块名（如 src.ml_models）
        level : 日志级别，默认 INFO
    返回：
        logging.Logger 对象
    """
    global _CONFIGURED, _RUN_FILE

    if not _CONFIGURED:                       # 只有第一次调用才真正配置
        config.ensure_dirs()                  # 先确保 logs 目录存在
        _RUN_FILE = config.LOG_DIR / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

        fmt = logging.Formatter(              # 统一格式：时间 | 级别 | 模块 | 内容
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        root = logging.getLogger()            # 根日志器，全局唯一
        root.setLevel(level)

        fh = logging.FileHandler(_RUN_FILE, encoding="utf-8")  # 文件输出
        fh.setFormatter(fmt)
        root.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)  # 控制台输出，实时看进度
        ch.setFormatter(fmt)
        root.addHandler(ch)

        _CONFIGURED = True                    # 标记已配置，后续调用直接返回子日志器

    return logging.getLogger(name)            # 子日志器会向上传播到根日志器


def current_log_file() -> str:
    """返回本次运行的日志文件路径（未配置时返回空字符串）。"""
    return str(_RUN_FILE) if _RUN_FILE else ""


def log_config(logger: logging.Logger) -> None:
    """把当前参数快照写进日志，作为实验的可追溯凭证。"""
    for k, v in config.snapshot().items():    # 遍历参数快照
        logger.info("参数 %-18s = %s", k, v)  # 逐行输出，便于检索
