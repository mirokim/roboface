"""중앙 로거. loguru를 얇게 감쌈."""

from __future__ import annotations

import sys

from loguru import logger as _logger

from src.config import LOG_LEVEL

# 기본 핸들러 제거 후 재구성
_logger.remove()
_logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)


def get_logger(name: str = "roboface"):
    return _logger.bind(name=name)
