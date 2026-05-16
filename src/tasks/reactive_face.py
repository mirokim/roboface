"""짧은 표정 반응 (micro-expressions) — 큰 자극이 오면 잠깐 놀라거나 반가워하기.

PRESENCE_NEW, 큰 소리, 큰 모션 등의 이벤트에 짧게 (300~500ms) 표정 깜빡이고
원래 표정으로 복귀. mood_drift나 state 핸들러를 영구적으로 건드리지 않음.

활용:
    from src.tasks.reactive_face import flash_expression
    flash_expression(face, SURPRISED, 0.4)  # 400ms 후 자동 복귀
"""

from __future__ import annotations

import asyncio
import time

from src.face.expressions import Expression
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("reactive_face")

# 동시에 여러 reactive task 실행되지 않게 — 마지막 것만 유효
_active_task: asyncio.Task | None = None
_lock_until: float = 0.0


def is_locked() -> bool:
    """현재 reactive 표정 표시 중인지 — mood_drift 등이 양보할 때 사용."""
    return time.monotonic() < _lock_until


async def _flash(face: FaceState, expr: Expression, duration: float) -> None:
    global _lock_until
    saved = face.expression
    face.apply_expression(expr)
    _lock_until = time.monotonic() + duration
    try:
        await asyncio.sleep(duration)
    finally:
        # 그 사이 다른 곳에서 표정 바꿨으면 덮어쓰지 않음
        if face.expression.name == expr.name:
            face.apply_expression(saved)
        _lock_until = 0.0


def flash_expression(
    face: FaceState,
    expr: Expression,
    duration: float = 0.4,
) -> None:
    """비동기로 짧은 표정 반응 발사. 이전 reactive가 끝나지 않았으면 교체."""
    global _active_task
    if _active_task is not None and not _active_task.done():
        _active_task.cancel()
    try:
        _active_task = asyncio.create_task(_flash(face, expr, duration))
    except RuntimeError:
        # 이벤트 루프 없음 (테스트 등) — 무시
        log.debug("flash_expression: no running loop")
