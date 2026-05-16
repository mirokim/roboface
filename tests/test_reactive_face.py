"""reactive_face flash 테스트."""

from __future__ import annotations

import asyncio

from src.face.expressions import NEUTRAL, SURPRISED, HAPPY
from src.face.renderer import FaceState
from src.tasks import reactive_face


def test_flash_restores_original_after_duration():
    face = FaceState(expression=NEUTRAL)

    async def go():
        reactive_face.flash_expression(face, SURPRISED, 0.1)
        await asyncio.sleep(0.05)
        assert face.expression.name == "surprised"
        assert reactive_face.is_locked()
        await asyncio.sleep(0.15)

    asyncio.run(go())
    assert face.expression.name == "neutral"
    assert not reactive_face.is_locked()


def test_flash_replaces_in_progress():
    """직전 flash 끝나기 전에 새 flash 들어오면 새 것이 이김."""
    face = FaceState(expression=NEUTRAL)

    async def go():
        reactive_face.flash_expression(face, SURPRISED, 0.5)
        await asyncio.sleep(0.02)
        reactive_face.flash_expression(face, HAPPY, 0.1)
        await asyncio.sleep(0.05)
        assert face.expression.name == "happy"
        await asyncio.sleep(0.15)
        # HAPPY가 끝나면 원래 (NEUTRAL)로 복귀
        assert face.expression.name == "neutral"

    asyncio.run(go())


def test_flash_does_not_overwrite_if_someone_else_changed():
    """flash 중에 다른 곳에서 표정 바꿔놓으면 그것을 존중 (덮어쓰지 않음)."""
    face = FaceState(expression=NEUTRAL)

    async def go():
        reactive_face.flash_expression(face, SURPRISED, 0.1)
        await asyncio.sleep(0.03)
        # 외부에서 다른 표정 set
        face.apply_expression(HAPPY)
        await asyncio.sleep(0.2)
        # flash가 자기 SURPRISED만 알기에 — HAPPY가 보존돼야 함
        assert face.expression.name == "happy"

    asyncio.run(go())


def test_is_locked_without_active_flash():
    assert reactive_face.is_locked() is False
