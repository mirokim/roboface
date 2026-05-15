"""미리 정의된 머리 동작 시퀀스 — 모두 async."""

from __future__ import annotations

import asyncio

from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
from src.motion.servos import ServoController


async def nod(servos: ServoController, times: int = 2) -> None:
    """끄덕끄덕."""
    for _ in range(times):
        await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG + 15, 0.25)
        await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG, 0.25)


async def shake(servos: ServoController, times: int = 2) -> None:
    """좌우 흔들기 (아니오)."""
    for _ in range(times):
        await servos.smooth_to_async(PAN_CENTER_DEG - 20, TILT_CENTER_DEG, 0.25)
        await servos.smooth_to_async(PAN_CENTER_DEG + 20, TILT_CENTER_DEG, 0.25)
    await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG, 0.2)


async def look_around(servos: ServoController) -> None:
    """두리번."""
    await servos.smooth_to_async(PAN_CENTER_DEG - 30, TILT_CENTER_DEG, 0.5)
    await asyncio.sleep(0.3)
    await servos.smooth_to_async(PAN_CENTER_DEG + 30, TILT_CENTER_DEG, 0.7)
    await asyncio.sleep(0.3)
    await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG, 0.4)


async def greeting(servos: ServoController) -> None:
    """인사 — 살짝 위로 들었다 끄덕."""
    await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG - 10, 0.3)
    await asyncio.sleep(0.1)
    await nod(servos, times=1)


async def tilt_curious(servos: ServoController) -> None:
    """호기심 — 머리 살짝 기울임."""
    await servos.smooth_to_async(PAN_CENTER_DEG + 10, TILT_CENTER_DEG - 5, 0.4)


async def home(servos: ServoController) -> None:
    """중앙으로 복귀."""
    await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG, 0.5)
