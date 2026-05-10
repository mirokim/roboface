"""미리 정의된 머리 동작 시퀀스."""

from __future__ import annotations

import asyncio

from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
from src.motion.servos import ServoController


async def nod(servos: ServoController, times: int = 2) -> None:
    """끄덕끄덕."""
    for _ in range(times):
        servos.smooth_to(PAN_CENTER_DEG, TILT_CENTER_DEG + 15, duration_sec=0.25)
        servos.smooth_to(PAN_CENTER_DEG, TILT_CENTER_DEG, duration_sec=0.25)


async def shake(servos: ServoController, times: int = 2) -> None:
    """좌우로 흔들기 (아니오)."""
    for _ in range(times):
        servos.smooth_to(PAN_CENTER_DEG - 20, TILT_CENTER_DEG, duration_sec=0.25)
        servos.smooth_to(PAN_CENTER_DEG + 20, TILT_CENTER_DEG, duration_sec=0.25)
    servos.smooth_to(PAN_CENTER_DEG, TILT_CENTER_DEG, duration_sec=0.2)


async def look_around(servos: ServoController) -> None:
    """두리번."""
    servos.smooth_to(PAN_CENTER_DEG - 30, TILT_CENTER_DEG, duration_sec=0.5)
    await asyncio.sleep(0.3)
    servos.smooth_to(PAN_CENTER_DEG + 30, TILT_CENTER_DEG, duration_sec=0.7)
    await asyncio.sleep(0.3)
    servos.smooth_to(PAN_CENTER_DEG, TILT_CENTER_DEG, duration_sec=0.4)


async def greeting(servos: ServoController) -> None:
    """인사 — 살짝 위로 들었다가 끄덕."""
    servos.smooth_to(PAN_CENTER_DEG, TILT_CENTER_DEG - 10, duration_sec=0.3)
    await asyncio.sleep(0.1)
    await nod(servos, times=1)


async def tilt_curious(servos: ServoController) -> None:
    """호기심 — 머리 살짝 기울임."""
    servos.smooth_to(PAN_CENTER_DEG + 10, TILT_CENTER_DEG - 5, duration_sec=0.4)
