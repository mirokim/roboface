"""Idle 애니메이션 — 가만히 있을 때 가끔 두리번 + 미세 동작.

WATCHING/IDLE 상태에서 비동기로 동작.
"""

from __future__ import annotations

import asyncio
import random

from src.config import BEHAVIOR
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("idle_anim")


async def run_idle_gaze(face: FaceState) -> None:
    """무한루프: 가끔 시선 살짝 움직임."""
    while True:
        wait = random.uniform(
            BEHAVIOR.idle_look_min_interval_sec,
            BEHAVIOR.idle_look_max_interval_sec,
        )
        await asyncio.sleep(wait)

        # 무작위 방향으로 살짝 응시 (-0.6 ~ +0.6)
        dx = random.uniform(-0.6, 0.6)
        dy = random.uniform(-0.3, 0.3)
        face.eye_state.gaze_x = dx
        face.eye_state.gaze_y = dy
        log.debug(f"idle gaze → ({dx:.2f}, {dy:.2f})")

        # 잠깐 응시
        hold = BEHAVIOR.idle_look_duration_ms / 1000
        await asyncio.sleep(hold)

        # 다시 정면으로
        face.eye_state.gaze_x = 0.0
        face.eye_state.gaze_y = 0.0
