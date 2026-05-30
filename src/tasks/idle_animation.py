"""Idle 애니메이션 — 가만히 있을 때 가끔 두리번 + 미세 동작.

WATCHING/IDLE 상태에서 비동기로 동작.
"""

from __future__ import annotations

import asyncio
import random

from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.config import BEHAVIOR
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.utils.logger import get_logger

log = get_logger("idle_anim")


async def run_idle_gaze(
    face: FaceState,
    perception: PerceptionState | None = None,
) -> None:
    """무한루프: 가끔 시선 살짝 움직임.

    perception이 주어지고 사람이 보이면 skip — eye_tracker에 양보.
    """
    while True:
        wait = random.uniform(
            BEHAVIOR.idle_look_min_interval_sec,
            BEHAVIOR.idle_look_max_interval_sec,
        )
        await asyncio.sleep(wait)

        # 사람이 보이면 idle 두리번은 양보 — eye_tracker가 사용자 방향으로 lock
        if perception is not None and perception.person_present:
            continue

        # 무작위 방향으로 살짝 응시 — 좀 차분하게 (-0.4 ~ +0.4)
        dx = random.uniform(-0.4, 0.4)
        dy = random.uniform(-0.2, 0.2)
        face.eye_state.gaze_x = dx
        face.eye_state.gaze_y = dy
        log.debug(f"idle gaze → ({dx:.2f}, {dy:.2f})")

        # 잠깐 응시
        hold = BEHAVIOR.idle_look_duration_ms / 1000
        await asyncio.sleep(hold)

        # 다시 정면으로
        face.eye_state.gaze_x = 0.0
        face.eye_state.gaze_y = 0.0


# === Ambient motion (백그라운드 미세 댄스) ===
# 평소에 가끔 살랑살랑 — "살아있는 느낌". 대화/알림 중이면 양보.
# 사람이 보이면 짧고 작게, 안 보이면 좀 더 길고 크게.

_AMBIENT_MIN_INTERVAL_SEC = 20.0
_AMBIENT_MAX_INTERVAL_SEC = 60.0

# 가끔(12%) bpm/진폭 키워 "신난 살랑이"
_AMBIENT_LIVELY_PROB = 0.12

# 동작 막아야 하는 상태들 (대화/발화/알림 중엔 sway 금지)
_BLOCKED_STATES = (
    State.TALKING,
    State.LISTENING,
    State.GREETING,
    State.ALERTING,
)


async def run_ambient_motion(
    servos: ServoController,
    ctx: StateContext,
) -> None:
    """평상시 가끔 sway 모션 발동.

    - state가 _BLOCKED_STATES면 skip
    - 사용자 있을 때: 빈도 ↓ + 진폭 ↓ — sway 후 head_tracker 복귀 시
      '갑자기 휙 돌림' 패턴 완화. 곁눈질 정도만.
    - 사용자 없을 때: 좀 더 자유롭게 (2-4 beats, 일반 진폭)
    """
    log.info("ambient motion 시작")
    while True:
        # 사용자 있을 때 빈도 절반(40-120s), 없을 때 기본(20-60s)
        if ctx.user_present:
            wait = random.uniform(40.0, 120.0)
        else:
            wait = random.uniform(_AMBIENT_MIN_INTERVAL_SEC, _AMBIENT_MAX_INTERVAL_SEC)
        await asyncio.sleep(wait)

        if ctx.state in _BLOCKED_STATES:
            continue

        # 사용자가 있으면 head_tracker를 너무 길게 끊지 않게 짧고 작게.
        # 진폭 1.5-2.5° — 머리 거의 안 움직이는 미세 sway. 복귀 시 점프 X.
        if ctx.user_present:
            params = dict(
                bpm=random.randint(40, 55),
                beats=1,
                pan_amp_deg=random.uniform(1.5, 2.5),
                tilt_amp_deg=random.uniform(0.5, 1.0),
            )
        elif random.random() < _AMBIENT_LIVELY_PROB:
            params = dict(bpm=80, beats=6, pan_amp_deg=10.0, tilt_amp_deg=3.0)
        else:
            params = dict(
                bpm=random.randint(45, 65),
                beats=random.choice([2, 3, 4]),
                pan_amp_deg=random.uniform(4.0, 7.0),
                tilt_amp_deg=random.uniform(1.0, 2.5),
            )

        async with motion_busy_scope(ctx):
            try:
                await poses.sway(servos, **params)
            except Exception as e:
                log.warning(f"sway 에러: {e}")
