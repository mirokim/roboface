"""미리 정의된 머리 동작 시퀀스 — 모두 async."""

from __future__ import annotations

import asyncio
import math
import random
import time

from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
from src.face import expressions as expr
from src.face.expressions import MouthShape
from src.face.renderer import FaceState
from src.motion.servos import ServoController
from src.utils.logger import get_logger

log = get_logger("poses")


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


# === 댄스 ===
_DANCE_EXPRESSIONS = [
    expr.HAPPY,
    expr.EXCITED,
    expr.STARSTRUCK,
    expr.LOVE,
    expr.WINK,
    expr.WINK_R,
    expr.PROUD,
    expr.CONTENT,
]

_DANCE_MOUTHS = [
    MouthShape.SMILE,
    MouthShape.GRIN,
    MouthShape.OPEN_SMALL,
    MouthShape.OPEN_MID,
    MouthShape.OPEN_LARGE,
    MouthShape.O,
]


async def dance(
    servos: ServoController,
    face: FaceState,
    *,
    bpm: int = 100,
    beats: int = 8,
    pan_amp_deg: float = 15.0,    # 25 → 15 (격함 줄이기)
    tilt_amp_deg: float = 6.0,    # 10 → 6
    update_hz: int = 30,
) -> None:
    """춤추는 듯한 머리 흔들기 + 표정/입 사이클.

    pan은 1박자 sine wave, tilt는 2배 빠른 bob.
    매 beat마다 mouth 변경, 매 2 beats마다 expression 변경.
    시작/끝은 ease-in/out envelope으로 자연스럽게.
    """
    log.info(f"💃 dance: {bpm} BPM × {beats} beats")
    period = 60.0 / bpm
    total = period * beats
    dt = 1.0 / update_hz

    saved_expr = face.expression
    saved_mouth = face.mouth_state.shape
    saved_amp = face.mouth_state.talk_amplitude

    expr_idx = random.randint(0, len(_DANCE_EXPRESSIONS) - 1)
    last_beat = -1

    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            if t > total:
                break

            phase = (t / period) * 2 * math.pi
            # envelope — 처음 한 박자 fade-in, 끝 한 박자 fade-out
            env = min(1.0, t / period) * min(1.0, (total - t) / period)
            env = max(0.0, env)

            d_pan = math.sin(phase) * pan_amp_deg * env
            # tilt는 2배 빠른 head bob (위→아래)
            d_tilt = math.sin(phase * 2 + math.pi / 2) * tilt_amp_deg * env * 0.6

            servos.set_angles(
                PAN_CENTER_DEG + d_pan,
                TILT_CENTER_DEG + d_tilt,
            )

            cur_beat = int(t / period)
            if cur_beat != last_beat:
                last_beat = cur_beat
                # 매 beat — 입 모양 바꿈
                face.mouth_state.shape = random.choice(_DANCE_MOUTHS)
                face.mouth_state.talk_amplitude = random.uniform(0.5, 1.0)
                # 매 2 beats — 표정 바꿈 (mouth는 다시 expression 기본값으로 덮이지 않게
                # apply_expression 후 mouth만 재설정)
                if cur_beat % 2 == 0:
                    next_expr = _DANCE_EXPRESSIONS[expr_idx % len(_DANCE_EXPRESSIONS)]
                    expr_idx += 1
                    # 표정만 갱신, blink trigger 안 함 (apply_expression이 매번
                    # 깜빡거리면 dance 중 LCD가 깜빡임처럼 보임)
                    face.expression = next_expr
                    face.eye_state.shape = next_expr.eye
                    face.mouth_state.shape = random.choice(_DANCE_MOUTHS)

            await asyncio.sleep(dt)
    finally:
        # 입/표정 복귀
        face.mouth_state.talk_amplitude = saved_amp
        face.mouth_state.shape = saved_mouth
        face.apply_expression(saved_expr)
        # 머리 중앙으로
        await servos.smooth_to_async(PAN_CENTER_DEG, TILT_CENTER_DEG, 0.4)


async def sway(
    servos: ServoController,
    *,
    bpm: int = 55,                # 70 → 55 (더 느긋)
    beats: int = 4,
    pan_amp_deg: float = 7.0,     # 10 → 7
    tilt_amp_deg: float = 2.0,    # 3 → 2
    update_hz: int = 25,
) -> None:
    """은은한 살랑살랑 — 표정/입은 건드리지 않음. ambient 백그라운드용.

    dance보다 진폭/속도 작고, beats도 짧음. envelope으로 자연스럽게 진입/종료.

    중요: base는 **현재 서보 위치** — CENTER 아님. head_tracker가 사용자 따라
    옆을 보고 있을 때 sway가 CENTER 기준이면 시작 시 옆→정면, 끝 시 정면→옆
    양방향 휙 발생. 현재 위치 기준으로 ± amp 흔들고 끝나면 그 위치로 복귀.
    """
    log.debug(f"sway: {bpm} BPM × {beats} beats")
    period = 60.0 / bpm
    total = period * beats
    dt = 1.0 / update_hz

    # 살짝 다양화 — 매 호출마다 위상 다르게
    phase_offset = random.uniform(0, 2 * math.pi)
    pan_amp = pan_amp_deg * random.uniform(0.7, 1.0)
    tilt_amp = tilt_amp_deg * random.uniform(0.5, 1.0)

    # base = 현재 서보 위치. CENTER 강제 X — 사용자 따라가던 위치 그대로 유지.
    base_pan = servos.position.pan
    base_tilt = servos.position.tilt

    start = time.monotonic()
    try:
        while True:
            t = time.monotonic() - start
            if t > total:
                break
            phase = (t / period) * 2 * math.pi + phase_offset
            env = min(1.0, t / period) * min(1.0, (total - t) / period)
            env = max(0.0, env)
            d_pan = math.sin(phase) * pan_amp * env
            d_tilt = math.sin(phase * 1.5 + math.pi / 2) * tilt_amp * env
            servos.set_angles(
                base_pan + d_pan,
                base_tilt + d_tilt,
            )
            await asyncio.sleep(dt)
    finally:
        # 시작 위치로 부드럽게 복귀 — envelope이 0으로 잘 떨어졌어도 안전상 한 번 더.
        await servos.smooth_to_async(base_pan, base_tilt, 0.3)
