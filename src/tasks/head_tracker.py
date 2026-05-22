"""Head Tracker — perception 상태에 따라 서보로 머리 회전.

WATCHING / IDLE 상태에서만 활성.
TALKING / ALERTING 등엔 서보 제어권을 다른 task에 양보.

알고리즘:
- 사람 bbox 중심 (0~1 정규화) → 서보 각도로 매핑
- 지수 평활화로 부드럽게 추적 (jitter 제거)
- 사람 없으면 중앙으로 천천히 복귀
"""

from __future__ import annotations

import asyncio
import math
import time

from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.config import (
    PAN_CENTER_DEG, PAN_MAX_DEG, PAN_MIN_DEG,
    TILT_CENTER_DEG, TILT_MAX_DEG, TILT_MIN_DEG,
)
from src.motion.servos import ServoController
from src.utils.logger import get_logger

log = get_logger("head_tracker")


# 추적 파라미터
UPDATE_HZ = 10                 # 초당 갱신 횟수
SMOOTHING_ALPHA = 0.08         # 0.0(고정) ~ 1.0(즉시). 낮을수록 천천히. 0.25 → 0.08
PAN_RANGE_DEG = 50             # bbox.x 좌→우 = ±이만큼 회전 (60 → 50)
TILT_RANGE_DEG = 18            # bbox.y 위→아래 (25 → 18)
RETURN_TO_CENTER_AFTER_SEC = 3
# 한 프레임당 최대 회전량 (도). smoothing이 빠른 동작 만들어도 이 이상은 안 돌게.
MAX_STEP_DEG = 4.0

# 카메라 마운트 방향에 따른 뒤집기 — 동작 확인하면서 조정 필요
PAN_INVERT = True   # 카메라가 사람을 화면 왼쪽에 볼 때 → 오른쪽으로 회전?
TILT_INVERT = True  # 카메라가 사람 위쪽 볼 때 → 머리 위로?

# === Breathing — 정지하지 않게 항상 미세 sine wave 추가 ===
# 4초 주기로 위아래 ±1.5°, 5.7초 주기로 좌우 ±0.5° (소수 점 어긋난 주기 = 자연스러움)
BREATH_TILT_AMP_DEG = 1.5
BREATH_TILT_PERIOD_SEC = 4.0
BREATH_PAN_AMP_DEG = 0.5
BREATH_PAN_PERIOD_SEC = 5.7


def _breathing_offsets(t: float) -> tuple[float, float]:
    """현재 시각의 호흡 (pan_offset, tilt_offset)."""
    pan = math.sin(t * 2 * math.pi / BREATH_PAN_PERIOD_SEC) * BREATH_PAN_AMP_DEG
    tilt = math.sin(t * 2 * math.pi / BREATH_TILT_PERIOD_SEC) * BREATH_TILT_AMP_DEG
    return pan, tilt


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def run_head_tracker(
    servos: ServoController,
    perception: PerceptionState,
    ctx: StateContext,
) -> None:
    period = 1.0 / UPDATE_HZ
    pan_current = float(PAN_CENTER_DEG)
    tilt_current = float(TILT_CENTER_DEG)

    log.info("head tracker 시작")

    while True:
        await asyncio.sleep(period)

        # 다른 task가 서보를 점유 중이면 양보
        if ctx.state in (State.TALKING, State.GREETING, State.LISTENING):
            continue
        if ctx.ambient_motion_active:
            continue

        # 타겟 각도 계산
        if perception.person_present:
            cx, cy = perception.person_bbox_center
            # bbox 중심 (0~1) → 화면 중앙 기준 오프셋 (-0.5 ~ +0.5)
            ox = cx - 0.5
            oy = cy - 0.5
            if PAN_INVERT:
                ox = -ox
            if TILT_INVERT:
                oy = -oy
            target_pan = PAN_CENTER_DEG + ox * PAN_RANGE_DEG * 2
            target_tilt = TILT_CENTER_DEG + oy * TILT_RANGE_DEG * 2
        else:
            # 사람 없으면 중앙 복귀
            target_pan = PAN_CENTER_DEG
            target_tilt = TILT_CENTER_DEG

        # 가동 범위 제한
        target_pan = _clamp(target_pan, PAN_MIN_DEG, PAN_MAX_DEG)
        target_tilt = _clamp(target_tilt, TILT_MIN_DEG, TILT_MAX_DEG)

        # 지수 평활화 + 프레임당 최대 회전량 제한 — 천천히, 격하지 않게
        pan_delta = (target_pan - pan_current) * SMOOTHING_ALPHA
        tilt_delta = (target_tilt - tilt_current) * SMOOTHING_ALPHA
        pan_delta = _clamp(pan_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        tilt_delta = _clamp(tilt_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        pan_current += pan_delta
        tilt_current += tilt_delta

        # 호흡 오프셋 — 항상 살아있는 미세 진동
        breath_pan, breath_tilt = _breathing_offsets(time.monotonic())
        out_pan = _clamp(pan_current + breath_pan, PAN_MIN_DEG, PAN_MAX_DEG)
        out_tilt = _clamp(tilt_current + breath_tilt, TILT_MIN_DEG, TILT_MAX_DEG)

        try:
            servos.set_angles(out_pan, out_tilt)
        except Exception as e:
            log.warning(f"servo set_angles 실패: {e}")
            await asyncio.sleep(1.0)
