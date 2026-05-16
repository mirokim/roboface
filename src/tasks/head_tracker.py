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
SMOOTHING_ALPHA = 0.25         # 0.0(고정) ~ 1.0(즉시) — 부드러움
PAN_RANGE_DEG = 60             # bbox.x 좌→우 = ±이만큼 회전
TILT_RANGE_DEG = 25            # bbox.y 위→아래 = ±이만큼
RETURN_TO_CENTER_AFTER_SEC = 3  # 사람 부재 N초 후 중앙 복귀

# 카메라 마운트 방향에 따른 뒤집기 — 동작 확인하면서 조정 필요
PAN_INVERT = True   # 카메라가 사람을 화면 왼쪽에 볼 때 → 오른쪽으로 회전?
TILT_INVERT = True  # 카메라가 사람 위쪽 볼 때 → 머리 위로?


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

        # 지수 평활화 — 떨림 방지
        pan_current += (target_pan - pan_current) * SMOOTHING_ALPHA
        tilt_current += (target_tilt - tilt_current) * SMOOTHING_ALPHA

        try:
            servos.set_angles(pan_current, tilt_current)
        except Exception as e:
            log.warning(f"servo set_angles 실패: {e}")
            await asyncio.sleep(1.0)
