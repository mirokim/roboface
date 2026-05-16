"""Eye tracker — 사용자 방향에 따라 눈동자가 조금씩 움직임.

머리(head_tracker)는 느리고 큰 움직임, 눈동자는 빠르고 작은 움직임.
실제 사람도 눈이 먼저 가고 머리가 뒤따라옴.

진폭은 ±0.3 (gaze 단위, 화면상 ~10~15px). saccade(±0.04)와 합산되어 그려짐.
"""

from __future__ import annotations

import asyncio

from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext  # noqa: F401  (향후 확장용)
from src.face.renderer import FaceState
from src.tasks.head_tracker import PAN_INVERT, TILT_INVERT
from src.utils.logger import get_logger

log = get_logger("eye_tracker")


UPDATE_HZ = 15           # 머리(10Hz)보다 살짝 빠르게 — 눈이 먼저 가는 느낌
BLEND = 0.4              # 보간 속도 (클수록 빠르게 따라감)
MAX_GAZE_X = 0.3         # 좌우 진폭 (±)
MAX_GAZE_Y = 0.18        # 상하 진폭 (눈동자는 위아래 적음)
RETURN_BLEND = 0.08      # 사람 없을 때 0으로 복귀 속도 (천천히)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def run_eye_tracker(
    face: FaceState,
    perception: PerceptionState,
    ctx: StateContext,
) -> None:
    """사람이 보이면 눈동자가 그쪽으로 슬쩍 움직임."""
    log.info("eye tracker 시작")
    period = 1.0 / UPDATE_HZ

    while True:
        await asyncio.sleep(period)

        if perception.person_present:
            cx, cy = perception.person_bbox_center
            # bbox 중심 (0~1) → 화면 중앙 기준 오프셋 (-0.5 ~ +0.5)
            ox = cx - 0.5
            oy = cy - 0.5
            # 카메라 inversion (head_tracker와 동일)
            if PAN_INVERT:
                ox = -ox
            if TILT_INVERT:
                oy = -oy
            # 오프셋을 gaze 진폭으로 매핑 (-0.5 ~ +0.5 → -MAX ~ +MAX)
            target_x = _clamp(ox * 2 * MAX_GAZE_X, -MAX_GAZE_X, MAX_GAZE_X)
            target_y = _clamp(oy * 2 * MAX_GAZE_Y, -MAX_GAZE_Y, MAX_GAZE_Y)
            blend = BLEND
        else:
            # 사람 없으면 천천히 0으로 복귀 (idle_gaze가 그 위에 가끔 큰 흔들기)
            target_x = 0.0
            target_y = 0.0
            blend = RETURN_BLEND

        face.eye_state.gaze_x += (target_x - face.eye_state.gaze_x) * blend
        face.eye_state.gaze_y += (target_y - face.eye_state.gaze_y) * blend
