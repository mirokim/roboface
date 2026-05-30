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
import random
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


# 추적 파라미터 — 적극적이지만 자잘한 움직임 X.
# MAX_STEP_DEG × UPDATE_HZ = 초당 최대 회전 속도. 사람 자연 ~60°/sec.
UPDATE_HZ = 15
SMOOTHING_ALPHA = 0.10   # 0.15 → 0.10 (더 천천히 — bbox 노이즈 따라 휙 방지)
PAN_RANGE_DEG = 70
TILT_RANGE_DEG = 30
RETURN_TO_CENTER_AFTER_SEC = 3
MAX_STEP_DEG = 2.5   # 4 → 2.5 (37°/sec — 사람보다 약간 느리게)

# 데드존 — 자잘한 떨림 방지를 위한 3단계:
# 1) bbox 센터 자체 N프레임 median (입력 노이즈 제거 — 평균 대신 outlier 강함)
# 2) 타깃 각도 변화가 작으면 stable target 유지 (수렴 안정성)
# 3) 출력 각도 변화가 0.5도 이내면 서보 명령 자체 skip (PWM 떨림 방지)
BBOX_SMOOTH_N = 8
TARGET_DEADZONE_DEG = 6.0   # 3 → 6 (작은 노이즈 무시)
# bbox가 한 프레임에 이 비율 이상 점프하면 noise로 보고 reject — 두 사람 감지
# 시 다른 person 갑자기 잡혀서 머리 휙 돌아가는 거 차단.
BBOX_JUMP_REJECT = 0.25
# 출력 데드존 — breath OFF 됐으니 크게. 작은 PWM 떨림/노이즈 묻힘.
OUTPUT_DEADZONE_DEG = 0.5
# 서보 settle 끔 — 카메라가 머리에 함께 달린 구조에서 settle 동안 perception을
# 무시하면, 카메라 회전 후 bbox 위치가 크게 점프 → 다음 사이클에 큰 폭 명령 →
# 양성 피드백 발산. 정상 데드존 + 스무딩이 이미 떨림 흡수.
SERVO_SETTLE_SEC = 0.0

PAN_INVERT = True
TILT_INVERT = True

# === Breathing — 자연 호흡은 위아래만. PAN은 0으로 (갸우뚱 거슬림). ===
# OFF (amp=0) — 머리+카메라가 한 덩어리라 tilt 진동이 bbox 흔들림 → 추적 떨림으로 증폭됐음.
# 1시간 간격 게이팅 로직은 남겨두되 진폭이 0이라 시각적 영향 없음.
BREATH_TILT_AMP_MIN_DEG = 0.0
BREATH_TILT_AMP_MAX_DEG = 0.0
BREATH_TILT_PERIOD_SEC = 60.0
BREATH_PAN_AMP_DEG = 0.0    # 좌우 호흡 거슬리니까 끔
BREATH_PAN_PERIOD_SEC = 60.0
BREATH_INTERVAL_SEC = 3600.0  # 1시간에 한 번 사이클 시작


# 호흡 사이클 시작 시각 + 그 사이클의 진폭.
_breath_state: dict[str, float | None] = {
    "cycle_start": None,
    "current_amp": BREATH_TILT_AMP_MAX_DEG,
    "next_cycle_at": 0.0,
}


def _breathing_offsets(t: float) -> tuple[float, float]:
    """현재 시각의 호흡 (pan_offset, tilt_offset).

    1시간에 한 번만 60초짜리 sin 한 주기를 돌고, 그 외 시간은 (0, 0).
    """
    cycle_start = _breath_state["cycle_start"]

    if cycle_start is None:
        # 아직 사이클 중이 아님 — 시작 시각인지 확인.
        if t >= (_breath_state["next_cycle_at"] or 0.0):
            _breath_state["cycle_start"] = t
            _breath_state["current_amp"] = random.uniform(
                BREATH_TILT_AMP_MIN_DEG, BREATH_TILT_AMP_MAX_DEG,
            )
            _breath_state["next_cycle_at"] = t + BREATH_INTERVAL_SEC
            cycle_start = t
        else:
            return 0.0, 0.0

    elapsed = t - cycle_start
    if elapsed >= BREATH_TILT_PERIOD_SEC:
        # 사이클 끝 — 다음 1시간까지 휴면.
        _breath_state["cycle_start"] = None
        return 0.0, 0.0

    pan = math.sin(elapsed * 2 * math.pi / BREATH_PAN_PERIOD_SEC) * BREATH_PAN_AMP_DEG
    tilt = (
        math.sin(elapsed * 2 * math.pi / BREATH_TILT_PERIOD_SEC)
        * _breath_state["current_amp"]
    )
    return pan, tilt


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


async def run_head_tracker(
    servos: ServoController,
    perception: PerceptionState,
    ctx: StateContext,
) -> None:
    from collections import deque
    period = 1.0 / UPDATE_HZ
    pan_current = float(PAN_CENTER_DEG)
    tilt_current = float(TILT_CENTER_DEG)
    # bbox 센터 스무딩 — 최근 N프레임 평균
    bbox_history: deque[tuple[float, float]] = deque(maxlen=BBOX_SMOOTH_N)
    # 마지막으로 처리한 stable target (데드존 비교용)
    stable_target_pan = float(PAN_CENTER_DEG)
    stable_target_tilt = float(TILT_CENTER_DEG)
    # 마지막으로 서보에 보낸 각도 (출력 데드존용)
    last_sent_pan = pan_current
    last_sent_tilt = tilt_current
    # 마지막 서보 명령 시각 — settling 동안 perception 입력 무시
    last_servo_move_at = 0.0

    log.info("head tracker 시작")

    while True:
        await asyncio.sleep(period)

        if ctx.state in (State.TALKING, State.GREETING, State.LISTENING):
            continue
        # 다른 모션 task가 서보 점유 중이면 양보 (sway/nod/shake/dance/...)
        if ctx.motion_busy:
            continue

        # 서보 움직임 직후엔 카메라가 같이 흔들려서 bbox가 거짓 신호.
        # settling 동안엔 새 입력 무시하고 stable target 유지.
        in_settle = (time.monotonic() - last_servo_move_at) < SERVO_SETTLE_SEC

        # 타깃 각도 계산
        if perception.person_present:
            if not in_settle:
                cx, cy = perception.person_bbox_center
                # bbox jump 가드 — 직전 프레임 대비 큰 점프(예: 다른 사람)면 reject
                if bbox_history:
                    last_cx, last_cy = bbox_history[-1]
                    if (abs(cx - last_cx) > BBOX_JUMP_REJECT
                            or abs(cy - last_cy) > BBOX_JUMP_REJECT):
                        # 노이즈 — 이번 프레임 무시. stable target 그대로 유지.
                        target_pan = stable_target_pan
                        target_tilt = stable_target_tilt
                        # fall through to clamp+smoothing 단계 (아래)
                        target_pan = _clamp(target_pan, PAN_MIN_DEG, PAN_MAX_DEG)
                        target_tilt = _clamp(target_tilt, TILT_MIN_DEG, TILT_MAX_DEG)
                        pan_delta = (target_pan - pan_current) * SMOOTHING_ALPHA
                        tilt_delta = (target_tilt - tilt_current) * SMOOTHING_ALPHA
                        pan_delta = _clamp(pan_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
                        tilt_delta = _clamp(tilt_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
                        pan_current += pan_delta
                        tilt_current += tilt_delta
                        continue
                # bbox 센터 스무딩 — median (outlier 강함, 평균보다 noise 견고)
                bbox_history.append((cx, cy))
                sorted_x = sorted(p[0] for p in bbox_history)
                sorted_y = sorted(p[1] for p in bbox_history)
                avg_cx = sorted_x[len(sorted_x) // 2]
                avg_cy = sorted_y[len(sorted_y) // 2]

                # 새 타깃 계산 후 stable과 비교 (타깃 데드존)
                ox = avg_cx - 0.5
                oy = avg_cy - 0.5
                if PAN_INVERT:
                    ox = -ox
                if TILT_INVERT:
                    oy = -oy
                new_target_pan = PAN_CENTER_DEG + ox * PAN_RANGE_DEG * 2
                new_target_tilt = TILT_CENTER_DEG + oy * TILT_RANGE_DEG * 2

                # 타깃 데드존: 작은 변화는 무시 (수렴 안정성)
                if (abs(new_target_pan - stable_target_pan) > TARGET_DEADZONE_DEG
                        or abs(new_target_tilt - stable_target_tilt) > TARGET_DEADZONE_DEG):
                    stable_target_pan = new_target_pan
                    stable_target_tilt = new_target_tilt

            target_pan = stable_target_pan
            target_tilt = stable_target_tilt
        else:
            bbox_history.clear()
            target_pan = PAN_CENTER_DEG
            target_tilt = TILT_CENTER_DEG
            stable_target_pan = PAN_CENTER_DEG
            stable_target_tilt = TILT_CENTER_DEG

        target_pan = _clamp(target_pan, PAN_MIN_DEG, PAN_MAX_DEG)
        target_tilt = _clamp(target_tilt, TILT_MIN_DEG, TILT_MAX_DEG)

        pan_delta = (target_pan - pan_current) * SMOOTHING_ALPHA
        tilt_delta = (target_tilt - tilt_current) * SMOOTHING_ALPHA
        pan_delta = _clamp(pan_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        tilt_delta = _clamp(tilt_delta, -MAX_STEP_DEG, MAX_STEP_DEG)
        pan_current += pan_delta
        tilt_current += tilt_delta

        # 호흡 오프셋 — 작게 (자잘한 떨림 안 보일 정도)
        breath_pan, breath_tilt = _breathing_offsets(time.monotonic())
        out_pan = _clamp(pan_current + breath_pan, PAN_MIN_DEG, PAN_MAX_DEG)
        out_tilt = _clamp(tilt_current + breath_tilt, TILT_MIN_DEG, TILT_MAX_DEG)

        # 출력 데드존 — 직전 명령과 OUTPUT_DEADZONE_DEG 이내면 명령 skip
        if (abs(out_pan - last_sent_pan) < OUTPUT_DEADZONE_DEG
                and abs(out_tilt - last_sent_tilt) < OUTPUT_DEADZONE_DEG):
            continue
        last_sent_pan = out_pan
        last_sent_tilt = out_tilt
        last_servo_move_at = time.monotonic()

        try:
            servos.set_angles(out_pan, out_tilt)
            # perception 공유 — agent가 자기 머리 방향 인지
            perception.head_pan_deg = out_pan
            perception.head_tilt_deg = out_tilt
            perception.head_angle_at = time.time()
        except Exception as e:
            log.warning(f"servo set_angles 실패: {e}")
            await asyncio.sleep(1.0)
