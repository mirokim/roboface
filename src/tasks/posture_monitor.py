"""자세 감시 — IMX500 pose keypoints 또는 mock data로 분석.

perception에 last_pose_keypoints가 있으면 그것으로 실제 자세 계산,
없으면 mock 데이터로 fallback.

판단 기준:
- 목 각도: 귀-어깨 벡터가 수직에서 25° 이상 기울면 거북목
- 어깨 기울기: 좌우 어깨 y 좌표 차이로 측정
- 지속 시간: N분 이상 안 좋은 자세 → 알림
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR
from src.face.expressions import WORRIED
from src.face.renderer import FaceState
from src.tasks import behavior_speaker
from src.utils.logger import get_logger

log = get_logger("posture_monitor")


@dataclass
class PostureReading:
    neck_angle_deg: float       # 수직에서 벗어난 각도. 0=정상, 30+=거북목
    shoulder_tilt_deg: float    # 어깨 좌우 기울기
    timestamp: float

    @property
    def is_bad(self) -> bool:
        return self.neck_angle_deg > 25 or abs(self.shoulder_tilt_deg) > 15

    @property
    def category(self) -> str:
        """카테고리화 — agent 컨텍스트용. upright/slouched/leaning."""
        if self.neck_angle_deg > 25:
            return "slouched"
        if abs(self.shoulder_tilt_deg) > 15:
            return "leaning"
        return "upright"


class MockPostureProvider:
    """가짜 자세 데이터 — 시간 따라 천천히 변동."""

    def __init__(self) -> None:
        self._neck = 10.0
        self._shoulder = 0.0

    def read(self) -> PostureReading:
        self._neck += random.uniform(-2, 3)
        self._neck = max(0, min(60, self._neck))
        self._shoulder += random.uniform(-2, 2)
        self._shoulder = max(-30, min(30, self._shoulder))
        return PostureReading(
            neck_angle_deg=self._neck,
            shoulder_tilt_deg=self._shoulder,
            timestamp=time.time(),
        )


def posture_from_keypoints(keypoints) -> PostureReading | None:
    """pose model keypoints에서 실제 자세 측정.

    keypoints: shape (17, 3) — (x, y, conf), 0~1 정규화.
    COCO 17: nose(0), l_eye(1), r_eye(2), l_ear(3), r_ear(4),
             l_shoulder(5), r_shoulder(6), ...

    None 반환 시 keypoint 신뢰도 부족.
    """
    if keypoints is None:
        return None
    try:
        l_ear = keypoints[3]
        r_ear = keypoints[4]
        l_sh = keypoints[5]
        r_sh = keypoints[6]
    except Exception:
        return None
    # 어깨 둘 다 보이고 신뢰도 충분해야 의미 있음
    if l_sh[2] < 0.2 or r_sh[2] < 0.2:
        return None

    # 어깨 기울기 — 좌우 y 차이를 어깨 너비로 정규화 후 각도로
    shoulder_dx = float(l_sh[0] - r_sh[0])   # 정규화 좌표
    shoulder_dy = float(l_sh[1] - r_sh[1])
    width = abs(shoulder_dx)
    if width < 0.02:
        return None
    # tilt 각도 — atan2(dy, dx). 양수면 왼쪽 어깨 아래 / 오른쪽 어깨 위
    import math
    shoulder_tilt_deg = math.degrees(math.atan2(shoulder_dy, shoulder_dx))

    # 거북목 — 귀와 어깨 중심을 잇는 선의 수직 기울기
    # 정상은 귀가 어깨 바로 위 (y_ear < y_shoulder, x 비슷). 거북목은 귀가 앞.
    ear_x = ear_y = ear_conf = 0.0
    valid_ears = 0
    if l_ear[2] >= 0.2:
        ear_x += float(l_ear[0]); ear_y += float(l_ear[1]); valid_ears += 1
    if r_ear[2] >= 0.2:
        ear_x += float(r_ear[0]); ear_y += float(r_ear[1]); valid_ears += 1
    if valid_ears == 0:
        # 귀 안 보이면 목 기울기 측정 불가 — 어깨 정보만
        return PostureReading(
            neck_angle_deg=0.0,
            shoulder_tilt_deg=shoulder_tilt_deg,
            timestamp=time.time(),
        )
    ear_x /= valid_ears
    ear_y /= valid_ears
    sh_mid_x = float((l_sh[0] + r_sh[0]) / 2)
    sh_mid_y = float((l_sh[1] + r_sh[1]) / 2)
    # 귀-어깨중심 벡터의 수직선(아래방향) 대비 각도
    dx = ear_x - sh_mid_x
    dy = sh_mid_y - ear_y   # 위로 가는 양수
    if dy <= 0.01:
        neck_angle_deg = 90.0   # 귀가 어깨보다 아래?? 매우 이상
    else:
        neck_angle_deg = abs(math.degrees(math.atan2(dx, dy)))

    return PostureReading(
        neck_angle_deg=neck_angle_deg,
        shoulder_tilt_deg=shoulder_tilt_deg,
        timestamp=time.time(),
    )


class PostureMonitor:
    """안 좋은 자세 지속 시간 추적 + 단계별 알림.

    keypoints_provider: callable returning current pose keypoints (17, 3) or None.
    None 반환 시 mock 데이터 fallback.
    """

    def __init__(
        self,
        provider: MockPostureProvider | None = None,
        keypoints_provider=None,
        perception: PerceptionState | None = None,
    ) -> None:
        self.provider = provider or MockPostureProvider()
        self.keypoints_provider = keypoints_provider
        self.perception = perception
        self.bad_started_at: float | None = None
        self.warn_level: int = 0

    def _level_for(self, bad_duration_sec: float) -> int:
        if bad_duration_sec >= BEHAVIOR.posture_strong_continuous_sec:
            return 3
        if bad_duration_sec >= BEHAVIOR.posture_warn_continuous_sec:
            return 1
        return 0

    async def run(self, ctx: StateContext, face: FaceState) -> None:
        """매 30초마다 자세 측정. 실제 keypoints 있을 때만 — mock은 비활성.

        가짜 자세 데이터로 false alert 생성하면 사용자 신뢰 깨짐.
        실제 신호 들어올 때만 의미 있는 판단.
        """
        while True:
            await asyncio.sleep(30)
            reading: PostureReading | None = None
            if self.keypoints_provider is not None:
                try:
                    kp = self.keypoints_provider()
                    reading = posture_from_keypoints(kp)
                except Exception as e:
                    log.debug(f"keypoint 자세 분석 실패: {e}")
            if reading is None:
                # 실제 데이터 없음 — mock fallback 안 함. 다음 tick 대기.
                continue
            log.debug(f"posture: neck={reading.neck_angle_deg:.1f}° "
                      f"shoulder={reading.shoulder_tilt_deg:.1f}°")

            # perception 공유 — agent가 자세 카테고리 인지하도록
            if self.perception is not None:
                self.perception.posture_category = reading.category
                self.perception.posture_category_at = time.time()

            if not ctx.user_present:
                self.bad_started_at = None
                self.warn_level = 0
                continue

            if reading.is_bad:
                if self.bad_started_at is None:
                    self.bad_started_at = time.time()
                bad_for = time.time() - self.bad_started_at
                new_level = self._level_for(bad_for)
                if new_level > self.warn_level:
                    self.warn_level = new_level
                    bad_min = int(bad_for // 60)
                    log.info(f"자세 알림 level {new_level} (안 좋은 자세 {bad_min}분)")
                    # 음성 알림 — Claude가 상황 보고 멘트 생성
                    if new_level >= 1:
                        strength = "심하게" if new_level >= 3 else ""
                        # 어떤 부분이 안 좋은지도 알림
                        problems = []
                        if reading.neck_angle_deg > 25:
                            problems.append(f"목 {strength} 굽음 ({reading.neck_angle_deg:.0f}°)")
                        if abs(reading.shoulder_tilt_deg) > 15:
                            problems.append(f"어깨 기울어짐 ({reading.shoulder_tilt_deg:+.0f}°)")
                        desc = (
                            f"사용자 자세가 {bad_min}분째 안 좋음 "
                            f"({', '.join(problems)}) — 살짝 알려주는 멘트"
                        )
                        from src.brain import memory
                        memory.log_user(
                            f"(자세 안 좋은 채로 {bad_min}분 지속)",
                            kind="bad_posture",
                        )
                        msg = ""
                        try:
                            from src.brain import conversation
                            msg = conversation.generate_situational(
                                desc,
                                user_name=getattr(ctx, "user_name", None),
                                extra={
                                    "bad_minutes": bad_min,
                                    "neck_angle": round(reading.neck_angle_deg, 1),
                                    "shoulder_tilt": round(reading.shoulder_tilt_deg, 1),
                                    "level": new_level,
                                },
                                max_tokens=80,
                            )
                        except Exception as e:
                            log.debug(f"자세 멘트 생성 실패: {e}")
                        if not msg:
                            msg = (
                                f"{bad_min}분째 자세가 안 좋아. 잠깐 펴봐."
                                if new_level < 3
                                else f"{bad_min}분 동안 같은 자세야. 진짜 한 번 일어나야 해."
                            )
                        behavior_speaker.say(
                            face, ctx, msg,
                            kind="posture_warn",
                            cooldown_sec=300.0,   # 같은 단계 5분 한 번
                            expression=WORRIED,
                        )
            else:
                self.bad_started_at = None
                self.warn_level = 0
