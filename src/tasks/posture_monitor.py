"""자세 감시 — AI Camera + Pose Estimation 결과를 받아 분석.

부품 도착 전: mock posture data (랜덤 변동) 사용.
부품 도착 후: vision/pose_estimator.py가 실제 keypoint 제공.

판단 기준:
- 목 각도: 귀-어깨 직선이 수직에서 30° 이상 기울면 거북목
- 어깨 기울기: 좌우 차이 큰 경우 경고
- 지속 시간: N분 이상 안 좋은 자세 → 알림
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR
from src.face.expressions import WORRIED
from src.face.renderer import FaceState
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


class MockPostureProvider:
    """가짜 자세 데이터 — 시간 따라 천천히 변동."""

    def __init__(self) -> None:
        self._neck = 10.0
        self._shoulder = 0.0

    def read(self) -> PostureReading:
        self._neck += random.uniform(-2, 3)  # 살짝 안 좋아지는 경향
        self._neck = max(0, min(60, self._neck))
        self._shoulder += random.uniform(-2, 2)
        self._shoulder = max(-30, min(30, self._shoulder))
        return PostureReading(
            neck_angle_deg=self._neck,
            shoulder_tilt_deg=self._shoulder,
            timestamp=time.time(),
        )


class PostureMonitor:
    """안 좋은 자세 지속 시간 추적 + 단계별 알림."""

    def __init__(self, provider: MockPostureProvider | None = None) -> None:
        self.provider = provider or MockPostureProvider()
        self.bad_started_at: float | None = None
        self.warn_level: int = 0   # 0=없음, 1=soft, 2=강, 3=강력

    def _level_for(self, bad_duration_sec: float) -> int:
        if bad_duration_sec >= BEHAVIOR.posture_strong_continuous_sec:
            return 3
        if bad_duration_sec >= BEHAVIOR.posture_warn_continuous_sec:
            return 1
        return 0

    async def run(self, ctx: StateContext, face: FaceState) -> None:
        """매 30초마다 자세 측정."""
        while True:
            await asyncio.sleep(30)
            reading = self.provider.read()
            log.debug(f"posture: neck={reading.neck_angle_deg:.1f}° "
                      f"shoulder={reading.shoulder_tilt_deg:.1f}°")

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
                    log.info(f"자세 알림 level {new_level} (안 좋은 자세 "
                             f"{int(bad_for // 60)}분 지속)")
                    if ctx.state == State.WATCHING:
                        ctx.transition(State.ALERTING, face)
                        face.apply_expression(WORRIED)
            else:
                self.bad_started_at = None
                self.warn_level = 0
