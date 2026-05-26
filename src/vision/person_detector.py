"""사람 감지기 — Detection 스트림에서 안정적인 presence 신호 추출.

원시 detections는 프레임마다 깜빡일 수 있음 (조명/각도 등).
디바운스 + 히스테리시스로 안정적 상태 전이 신호 생성:

  AWAY ──(person N프레임 연속)──→ PRESENT
  PRESENT ──(no person M초)──→ AWAY

이벤트로 변환:
  AWAY → PRESENT: PRESENCE_NEW
  PRESENT → AWAY: PRESENCE_LEFT
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from src.sensors.base import SensorEvent, SensorEventType
from src.utils.logger import get_logger
from src.vision.camera import Detection

log = get_logger("person_detector")


class PresenceState(Enum):
    AWAY = "away"
    PRESENT = "present"


@dataclass
class PersonDetector:
    """프레임 시퀀스 → 안정적 presence 상태."""

    # 튜닝 파라미터
    person_class: str = "person"
    confirm_frames: int = 3       # AWAY → PRESENT 전이에 필요한 연속 감지 프레임
    # 10→30 — 컵 들어 가리거나 의자 잠깐 빠지는 정도(10~20s)는 LEFT로 보지 말 것.
    # 진짜 자리 비움(다른 방 이동 등)만 잡힘.
    away_timeout_sec: float = 30.0
    min_confidence: float = 0.5   # 사람 인정 최소 신뢰도 (pose 모드는 0.3 권장)

    # 내부 상태
    state: PresenceState = PresenceState.AWAY
    _consecutive: int = 0
    _last_seen_at: float | None = None
    _last_distance: float | None = None

    def process(self, detections: list[Detection]) -> list[SensorEvent]:
        """detection 한 프레임을 처리. 상태 전이 발생 시 이벤트 리스트 반환."""
        now = time.time()
        events: list[SensorEvent] = []

        person_seen = any(
            d.class_name == self.person_class and d.confidence >= self.min_confidence
            for d in detections
        )

        if person_seen:
            self._consecutive += 1
            self._last_seen_at = now
            # 가장 큰 bbox = 가까운 사람 가정
            biggest = max(
                (d for d in detections if d.class_name == self.person_class),
                key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                default=None,
            )
            if biggest:
                self._last_distance = _bbox_to_distance(biggest.bbox)
        else:
            self._consecutive = 0

        # AWAY → PRESENT
        if self.state == PresenceState.AWAY and self._consecutive >= self.confirm_frames:
            self.state = PresenceState.PRESENT
            log.info(f"사람 감지 (카메라) — 거리 ~{self._last_distance:.0f}cm")
            events.append(SensorEvent(
                SensorEventType.PRESENCE_NEW,
                timestamp=now,
                data={
                    "source": "camera",
                    "distance_cm": self._last_distance or -1,
                },
            ))

        # PRESENT → AWAY
        elif (
            self.state == PresenceState.PRESENT
            and self._last_seen_at is not None
            and (now - self._last_seen_at) > self.away_timeout_sec
        ):
            self.state = PresenceState.AWAY
            log.info("사람 사라짐 (카메라)")
            events.append(SensorEvent(
                SensorEventType.PRESENCE_LEFT,
                timestamp=now,
                data={"source": "camera"},
            ))

        return events


def _bbox_to_distance(bbox: tuple[float, float, float, float]) -> float:
    """Bounding box 면적으로 거리 대략 추정.

    bbox는 정규화 0~1. 면적 클수록 가까움.
    실제 거리는 사람 키/카메라 FoV 등에 따라 다르지만 대략적 지표.
    """
    x0, y0, x1, y1 = bbox
    area = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    if area <= 0.0:
        return -1.0
    # 거친 캘리브레이션: 면적 0.3 (=화면 30%) ≈ 1m, 0.05 ≈ 3m
    # 거리(cm) ≈ 100 / sqrt(area)
    import math
    return float(min(500.0, max(20.0, 100.0 / math.sqrt(area))))
