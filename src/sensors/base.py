"""센서 추상 베이스 + 이벤트 타입."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SensorEventType(Enum):
    PRESENCE_NEW = "presence_new"          # 사용자 신규 등장
    PRESENCE_LEFT = "presence_left"        # 사용자 자리 비움
    PRESENCE_STATIC = "presence_static"    # 정적 존재 (가만히 있음)
    DISTANCE_CHANGED = "distance_changed"  # 거리 변화
    ENV_TEMP = "env_temp"                  # 온도 측정
    ENV_HUMIDITY = "env_humidity"          # 습도 측정
    ENV_ALERT = "env_alert"                # 환경 급변
    GESTURE_WAVE = "gesture_wave"          # 손 흔들기 감지
    GESTURE_HANDS_UP = "gesture_hands_up"  # 양손 만세
    GESTURE_HEAD_NOD = "gesture_head_nod"  # 고개 끄덕임 (yes)
    GESTURE_HEAD_SHAKE = "gesture_head_shake"  # 고개 도리도리 (no)
    GAZE_AT_ME = "gaze_at_me"              # 사용자가 로봇 정면 응시


@dataclass
class SensorEvent:
    type: SensorEventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


class Sensor(ABC):
    """센서 추상 클래스."""

    name: str = "sensor"

    @abstractmethod
    def poll(self) -> list[SensorEvent]:
        """센서 읽고 이벤트 반환. 빈 리스트 가능."""

    def close(self) -> None:
        """리소스 정리."""
