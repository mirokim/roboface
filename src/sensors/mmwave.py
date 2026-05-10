"""S3KM1110 24GHz mmWave 센서 — UART 기반 인체 감지.

simulator: MockMmWave (키보드/타이머로 이벤트 생성)
robot: S3KM1110Sensor (시리얼 패킷 파싱)

실제 패킷 포맷은 데이터시트에 의존. 일반적인 HLK-LD2410 계열 센서와 유사한
패킷 헤더(예: 0xF4 0xF3 0xF2 0xF1)로 시작하는 구조를 가정.
실제 펌웨어와 다를 수 있어, 도착 후 데이터시트 보고 _parse_packet 조정.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.config import MMWAVE_BAUDRATE, MMWAVE_UART, is_robot
from src.sensors.base import Sensor, SensorEvent, SensorEventType
from src.utils.logger import get_logger

log = get_logger("mmwave")


@dataclass
class MmWaveReading:
    presence: bool          # 사람 있음
    static_presence: bool   # 가만히 있는 사람 (호흡 등 미세동작 감지)
    distance_cm: float      # 거리 (없으면 -1)
    movement_energy: float  # 움직임 강도 (0~100)


class MockMmWaveSensor(Sensor):
    """시뮬레이터 — 랜덤하게 사용자 등장/이탈 이벤트 생성."""

    name = "mmwave_mock"

    def __init__(self) -> None:
        self._last_state = MmWaveReading(False, False, -1, 0)
        self._scenario_t0 = time.time()
        self._next_event_at = time.time() + random.uniform(3, 8)

    def poll(self) -> list[SensorEvent]:
        events: list[SensorEvent] = []
        now = time.time()

        if now >= self._next_event_at:
            # 토글
            if self._last_state.presence:
                self._last_state = MmWaveReading(False, False, -1, 0)
                events.append(SensorEvent(
                    SensorEventType.PRESENCE_LEFT,
                    data={"distance_cm": -1},
                ))
                self._next_event_at = now + random.uniform(15, 60)
            else:
                dist = random.uniform(50, 150)
                self._last_state = MmWaveReading(True, False, dist, 30)
                events.append(SensorEvent(
                    SensorEventType.PRESENCE_NEW,
                    data={"distance_cm": dist},
                ))
                self._next_event_at = now + random.uniform(120, 600)
        elif self._last_state.presence:
            # 이미 있음 — 가끔 거리 변동, 정적 존재 이벤트
            if random.random() < 0.02:
                self._last_state.distance_cm += random.uniform(-10, 10)
                events.append(SensorEvent(
                    SensorEventType.DISTANCE_CHANGED,
                    data={"distance_cm": self._last_state.distance_cm},
                ))
        return events

    def trigger_arrival(self, distance: float = 80.0) -> SensorEvent:
        """수동 트리거 (시뮬레이터에서 키 입력 등)."""
        self._last_state = MmWaveReading(True, False, distance, 30)
        self._next_event_at = time.time() + 60
        return SensorEvent(SensorEventType.PRESENCE_NEW, data={"distance_cm": distance})


class S3KM1110Sensor(Sensor):
    """실제 S3KM1110 센서 (UART)."""

    name = "mmwave_real"

    def __init__(self) -> None:
        import serial  # type: ignore[import-not-found]

        self.ser = serial.Serial(MMWAVE_UART, MMWAVE_BAUDRATE, timeout=0.1)
        self._last_presence = False
        self._buffer = b""
        log.info(f"S3KM1110 시리얼 열림: {MMWAVE_UART} @ {MMWAVE_BAUDRATE}bps")

    def poll(self) -> list[SensorEvent]:
        if not self.ser.in_waiting:
            return []
        chunk = self.ser.read(self.ser.in_waiting)
        self._buffer += chunk
        return self._parse_buffer()

    def _parse_buffer(self) -> list[SensorEvent]:
        """버퍼에서 패킷 추출 — 데이터시트 기반 추후 조정 필요.

        TODO: 실제 데이터시트 받으면 헤더/페이로드/체크섬 정확히 구현.
        지금은 단순 버퍼 클리어 + 임시 더미 동작.
        """
        events: list[SensorEvent] = []
        if len(self._buffer) > 1024:
            self._buffer = self._buffer[-256:]  # 오버플로우 방지
        # 임시: 데이터 받으면 presence True로 간주
        if self._buffer and not self._last_presence:
            events.append(SensorEvent(
                SensorEventType.PRESENCE_NEW,
                data={"distance_cm": -1, "raw": "stub"},
            ))
            self._last_presence = True
        return events

    def close(self) -> None:
        self.ser.close()


def create_sensor() -> Sensor:
    if is_robot():
        try:
            return S3KM1110Sensor()
        except Exception as e:
            log.warning(f"mmWave 초기화 실패, Mock 폴백: {e}")
            return MockMmWaveSensor()
    return MockMmWaveSensor()
