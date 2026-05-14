"""S3KM1110 / HLK-LD2410 계열 24GHz mmWave 인체 감지 센서 (UART).

simulator: MockMmWaveSensor (랜덤 이벤트 생성)
robot: S3KM1110Sensor (실제 UART 패킷 파싱)

프로토콜 (HLK-LD2410 / S3KM1110 호환):
  Engineering/Standard 모드 모두 동일한 헤더:
    헤더: F4 F3 F2 F1
    페이로드 길이: 2 bytes (LE)
    페이로드: ...
    꼬리: F8 F7 F6 F5

  Standard 모드 페이로드:
    [0]  = 데이터 타입 (0x02 표준)
    [1]  = 헤더 마커 (0xAA)
    [2]  = target_state (0=없음, 1=움직임, 2=정적, 3=움직임+정적)
    [3-4] = 움직이는 타겟 거리 (cm, LE)
    [5]   = 움직이는 타겟 에너지 (0-100)
    [6-7] = 정적 타겟 거리 (cm, LE)
    [8]   = 정적 타겟 에너지
    [9-10] = 탐지 거리 (cm, LE)
    [11]  = 0x55 (꼬리 마커)
    [12]  = 0x00

보드레이트 자동 감지:
  115200 (LD2410 기본) → 256000 (LD2410B) → 9600 (일부) 순서로 시도.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.config import MMWAVE_BAUDRATE, MMWAVE_UART, is_robot
from src.sensors.base import Sensor, SensorEvent, SensorEventType
from src.utils.logger import get_logger

log = get_logger("mmwave")


HEADER = b"\xf4\xf3\xf2\xf1"
FOOTER = b"\xf8\xf7\xf6\xf5"
CANDIDATE_BAUDRATES = [115200, 256000, 9600]


@dataclass
class MmWaveReading:
    presence: bool          # 사람 있음
    static_presence: bool   # 정적 (호흡 등 미세 동작)
    distance_cm: float      # 거리 (없으면 -1)
    movement_energy: float  # 움직임 강도 (0~100)


class MockMmWaveSensor(Sensor):
    """시뮬레이터 — 랜덤 등장/이탈."""

    name = "mmwave_mock"

    def __init__(self) -> None:
        self._presence = False
        self._next_event_at = time.time() + random.uniform(3, 8)

    def poll(self) -> list[SensorEvent]:
        events: list[SensorEvent] = []
        now = time.time()
        if now >= self._next_event_at:
            if self._presence:
                self._presence = False
                events.append(SensorEvent(
                    SensorEventType.PRESENCE_LEFT,
                    data={"source": "mmwave_mock", "distance_cm": -1},
                ))
                self._next_event_at = now + random.uniform(15, 60)
            else:
                dist = random.uniform(50, 150)
                self._presence = True
                events.append(SensorEvent(
                    SensorEventType.PRESENCE_NEW,
                    data={"source": "mmwave_mock", "distance_cm": dist},
                ))
                self._next_event_at = now + random.uniform(120, 600)
        return events

    def trigger_arrival(self, distance: float = 80.0) -> SensorEvent:
        self._presence = True
        self._next_event_at = time.time() + 60
        return SensorEvent(
            SensorEventType.PRESENCE_NEW,
            data={"source": "mmwave_mock", "distance_cm": distance},
        )


class S3KM1110Sensor(Sensor):
    """실제 S3KM1110 / HLK-LD2410 UART."""

    name = "mmwave_real"

    def __init__(self, port: str = MMWAVE_UART, baudrate: int = MMWAVE_BAUDRATE):
        import serial

        # 보드레이트 자동 감지 시도
        self.ser, used_baud = self._open_with_autodetect(serial, port, baudrate)
        self._buffer = b""
        self._last_presence = False
        self._last_distance_emit = 0.0
        log.info(f"S3KM1110 열림: {port} @ {used_baud}bps")

    def _open_with_autodetect(self, serial_mod, port: str, preferred: int):
        """preferred 우선, 데이터 안 들어오면 다른 보드레이트 시도."""
        baudrates = [preferred] + [b for b in CANDIDATE_BAUDRATES if b != preferred]
        last_err: Exception | None = None
        for baud in baudrates:
            try:
                ser = serial_mod.Serial(port, baud, timeout=0.1)
            except Exception as e:
                last_err = e
                continue
            # 0.5초간 데이터 + 헤더 검출
            deadline = time.time() + 0.5
            buf = b""
            while time.time() < deadline:
                chunk = ser.read(64)
                if chunk:
                    buf += chunk
                    if HEADER in buf:
                        return ser, baud
                time.sleep(0.02)
            # 헤더 없음 — 다음 보드레이트 시도
            ser.close()
        if last_err:
            raise last_err
        # 마지막 시도 — 그냥 preferred로 열어둠 (장비 미동작 등)
        return serial_mod.Serial(port, preferred, timeout=0.1), preferred

    def poll(self) -> list[SensorEvent]:
        if not self.ser.in_waiting:
            return []
        chunk = self.ser.read(self.ser.in_waiting)
        self._buffer += chunk
        if len(self._buffer) > 4096:
            self._buffer = self._buffer[-1024:]
        return self._parse_buffer()

    def _parse_buffer(self) -> list[SensorEvent]:
        """버퍼에서 완전한 패킷 추출 + 이벤트로 변환."""
        events: list[SensorEvent] = []
        while True:
            start = self._buffer.find(HEADER)
            if start < 0:
                # 헤더 없으면 버퍼 끝쪽만 남김 (다음 chunk에서 이어붙임)
                if len(self._buffer) > 64:
                    self._buffer = self._buffer[-32:]
                break
            # 헤더 뒤 길이 필드 (2 bytes LE)
            if len(self._buffer) < start + 6:
                self._buffer = self._buffer[start:]
                break
            length = int.from_bytes(self._buffer[start + 4:start + 6], "little")
            packet_end = start + 6 + length + 4  # +footer 4
            if len(self._buffer) < packet_end:
                self._buffer = self._buffer[start:]
                break
            payload = self._buffer[start + 6:start + 6 + length]
            footer = self._buffer[start + 6 + length:packet_end]
            self._buffer = self._buffer[packet_end:]
            if footer != FOOTER:
                # 동기화 깨짐 — 다음 헤더 찾기
                continue
            reading = _parse_payload(payload)
            if reading is None:
                continue
            events.extend(self._reading_to_events(reading))
        return events

    def _reading_to_events(self, r: MmWaveReading) -> list[SensorEvent]:
        out: list[SensorEvent] = []
        now = time.time()
        if r.presence and not self._last_presence:
            out.append(SensorEvent(
                SensorEventType.PRESENCE_NEW,
                data={
                    "source": "mmwave",
                    "distance_cm": r.distance_cm,
                    "static": r.static_presence,
                },
            ))
        elif not r.presence and self._last_presence:
            out.append(SensorEvent(
                SensorEventType.PRESENCE_LEFT,
                data={"source": "mmwave"},
            ))
        elif r.presence and r.static_presence and not self._last_presence_static:
            out.append(SensorEvent(
                SensorEventType.PRESENCE_STATIC,
                data={"source": "mmwave", "distance_cm": r.distance_cm},
            ))
        # 거리 변경 — 너무 자주 안 보내도록 throttle (3초)
        if r.presence and (now - self._last_distance_emit) > 3.0:
            out.append(SensorEvent(
                SensorEventType.DISTANCE_CHANGED,
                data={"source": "mmwave", "distance_cm": r.distance_cm},
            ))
            self._last_distance_emit = now

        self._last_presence = r.presence
        self._last_presence_static = r.static_presence
        return out

    _last_presence_static: bool = False

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


def _parse_payload(payload: bytes) -> MmWaveReading | None:
    """Standard 모드 페이로드 → MmWaveReading.

    페이로드 최소 13 bytes, 첫 두 byte는 [0x02, 0xAA].
    """
    if len(payload) < 13:
        return None
    if payload[0] != 0x02 or payload[1] != 0xAA:
        return None
    target_state = payload[2]
    moving_distance = int.from_bytes(payload[3:5], "little")
    moving_energy = payload[5]
    static_distance = int.from_bytes(payload[6:8], "little")
    static_energy = payload[8]
    detection_distance = int.from_bytes(payload[9:11], "little")

    presence = target_state != 0
    static = (target_state & 0x02) != 0  # 정적 비트
    distance = float(detection_distance) if detection_distance > 0 else (
        float(moving_distance) if moving_distance > 0 else float(static_distance)
    )
    energy = float(max(moving_energy, static_energy))

    return MmWaveReading(
        presence=presence,
        static_presence=static,
        distance_cm=distance,
        movement_energy=energy,
    )


def create_sensor() -> Sensor:
    if is_robot():
        try:
            return S3KM1110Sensor()
        except Exception as e:
            log.warning(f"mmWave 초기화 실패, Mock 폴백: {e}")
            return MockMmWaveSensor()
    return MockMmWaveSensor()
