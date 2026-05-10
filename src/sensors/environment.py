"""DHT22 온습도 센서.

simulator: MockDHT22 (실내 평균값 + 약간의 변동)
robot: DHT22Sensor (adafruit-circuitpython-dht)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass

from src.config import DHT22_GPIO, is_robot
from src.sensors.base import Sensor, SensorEvent, SensorEventType
from src.utils.logger import get_logger

log = get_logger("dht22")

POLL_INTERVAL_SEC = 10.0  # DHT22는 초당 1회 이하 권장


@dataclass
class EnvReading:
    temperature_c: float
    humidity_pct: float


class MockDHT22Sensor(Sensor):
    """시뮬레이터 — 랜덤 워크."""

    name = "dht22_mock"

    def __init__(self) -> None:
        self._temp = 23.5
        self._humid = 45.0
        self._last_poll = 0.0

    def poll(self) -> list[SensorEvent]:
        now = time.time()
        if now - self._last_poll < POLL_INTERVAL_SEC:
            return []
        self._last_poll = now

        self._temp += random.uniform(-0.3, 0.3)
        self._humid += random.uniform(-1.0, 1.0)
        self._temp = max(15, min(35, self._temp))
        self._humid = max(20, min(85, self._humid))

        return [
            SensorEvent(SensorEventType.ENV_TEMP, data={"value": round(self._temp, 1)}),
            SensorEvent(SensorEventType.ENV_HUMIDITY, data={"value": round(self._humid, 1)}),
        ]


class DHT22Sensor(Sensor):
    """실제 DHT22."""

    name = "dht22_real"

    def __init__(self) -> None:
        import adafruit_dht  # type: ignore[import-not-found]
        import board  # type: ignore[import-not-found]

        pin = getattr(board, f"D{DHT22_GPIO}")
        self.device = adafruit_dht.DHT22(pin, use_pulseio=False)
        self._last_poll = 0.0
        self._last_temp = 23.0
        self._last_humid = 45.0
        log.info(f"DHT22 초기화 완료 (GPIO {DHT22_GPIO})")

    def poll(self) -> list[SensorEvent]:
        now = time.time()
        if now - self._last_poll < POLL_INTERVAL_SEC:
            return []
        self._last_poll = now
        try:
            t = self.device.temperature
            h = self.device.humidity
            if t is None or h is None:
                return []
            self._last_temp = t
            self._last_humid = h
            return [
                SensorEvent(SensorEventType.ENV_TEMP, data={"value": round(t, 1)}),
                SensorEvent(SensorEventType.ENV_HUMIDITY, data={"value": round(h, 1)}),
            ]
        except RuntimeError as e:
            # DHT22는 가끔 타이밍 실패 — 무시하고 다음 폴링
            log.debug(f"DHT22 read fail (재시도): {e}")
            return []

    def close(self) -> None:
        try:
            self.device.exit()
        except Exception:
            pass


def create_sensor() -> Sensor:
    if is_robot():
        try:
            return DHT22Sensor()
        except Exception as e:
            log.warning(f"DHT22 초기화 실패, Mock 폴백: {e}")
            return MockDHT22Sensor()
    return MockDHT22Sensor()
