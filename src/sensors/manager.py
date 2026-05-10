"""센서 통합 관리자 — 모든 센서 폴링하고 이벤트 큐에 모음."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field

from src.sensors import environment, mmwave
from src.sensors.base import Sensor, SensorEvent
from src.utils.logger import get_logger

log = get_logger("sensor_manager")


@dataclass
class SensorManager:
    sensors: list[Sensor] = field(default_factory=list)
    events: deque[SensorEvent] = field(default_factory=lambda: deque(maxlen=200))
    poll_interval_sec: float = 0.1

    def register_default(self) -> None:
        """기본 센서 (mmWave + DHT22) 등록."""
        self.sensors.append(mmwave.create_sensor())
        self.sensors.append(environment.create_sensor())
        log.info(f"센서 {len(self.sensors)}개 등록: " +
                 ", ".join(s.name for s in self.sensors))

    async def run(self) -> None:
        """비동기 폴링 루프."""
        log.info("센서 매니저 시작")
        while True:
            for s in self.sensors:
                try:
                    new_events = s.poll()
                    for e in new_events:
                        self.events.append(e)
                        log.debug(f"[{s.name}] {e.type.value} {e.data}")
                except Exception as ex:
                    log.warning(f"센서 {s.name} 폴링 에러: {ex}")
            await asyncio.sleep(self.poll_interval_sec)

    def drain_events(self) -> list[SensorEvent]:
        """누적된 이벤트 가져가고 비움."""
        out = list(self.events)
        self.events.clear()
        return out

    def close(self) -> None:
        for s in self.sensors:
            try:
                s.close()
            except Exception as e:
                log.warning(f"close 에러: {e}")
