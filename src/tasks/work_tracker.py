"""작업 시간 자동 추적 — mmWave 이벤트 → SQLite 세션 + 휴식 알림 트리거.

사용자 등장: 세션 시작
사용자 이탈: 세션 종료
임계 시간 초과: 휴식 권유 트리거
"""

from __future__ import annotations

import asyncio
import time

from src.brain import memory, triggers
from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR
from src.sensors.base import SensorEvent, SensorEventType
from src.sensors.manager import SensorManager
from src.utils.logger import get_logger

log = get_logger("work_tracker")


class WorkTracker:
    """현재 진행 중인 작업 세션 관리."""

    def __init__(self) -> None:
        self.current_session_id: int | None = None
        self.last_warn_minutes: int = 0  # 같은 단계 중복 알림 방지

    def on_event(self, ev: SensorEvent, ctx: StateContext) -> None:
        if ev.type == SensorEventType.PRESENCE_NEW and self.current_session_id is None:
            self.current_session_id = memory.start_work_session()
            self.last_warn_minutes = 0
            log.info(f"작업 세션 시작 #{self.current_session_id}")
        elif ev.type == SensorEventType.PRESENCE_LEFT and self.current_session_id is not None:
            memory.end_work_session(self.current_session_id)
            log.info(f"작업 세션 종료 #{self.current_session_id}")
            self.current_session_id = None
            self.last_warn_minutes = 0

    def check_break_due(self, ctx: StateContext) -> bool:
        """휴식 권유 시점에 도달했는지. True면 트리거 발동 권장."""
        if self.current_session_id is None:
            return False
        duration_min = memory.current_work_duration(self.current_session_id) / 60
        for threshold in (
            BEHAVIOR.work_break_alarm_minutes,
            BEHAVIOR.work_break_strong_minutes,
            BEHAVIOR.work_break_warn_minutes,
        ):
            if duration_min >= threshold > self.last_warn_minutes:
                self.last_warn_minutes = threshold
                return True
        return False

    async def run(self, ctx: StateContext) -> None:
        """1분마다 휴식 권유 체크."""
        while True:
            await asyncio.sleep(60)
            if self.check_break_due(ctx):
                log.info(f"휴식 권유 트리거 (작업 {self.last_warn_minutes}분 도달)")
