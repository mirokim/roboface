"""발화 → 일정 추출 → SQLite 저장 → ThinkTank Calendar 동기화.

ambient_listener의 handler로 등록.
LLM 실패 / ThinkTank 다운 시 로컬 큐에 쌓고 나중에 재전송.
"""

from __future__ import annotations

import asyncio
import time

from src.brain import conversation, memory
from src.integrations.thinktank.client import CalendarEvent, ThinkTankClient
from src.utils.logger import get_logger

log = get_logger("schedule")


async def handle_transcript(text: str) -> None:
    """ambient_listener handler — 발화 한 줄당 호출."""
    events = conversation.extract_schedule(text)
    if not events:
        return

    log.info(f"일정 {len(events)}개 추출: {text[:50]}...")
    for ev in events:
        schedule_id = memory.add_schedule(
            event_type=ev.get("type", "reminder"),
            event_datetime=ev.get("datetime", ""),
            description=ev.get("description", text),
            confidence=ev.get("confidence", 0.7),
        )
        log.info(f"  → DB 저장 #{schedule_id}: {ev.get('description')}")


async def sync_pending_to_thinktank() -> None:
    """주기적으로 미동기화된 일정을 ThinkTank Calendar에 POST.

    백그라운드 task로 5분마다 실행.
    """
    while True:
        await asyncio.sleep(300)  # 5분
        pending = memory.unsynced_schedules()
        if not pending:
            continue
        log.info(f"미동기 일정 {len(pending)}개 ThinkTank로 동기화 시도")
        try:
            async with ThinkTankClient() as client:
                for sched in pending:
                    cal_event = CalendarEvent(
                        title=sched.description,
                        start=sched.event_datetime or _now_iso(),
                        description=f"[로봇 자동 추출, 신뢰도 {sched.confidence:.2f}]",
                        source="robot",
                    )
                    try:
                        await client.add_calendar_event(cal_event)
                        memory.mark_schedule_synced(sched.id)
                        log.info(f"  ✓ #{sched.id} 동기화 완료")
                    except Exception as e:
                        log.debug(f"  ✗ #{sched.id} 실패 (재시도 예정): {e}")
        except Exception as e:
            log.warning(f"ThinkTank 동기화 일괄 실패: {e}")


def _now_iso() -> str:
    from datetime import datetime
    return datetime.now().isoformat(timespec="minutes")
