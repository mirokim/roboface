"""발화 → 일정 추출 → SQLite 저장 → ThinkTank Calendar 동기화.

ambient_listener의 handler로 등록.
LLM 실패 / ThinkTank 다운 시 로컬 큐에 쌓고 나중에 재전송.
"""

from __future__ import annotations

import asyncio
import re
import time

from src.brain import conversation, memory
from src.integrations.thinktank.client import CalendarEvent, ThinkTankClient
from src.utils.logger import get_logger

log = get_logger("schedule")


# 시간/일정 단서 사전 필터 — 이게 없으면 Claude 추출 호출 자체를 skip.
# ambient STT가 노이즈로 발화를 자주 잡아(시간당 수백) 매 발화마다
# extract_schedule(Claude)를 부르던 게 큰 숨은 비용원이었음. 일상 발화 대부분은
# 시간 표현이 없으므로 사전 정규식으로 거른 뒤에만 Claude 호출.
_SCHEDULE_HINT_RE = re.compile(
    r"\d+\s*(시|분|일|월|시간|주)"          # 3시, 10분, 5일, 2주
    r"|내일|모레|글피|이따|나중에"
    r"|약속|회의|미팅|마감|데드라인|예약|일정|스케줄|행사|모임"
    r"|[월화수목금토일]요일"
    r"|다음\s*주|이번\s*주|담주|주말"
    r"|오전|오후"
)


async def handle_transcript(text: str) -> None:
    """ambient_listener handler — 발화 한 줄당 호출."""
    # 시간/일정 단서 없으면 Claude 추출 skip (비용 절감)
    if not _SCHEDULE_HINT_RE.search(text):
        return
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
