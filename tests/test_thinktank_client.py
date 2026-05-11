"""ThinkTank 클라이언트 단위 테스트 — 실제 서버 호출 안 함, 구조만 확인."""

import pytest

from src.integrations.thinktank.client import (
    CalendarEvent, JournalEntry, ThinkTankClient,
)


def test_journal_entry_required_fields():
    e = JournalEntry(content="test")
    assert e.content == "test"
    assert e.mode == "auto"


def test_calendar_event_with_optional():
    e = CalendarEvent(title="회의", start="2026-05-15T14:00")
    assert e.title == "회의"
    assert e.end is None
    assert e.source == "robot"


@pytest.mark.asyncio
async def test_client_context_manager_close():
    """Context manager 진입/탈출 시 에러 없음."""
    async with ThinkTankClient(base_url="http://invalid.local") as c:
        assert c._client is not None
    # 탈출 후엔 닫혀야 함
