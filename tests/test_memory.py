"""SQLite 메모리 모듈 단위 테스트."""

import time

import pytest

from src.brain import memory


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """각 테스트마다 임시 DB."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_DB_INITIALIZED", False)
    memory.init_db(db_path)
    yield


def test_work_session_lifecycle():
    sid = memory.start_work_session()
    assert sid > 0
    time.sleep(0.05)
    memory.end_work_session(sid)


def test_today_total_seconds_starts_zero():
    assert memory.today_total_seconds() == 0


def test_proactive_log_and_count():
    assert memory.proactive_count_last_hour() == 0
    memory.log_proactive("greeting", "안녕!")
    assert memory.proactive_count_last_hour() == 1


def test_schedule_add_and_query():
    sid = memory.add_schedule("meeting", "2026-05-15T14:00", "Test meeting", 0.9)
    assert sid > 0
    pending = memory.unsynced_schedules()
    assert any(s.id == sid for s in pending)
    memory.mark_schedule_synced(sid)
    pending2 = memory.unsynced_schedules()
    assert not any(s.id == sid for s in pending2)


def test_user_pattern_kv():
    memory.set_pattern("test_key", {"value": 42})
    assert memory.get_pattern("test_key") == {"value": 42}
    assert memory.get_pattern("missing", default="x") == "x"
