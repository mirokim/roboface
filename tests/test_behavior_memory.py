"""behavior_memory — 하루 요약 + 기억 기반 멘트."""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta

import pytest

from src.brain import behavior_memory as bm, memory


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    monkeypatch.setattr(memory, "DB_PATH", db_path)
    monkeypatch.setattr(memory, "_DB_INITIALIZED", False)
    memory.init_db(db_path)
    yield


def _at(d: date, h: int, m: int = 0) -> float:
    return datetime(d.year, d.month, d.day, h, m).timestamp()


def _log(ts: float, speaker: str, kind: str, text: str = "x") -> None:
    with memory.db() as conn:
        conn.execute(
            "INSERT INTO conversation_log (ts, speaker, kind, text) VALUES (?, ?, ?, ?)",
            (ts, speaker, kind, text),
        )


def test_summarize_day_counts_events():
    today = date.today()
    _log(_at(today, 9, 5), "user", "presence_new")
    _log(_at(today, 9, 30), "user", "gesture_wave")
    _log(_at(today, 9, 31), "robot", "wave_reply")
    _log(_at(today, 10, 0), "user", "ambient")
    s = bm.summarize_day(today)
    assert s["arrival"] == _at(today, 9, 5)
    assert s["waves"] == 1 and s["speech"] == 1 and s["robot_said"] == 1
    assert s["last_seen"] == _at(today, 10, 0)


def test_remarks_compare_with_yesterday_and_streak():
    today = date.today()
    for i in range(1, 4):   # 3일 전~어제
        d = today - timedelta(days=i)
        _log(_at(d, 9, 0), "user", "presence_new")
        memory.set_pattern(f"behavior_day_{d.isoformat()}", bm.summarize_day(d))
    _log(_at(today, 10, 30), "user", "presence_new")   # 오늘 1.5h 늦게
    now = _at(today, 10, 40)
    r = bm.remarks(now=now)
    assert any("늦었" in x for x in r)
    assert any("4일 연속" in x for x in r)


def test_remarks_empty_without_data():
    assert bm.remarks() == []


def test_history_text_runs():
    assert isinstance(bm.history_text(3), str)
