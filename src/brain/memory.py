"""SQLite 기반 메모리 — 작업 세션, 일정, 사용자 패턴."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

from src.config import DB_PATH
from src.utils.logger import get_logger

log = get_logger("memory")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS work_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_ts REAL NOT NULL,
    end_ts REAL,
    duration_sec INTEGER,
    posture_warnings INTEGER DEFAULT 0,
    break_count INTEGER DEFAULT 0,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS proactive_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    trigger TEXT NOT NULL,
    message TEXT
);

CREATE TABLE IF NOT EXISTS schedules (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    extracted_at REAL NOT NULL,
    event_type TEXT,        -- meeting / deadline / reminder
    event_datetime TEXT,    -- ISO
    description TEXT,
    confidence REAL,
    synced_to_thinktank INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS env_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    temperature REAL,
    humidity REAL
);

CREATE TABLE IF NOT EXISTS user_patterns (
    key TEXT PRIMARY KEY,
    value TEXT,             -- JSON
    updated_at REAL
);
"""


_DB_INITIALIZED = False


def init_db(path: Path = DB_PATH) -> None:
    """스키마 생성 (idempotent). 프로세스당 한 번만 로그."""
    global _DB_INITIALIZED
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
    if not _DB_INITIALIZED:
        log.info(f"DB 초기화: {path}")
        _DB_INITIALIZED = True


@contextmanager
def db() -> Iterator[sqlite3.Connection]:
    if not _DB_INITIALIZED:
        init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# === Work Sessions ===

def start_work_session(now: float | None = None) -> int:
    now = now or time.time()
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO work_sessions (start_ts) VALUES (?)", (now,),
        )
        return cur.lastrowid or -1


def end_work_session(session_id: int, now: float | None = None) -> None:
    now = now or time.time()
    with db() as conn:
        row = conn.execute(
            "SELECT start_ts FROM work_sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if not row:
            return
        duration = int(now - row["start_ts"])
        conn.execute(
            "UPDATE work_sessions SET end_ts = ?, duration_sec = ? WHERE id = ?",
            (now, duration, session_id),
        )


def current_work_duration(session_id: int) -> float:
    """현재 진행 중인 세션의 경과 시간 (초)."""
    with db() as conn:
        row = conn.execute(
            "SELECT start_ts FROM work_sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if not row:
            return 0.0
        return time.time() - row["start_ts"]


def today_total_seconds() -> int:
    """오늘 누적 작업 시간."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
    with db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(duration_sec), 0) AS t FROM work_sessions "
            "WHERE start_ts >= ? AND end_ts IS NOT NULL",
            (today_start,),
        ).fetchone()
        return row["t"] or 0


# === Proactive Log ===

def log_proactive(trigger: str, message: str) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO proactive_log (ts, trigger, message) VALUES (?, ?, ?)",
            (time.time(), trigger, message),
        )


def proactive_count_last_hour() -> int:
    cutoff = time.time() - 3600
    with db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM proactive_log WHERE ts >= ?", (cutoff,),
        ).fetchone()
        return row["c"]


# === Schedules ===

@dataclass
class Schedule:
    id: int
    event_type: str
    event_datetime: str
    description: str
    confidence: float
    synced: bool


def add_schedule(
    event_type: str, event_datetime: str, description: str, confidence: float = 0.8,
) -> int:
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO schedules (extracted_at, event_type, event_datetime, description, confidence) "
            "VALUES (?, ?, ?, ?, ?)",
            (time.time(), event_type, event_datetime, description, confidence),
        )
        return cur.lastrowid or -1


def unsynced_schedules() -> list[Schedule]:
    with db() as conn:
        rows = conn.execute(
            "SELECT * FROM schedules WHERE synced_to_thinktank = 0 ORDER BY extracted_at",
        ).fetchall()
    return [
        Schedule(
            id=r["id"],
            event_type=r["event_type"],
            event_datetime=r["event_datetime"],
            description=r["description"],
            confidence=r["confidence"],
            synced=False,
        )
        for r in rows
    ]


def mark_schedule_synced(schedule_id: int) -> None:
    with db() as conn:
        conn.execute(
            "UPDATE schedules SET synced_to_thinktank = 1 WHERE id = ?",
            (schedule_id,),
        )


# === Env Log ===

def log_env(temperature: float, humidity: float) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO env_log (ts, temperature, humidity) VALUES (?, ?, ?)",
            (time.time(), temperature, humidity),
        )


# === User Patterns (key-value) ===

def set_pattern(key: str, value) -> None:
    with db() as conn:
        conn.execute(
            "INSERT INTO user_patterns (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )


def get_pattern(key: str, default=None):
    with db() as conn:
        row = conn.execute(
            "SELECT value FROM user_patterns WHERE key = ?", (key,),
        ).fetchone()
        if not row:
            return default
        return json.loads(row["value"])
