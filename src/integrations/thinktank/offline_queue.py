"""오프라인 큐 — ThinkTank 다운 시 SQLite에 쌓아두고 복구 시 재전송.

큐 자체는 SQLite의 별도 테이블. 페이로드는 JSON 직렬화.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from src.brain.memory import DB_PATH, init_db
from src.integrations.thinktank.client import (
    CalendarEvent, JournalEntry, ThinkTankClient,
)
from src.utils.logger import get_logger

log = get_logger("offline_queue")


_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS thinktank_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    enqueued_at REAL NOT NULL,
    op_type TEXT NOT NULL,           -- "journal" / "calendar" / "observation"
    payload TEXT NOT NULL,           -- JSON
    attempts INTEGER DEFAULT 0,
    last_error TEXT
);
"""


def _ensure_schema() -> None:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_QUEUE_SCHEMA)


@dataclass
class QueueItem:
    id: int
    op_type: str
    payload: dict[str, Any]
    attempts: int


def enqueue(op_type: str, payload: dict[str, Any]) -> int:
    _ensure_schema()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute(
            "INSERT INTO thinktank_queue (enqueued_at, op_type, payload) "
            "VALUES (?, ?, ?)",
            (time.time(), op_type, json.dumps(payload, ensure_ascii=False)),
        )
        log.debug(f"enqueue {op_type} #{cur.lastrowid}")
        return cur.lastrowid or -1


def _pending(limit: int = 50) -> list[QueueItem]:
    _ensure_schema()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM thinktank_queue ORDER BY enqueued_at LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        QueueItem(
            id=r["id"],
            op_type=r["op_type"],
            payload=json.loads(r["payload"]),
            attempts=r["attempts"],
        )
        for r in rows
    ]


def _delete(item_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM thinktank_queue WHERE id = ?", (item_id,))


def _bump_attempts(item_id: int, error: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "UPDATE thinktank_queue SET attempts = attempts + 1, last_error = ? "
            "WHERE id = ?",
            (error[:500], item_id),
        )


async def flush_once() -> tuple[int, int]:
    """큐 일괄 재전송 시도. (성공, 실패) 카운트 반환."""
    items = _pending()
    if not items:
        return 0, 0
    log.info(f"오프라인 큐 flush: {len(items)}개 처리")
    ok, fail = 0, 0
    try:
        async with ThinkTankClient() as client:
            health = await client.healthcheck()
            if not health:
                log.debug("ThinkTank 다운 — flush 중단")
                return 0, len(items)
            for it in items:
                try:
                    if it.op_type == "journal":
                        await client.add_journal(JournalEntry(**it.payload))
                    elif it.op_type == "calendar":
                        await client.add_calendar_event(CalendarEvent(**it.payload))
                    elif it.op_type == "observation":
                        await client.post_observation(
                            it.payload["type"], it.payload["data"],
                        )
                    else:
                        log.warning(f"미지원 op_type: {it.op_type}")
                        _delete(it.id)
                        continue
                    _delete(it.id)
                    ok += 1
                except Exception as e:
                    fail += 1
                    _bump_attempts(it.id, str(e))
    except Exception as e:
        log.warning(f"flush 일괄 실패: {e}")
        return ok, len(items) - ok
    log.info(f"  결과: ✓ {ok} / ✗ {fail}")
    return ok, fail


async def run_flusher(interval_sec: int = 300) -> None:
    """5분마다 큐 flush 시도."""
    while True:
        await asyncio.sleep(interval_sec)
        await flush_once()
