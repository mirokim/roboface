"""DB 자동 정리 — 오래된 행 purge + VACUUM. 매일 새벽 4시.

보존 정책:
- conversation_log: 90일 (ts)
- proactive_log: 90일 (ts)
- env_log: 30일 (ts)
- learned_facts: 365일 (learned_at) — agent가 일부러 저장한 정보. 보수적 보존.
- face_snapshots: 이미 photo_memory.purge_old가 7일 처리. 여기선 orphan 정리만.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from src.brain import memory
from src.utils.logger import get_logger

log = get_logger("db_cleanup")


CLEANUP_HOUR = 4
CHECK_INTERVAL_SEC = 300.0   # 5분마다 확인

# (테이블, 시각 컬럼명, 보존일수)
RETENTION = [
    ("conversation_log", "ts", 90),
    ("proactive_log",    "ts", 90),
    ("env_log",          "ts", 30),
    ("learned_facts",    "learned_at", 365),
]


def _cleanup_once() -> dict[str, int]:
    """모든 테이블 cleanup 실행. 삭제된 행 수 dict 반환."""
    deleted: dict[str, int] = {}
    now = time.time()
    with memory.db() as conn:
        for table, col, days in RETENTION:
            cutoff = now - days * 86400
            cur = conn.execute(
                f"DELETE FROM {table} WHERE {col} < ?", (cutoff,),
            )
            deleted[table] = cur.rowcount or 0
    # VACUUM — 디스크 회수 (별도 connection 필요)
    try:
        import sqlite3
        from src.config import DB_PATH
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("VACUUM")
    except Exception as e:
        log.debug(f"VACUUM 실패: {e}")
    return deleted


async def run_db_cleanup() -> None:
    log.info(f"db cleanup task 시작 — 매일 {CLEANUP_HOUR:02d}시 정리")
    last_cleanup_date: str | None = None
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)
        now = datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        if last_cleanup_date == today_str:
            continue
        if now.hour < CLEANUP_HOUR:
            continue
        try:
            deleted = _cleanup_once()
            total = sum(deleted.values())
            log.info(f"db cleanup 완료 — 삭제 {deleted}, 합계 {total}행")
        except Exception as e:
            log.warning(f"db cleanup 에러: {e}")
        last_cleanup_date = today_str
