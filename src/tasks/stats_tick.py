"""스탯 회복 task — 1분마다 사용자 존재 여부 보고 stats 갱신.

decay는 stats.get() 호출 시 자동 적용되니까 여기선 회복 트리거만.
"""

from __future__ import annotations

import asyncio

from src.brain import stats as robot_stats
from src.brain.state_machine import StateContext
from src.utils.logger import get_logger

log = get_logger("stats_tick")

INTERVAL_SEC = 60.0


async def run_stats_tick(ctx: StateContext) -> None:
    log.info("stats tick 시작")
    while True:
        await asyncio.sleep(INTERVAL_SEC)
        if ctx.user_present:
            robot_stats.on_presence_tick(INTERVAL_SEC)
