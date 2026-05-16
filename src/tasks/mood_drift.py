"""Mood drift — 시간대 + 부재시간 기반으로 베이스 표정이 미묘하게 흐름.

다른 task가 명시적 표정을 set 안 한 평상시(IDLE/WATCHING)에만 동작.
TALKING/GREETING/LISTENING/ALERTING 중엔 절대 끼어들지 않음.

흐름 예시:
- 아침: NEUTRAL (default 기분)
- 사용자 등장 직후 30초: HAPPY (반가움)
- 사용자 부재 5분~30분: CONTENT (혼자 만족)
- 사용자 부재 30분+: SLEEPY (멍 때림)
- 밤 22시~6시: SLEEPY 기본
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime

from src.brain.state_machine import State, StateContext
from src.face import expressions as expr
from src.face.expressions import Expression
from src.face.renderer import FaceState
from src.tasks import reactive_face
from src.utils.logger import get_logger

log = get_logger("mood_drift")


CHECK_INTERVAL_SEC = 30.0

# mood가 적용되는 상태들 — 그 외엔 절대 건드리지 않음
_ELIGIBLE_STATES = (State.IDLE, State.WATCHING)


def _select_mood(ctx: StateContext, now_ts: float, hour: int) -> Expression:
    """현재 상황에 어울리는 베이스 표정."""
    # 밤늦은 시간엔 졸린 게 기본
    night = hour >= 22 or hour < 6

    # 사용자 등장 직후 — 잠깐 반가운 분위기
    if ctx.user_present and ctx.last_user_seen_at:
        since_seen = now_ts - ctx.last_user_seen_at
        if since_seen < 30:
            # 처음 만나 살짝 반가운 모드, 가끔 STARSTRUCK
            return random.choices(
                [expr.HAPPY, expr.CONTENT, expr.STARSTRUCK],
                weights=[6, 3, 1],
            )[0]

    # 사용자 부재
    if not ctx.user_present:
        if ctx.last_user_seen_at is None:
            absent = 0.0
        else:
            absent = now_ts - ctx.last_user_seen_at
        if absent > 60 * 60:
            # 1시간+ 부재 → 졸음/하품
            return random.choices(
                [expr.SLEEPY, expr.YAWN],
                weights=[8, 2],
            )[0]
        if absent > 30 * 60:
            return expr.SLEEPY
        if absent > 5 * 60:
            return random.choices(
                [expr.CONTENT, expr.NEUTRAL, expr.THINKING],
                weights=[5, 3, 2],
            )[0]

    # 평상시
    if night:
        return random.choices(
            [expr.SLEEPY, expr.CONTENT],
            weights=[7, 3],
        )[0]
    if 6 <= hour < 10:
        # 아침엔 살짝 졸린 듯
        return random.choices(
            [expr.SLEEPY, expr.NEUTRAL, expr.CONTENT],
            weights=[3, 5, 2],
        )[0]
    return random.choices(
        [expr.NEUTRAL, expr.CONTENT, expr.THINKING, expr.HAPPY],
        weights=[4, 3, 2, 1],
    )[0]


async def run_mood_drift(face: FaceState, ctx: StateContext) -> None:
    """30초마다 베이스 표정 재선정."""
    log.info("mood drift 시작")
    while True:
        await asyncio.sleep(CHECK_INTERVAL_SEC)

        if ctx.state not in _ELIGIBLE_STATES:
            continue
        if reactive_face.is_locked():
            # 잠깐 reactive 표정 표시 중 — 양보
            continue

        now_ts = time.time()
        hour = datetime.now().hour
        new_mood = _select_mood(ctx, now_ts, hour)

        if face.expression.name != new_mood.name:
            log.debug(f"mood drift: {face.expression.name} → {new_mood.name}")
            face.apply_expression(new_mood)
