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

from src.brain import stats as robot_stats
from src.brain.state_machine import State, StateContext
from src.face import expressions as expr
from src.face.expressions import Expression
from src.face.renderer import FaceState
from src.tasks import reactive_face
from src.utils.logger import get_logger

log = get_logger("mood_drift")


CHECK_INTERVAL_SEC = 15.0   # 30→15 — 표정 변화 더 자주

# mood가 적용되는 상태들 — 그 외엔 절대 건드리지 않음
_ELIGIBLE_STATES = (State.IDLE, State.WATCHING)


def _select_mood(ctx: StateContext, now_ts: float, hour: int) -> Expression:
    """현재 상황에 어울리는 베이스 표정. 스탯 영향 우선 반영."""
    # 1) 스탯 기반 강한 추천 (낮은 스탯이 있으면 우선)
    suggested = robot_stats.suggested_expression()
    if suggested:
        ex = getattr(expr, suggested, None)
        if ex is not None:
            return ex

    # 밤늦은 시간엔 졸린 게 기본
    night = hour >= 22 or hour < 6

    # 사용자 등장 직후 — 잠깐 반가운 분위기
    if ctx.user_present and ctx.last_user_seen_at:
        since_seen = now_ts - ctx.last_user_seen_at
        if since_seen < 30:
            return random.choices(
                [expr.HAPPY, expr.CONTENT, expr.STARSTRUCK, expr.LOVE, expr.WINK],
                weights=[5, 3, 1, 1, 1],
            )[0]

    # 사용자 부재
    if not ctx.user_present:
        if ctx.last_user_seen_at is None:
            absent = 0.0
        else:
            absent = now_ts - ctx.last_user_seen_at
        if absent > 60 * 60:
            # 1시간+ 부재 → 졸음/하품/멍
            return random.choices(
                [expr.SLEEPY, expr.YAWN, expr.NEUTRAL],
                weights=[6, 2, 2],
            )[0]
        if absent > 30 * 60:
            return random.choices(
                [expr.SLEEPY, expr.CONTENT, expr.THINKING],
                weights=[6, 2, 2],
            )[0]
        if absent > 5 * 60:
            return random.choices(
                [expr.CONTENT, expr.NEUTRAL, expr.THINKING, expr.CURIOUS],
                weights=[4, 3, 2, 1],
            )[0]

    # 평상시 — 풀 더 풍부하게. 자주 안 쓰이는 표정도 가끔.
    if night:
        return random.choices(
            [expr.SLEEPY, expr.CONTENT, expr.THINKING, expr.NEUTRAL],
            weights=[5, 2, 2, 1],
        )[0]
    if 6 <= hour < 10:
        # 아침엔 살짝 졸린 듯 + 가끔 하품
        return random.choices(
            [expr.SLEEPY, expr.NEUTRAL, expr.CONTENT, expr.YAWN, expr.THINKING],
            weights=[3, 4, 2, 1, 1],
        )[0]
    # 낮 평상시 — 다양한 표정 섞기 (CURIOUS/WINK/PROUD/LOVE 가끔)
    return random.choices(
        [
            expr.NEUTRAL, expr.CONTENT, expr.THINKING, expr.HAPPY,
            expr.CURIOUS, expr.FOCUSED, expr.PROUD,
            expr.WINK, expr.WINK_R, expr.LOVE,
        ],
        weights=[4, 3, 2, 2, 2, 1, 1, 1, 1, 1],
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
