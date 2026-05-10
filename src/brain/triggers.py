"""트리거 평가기 — 1Hz로 호출되어 "지금 개입할까?" 판단.

각 트리거 함수는 ProactiveTrigger 또는 None 반환.
상태 머신 + 메모리 + 최근 센서 이벤트를 종합 분석.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.brain import memory
from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR
from src.utils.logger import get_logger

log = get_logger("triggers")


@dataclass
class ProactiveTrigger:
    """능동 멘트 트리거."""

    kind: str           # "work_break" / "env_change" / "greeting" / "long_silence" 등
    priority: int       # 0=low, 10=high
    context: dict       # LLM 프롬프트에 넣을 정보
    suggested_message: str | None = None   # 미리 정한 멘트가 있으면


# === 헬퍼 ===

def _is_quiet_hours(now: datetime | None = None) -> bool:
    now = now or datetime.now()
    start, end = BEHAVIOR.proactive_quiet_hours
    h = now.hour
    if start < end:
        return start <= h < end
    return h >= start or h < end


def _proactive_allowed(ctx: StateContext) -> bool:
    """능동 멘트 가능한 상태인지 종합 체크."""
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return False
    if not ctx.user_present:
        return False
    if _is_quiet_hours():
        return False
    if memory.proactive_count_last_hour() >= BEHAVIOR.proactive_max_per_hour:
        return False
    if ctx.last_proactive_at and (
        time.time() - ctx.last_proactive_at < BEHAVIOR.proactive_min_silence_sec
    ):
        return False
    return True


# === 개별 트리거 ===

def check_work_break(
    ctx: StateContext, current_session_id: int | None,
) -> Optional[ProactiveTrigger]:
    """장시간 작업 시 휴식 권유."""
    if current_session_id is None:
        return None
    duration_min = memory.current_work_duration(current_session_id) / 60
    if duration_min >= BEHAVIOR.work_break_alarm_minutes:
        return ProactiveTrigger(
            kind="work_break_alarm",
            priority=9,
            context={"work_minutes": int(duration_min)},
            suggested_message=f"{int(duration_min)}분 동안 쉬지 않으셨어요. 진짜 잠깐만 일어나주세요.",
        )
    if duration_min >= BEHAVIOR.work_break_strong_minutes:
        return ProactiveTrigger(
            kind="work_break_strong",
            priority=7,
            context={"work_minutes": int(duration_min)},
            suggested_message=f"벌써 {int(duration_min // 60)}시간이나 앉아 계셨네요. 잠깐 스트레칭은 어떠세요?",
        )
    if duration_min >= BEHAVIOR.work_break_warn_minutes:
        return ProactiveTrigger(
            kind="work_break_warn",
            priority=4,
            context={"work_minutes": int(duration_min)},
        )
    return None


def check_greeting(ctx: StateContext) -> Optional[ProactiveTrigger]:
    """방금 등장한 사용자에게 인사."""
    if ctx.state == State.GREETING:
        return None
    if not ctx.user_present:
        return None
    # 등장 직후 (last_user_seen_at이 매우 최근에 set됨)
    if ctx.last_user_seen_at and time.time() - ctx.last_user_seen_at < 3.0:
        # 그 직전 부재가 충분히 길었나? (단순 절전 깨우기와 구분)
        # 실제 구현은 sensor manager에서 이벤트 흐름으로 판단
        return ProactiveTrigger(
            kind="greeting",
            priority=8,
            context={},
            suggested_message=None,  # LLM이 시간대 보고 결정
        )
    return None


def check_long_silence(ctx: StateContext) -> Optional[ProactiveTrigger]:
    """4시간 이상 말 안 걸었으면 안부."""
    if not _proactive_allowed(ctx):
        return None
    if ctx.last_proactive_at is None:
        # 처음 시작 시 너무 빨리 발동되지 않게
        return None
    silence_sec = time.time() - ctx.last_proactive_at
    if silence_sec > 4 * 3600:
        return ProactiveTrigger(
            kind="long_silence",
            priority=2,
            context={"silence_hours": silence_sec / 3600},
        )
    return None


# === 통합 평가 ===

def evaluate_all(
    ctx: StateContext,
    current_session_id: int | None = None,
    env: dict | None = None,
) -> list[ProactiveTrigger]:
    """모든 트리거 평가. 우선순위 정렬해서 반환."""
    triggers: list[ProactiveTrigger] = []

    # 인사는 항상 우선
    if (g := check_greeting(ctx)) is not None:
        triggers.append(g)

    if _proactive_allowed(ctx):
        if (w := check_work_break(ctx, current_session_id)) is not None:
            triggers.append(w)
        if (s := check_long_silence(ctx)) is not None:
            triggers.append(s)

    triggers.sort(key=lambda t: -t.priority)
    return triggers
