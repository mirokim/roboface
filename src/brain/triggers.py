"""트리거 평가기 — 1Hz로 호출되어 "지금 개입할까?" 판단.

각 트리거 함수는 ProactiveTrigger 또는 None 반환.
상태 머신 + 메모리 + 최근 센서 이벤트를 종합 분석.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.brain import memory
from src.brain.perception import PerceptionState
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


# 잡담 멘트 풀 — 상황별. LLM 없이도 자연스러운 한국어.
# 트리거 발동 시 현재 상황에서 적합한 풀들 합쳐서 랜덤 선택.

_CHITCHAT_GENERIC = (
    "오늘은 좀 어때?",
    "잘 지내고 있어?",
    "지금 뭐 하고 있어?",
    "기분은 어때?",
    "잘 하고 있어, 그 페이스 좋아.",
    "조금 쉬어도 괜찮아.",
    "심호흡 한번 해볼래?",
    "괜찮아? 너무 무리하지 마.",
)

_CHITCHAT_HOT = (
    "더워… 에어컨 좀 켤까?",
    "물 자주 마셔, 더운 날이야.",
    "선풍기라도 켤까?",
    "땀나는데, 잠깐 시원한 데 가있을래?",
)

_CHITCHAT_WARM = (
    "오늘 좀 후덥지근하네.",
    "물 한 잔 어때?",
    "환기 한 번 시킬까?",
)

_CHITCHAT_COOL = (
    "쌀쌀한데 따뜻한 거 한 잔 어때?",
    "스웨터 하나 걸치자.",
    "따뜻한 차 한 잔 어때?",
)

_CHITCHAT_COLD = (
    "춥다… 히터 켤까?",
    "추워서 떨려. 담요 어디 있어?",
    "이런 날엔 핫초코지.",
    "발 시리지 않아? 따뜻하게 입어.",
)

_CHITCHAT_MORNING = (
    "좋은 아침이야!",
    "오늘 컨디션 어때?",
    "아침은 챙겨 먹었어?",
    "오늘 하루도 화이팅.",
)

_CHITCHAT_LUNCH = (
    "점심 뭐 먹을 거야?",
    "벌써 점심 시간이네.",
    "맛있는 거 먹어.",
)

_CHITCHAT_AFTERNOON = (
    "오후엔 살짝 졸리지.",
    "커피 한 잔 어때?",
    "잠깐 산책이라도 갈래?",
)

_CHITCHAT_EVENING = (
    "오늘 하루도 수고 많았어.",
    "저녁은 챙겨 먹었어?",
    "이제 슬슬 정리할 시간이지?",
    "하늘 색깔 한번 봐봐.",
)

_CHITCHAT_WORK_LONG = (
    "벌써 한참 일했네. 잠깐 쉴까?",
    "어깨 좀 펴봐, 굳었겠다.",
    "눈 좀 감고 30초만 쉬자.",
    "허리 똑바로! 자세 무너졌어.",
    "기지개 한번 쫙 펴봐.",
    "물 마시러 일어나자.",
)


def _time_of_day_pool(now: datetime) -> tuple[str, ...]:
    h = now.hour
    if 6 <= h < 11:
        return _CHITCHAT_MORNING
    if 11 <= h < 14:
        return _CHITCHAT_LUNCH
    if 14 <= h < 18:
        return _CHITCHAT_AFTERNOON
    if 18 <= h < 22:
        return _CHITCHAT_EVENING
    return ()  # 그 외 시간대는 generic만


def _temp_pool(temp_c: float | None) -> tuple[str, ...]:
    if temp_c is None:
        return ()
    if temp_c >= 30:
        return _CHITCHAT_HOT
    if temp_c >= 26:
        return _CHITCHAT_WARM
    if temp_c <= 15:
        return _CHITCHAT_COLD
    if temp_c <= 19:
        return _CHITCHAT_COOL
    return ()


def _build_chitchat_pool(
    perception: PerceptionState | None,
    work_minutes: float | None,
) -> tuple[tuple[str, ...], ...]:
    """현재 상황에 맞는 멘트 풀들의 튜플 반환.

    상황별 풀은 generic보다 가중치 약 2배 (튜플로 두 번 들어가게).
    """
    pools: list[tuple[str, ...]] = [_CHITCHAT_GENERIC]
    now = datetime.now()
    if (tod := _time_of_day_pool(now)):
        pools.append(tod)
        pools.append(tod)
    temp_c = perception.temperature_c if perception is not None else None
    if (tp := _temp_pool(temp_c)):
        pools.append(tp)
        pools.append(tp)
    # 오래 앉아 있으면 work_long 풀 가중치 더 크게
    if work_minutes is not None and work_minutes >= 45:
        pools.append(_CHITCHAT_WORK_LONG)
        pools.append(_CHITCHAT_WORK_LONG)
    return tuple(pools)


def _pick_chitchat_message(
    perception: PerceptionState | None,
    work_minutes: float | None,
) -> str:
    pools = _build_chitchat_pool(perception, work_minutes)
    # 풀 중 하나 선택 → 그 안에서 멘트 하나 선택 (균등 가중치)
    chosen_pool = random.choice(pools)
    return random.choice(chosen_pool)


def check_chitchat(
    ctx: StateContext,
    perception: PerceptionState | None = None,
    current_session_id: int | None = None,
) -> Optional[ProactiveTrigger]:
    """일정 간격으로 상황 맞춰 가벼운 잡담 — 사용자 곁에 있다는 느낌."""
    if not _proactive_allowed(ctx):
        return None
    last = ctx.last_proactive_at or 0.0
    silence_sec = time.time() - last
    if silence_sec < BEHAVIOR.chitchat_min_interval_sec:
        return None
    if silence_sec >= BEHAVIOR.chitchat_max_interval_sec:
        prob = 1.0
    else:
        span = BEHAVIOR.chitchat_max_interval_sec - BEHAVIOR.chitchat_min_interval_sec
        prob = (silence_sec - BEHAVIOR.chitchat_min_interval_sec) / span
    if random.random() > prob:
        return None

    work_minutes: float | None = None
    if current_session_id is not None:
        work_minutes = memory.current_work_duration(current_session_id) / 60

    msg = _pick_chitchat_message(perception, work_minutes)
    ctx_data: dict = {}
    if perception is not None and perception.temperature_c is not None:
        ctx_data["temperature_c"] = perception.temperature_c
    if work_minutes is not None:
        ctx_data["work_minutes"] = int(work_minutes)
    return ProactiveTrigger(
        kind="chitchat",
        priority=1,
        context=ctx_data,
        suggested_message=msg,
    )


# === 통합 평가 ===

def evaluate_all(
    ctx: StateContext,
    current_session_id: int | None = None,
    env: dict | None = None,
    perception: PerceptionState | None = None,
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
        if (c := check_chitchat(
            ctx, perception=perception, current_session_id=current_session_id,
        )) is not None:
            triggers.append(c)

    triggers.sort(key=lambda t: -t.priority)
    return triggers
