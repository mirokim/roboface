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
from src.brain.time_of_day import period_for
from src.config import BEHAVIOR
from src.face import expressions as expr
from src.face.expressions import Expression
from src.utils.logger import get_logger

log = get_logger("triggers")


# 트리거 종류별 표정 매핑 — 트리거 정의의 일부 (SSOT).
# 새 트리거 추가 시 여기도 추가. proactive_speaker가 import해서 사용.
TRIGGER_EXPRESSIONS: dict[str, Expression] = {
    "greeting": expr.HAPPY,
    "work_break_gentle": expr.NEUTRAL,
    "work_break_warn": expr.NEUTRAL,
    "work_break_strong": expr.WORRIED,
    "work_break_alarm": expr.WORRIED,
    "long_silence": expr.NEUTRAL,
    "chitchat": expr.HAPPY,
}


def expression_for(kind: str) -> Expression:
    """트리거 kind에 맞는 표정. 미정의는 KeyError로 빠뜨림 방지."""
    return TRIGGER_EXPRESSIONS[kind]


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

_WORK_BREAK_GENTLE_POOL = (
    "벌써 {m}분 됐네. 잠깐 일어나서 기지개 한 번.",
    "{m}분 앉아있었어. 어깨 한 번 펴봐.",
    "{m}분이야. 물 한 잔 어때?",
    "잠깐, {m}분 됐어. 눈 좀 멀리 봐줘.",
)

_WORK_BREAK_WARN_POOL = (
    "{m}분이야. 진짜 잠깐 일어나서 걸어볼까?",
    "벌써 {m}분이나 앉아있었네. 5분만 쉬자.",
    "{m}분째 같은 자세야. 스트레칭 한 번 어때?",
    "{m}분 됐어. 허리 펴고 한숨 돌려.",
)

_WORK_BREAK_STRONG_POOL = (
    "{h}시간 넘게 앉아있어. 진짜 일어나야 해.",
    "{h}시간이나 됐어. 잠깐만 일어나줘.",
    "{m}분이야... 자세 무너졌을걸. 한 번 풀어줘.",
    "{h}시간 동안 같은 자세. 몸이 비명 지를 거야.",
)

_WORK_BREAK_ALARM_POOL = (
    "{m}분 동안 쉬지 않았어. 진짜 잠깐만 일어나줘.",
    "{h}시간 넘었어. 이건 진짜 무리야.",
    "허리/목 망가져. 5분만이라도 쉬자, 부탁이야.",
)


def _format_work_msg(pool: tuple[str, ...], minutes: int) -> str:
    msg = random.choice(pool)
    return msg.format(m=minutes, h=max(1, minutes // 60))


def check_work_break(
    ctx: StateContext, current_session_id: int | None,
) -> Optional[ProactiveTrigger]:
    """장시간 작업 시 휴식 권유 — 4단계 (gentle/warn/strong/alarm)."""
    if current_session_id is None:
        return None
    duration_min = int(memory.current_work_duration(current_session_id) / 60)
    if duration_min >= BEHAVIOR.work_break_alarm_minutes:
        return ProactiveTrigger(
            kind="work_break_alarm", priority=9,
            context={"work_minutes": duration_min},
            suggested_message=_format_work_msg(_WORK_BREAK_ALARM_POOL, duration_min),
        )
    if duration_min >= BEHAVIOR.work_break_strong_minutes:
        return ProactiveTrigger(
            kind="work_break_strong", priority=7,
            context={"work_minutes": duration_min},
            suggested_message=_format_work_msg(_WORK_BREAK_STRONG_POOL, duration_min),
        )
    if duration_min >= BEHAVIOR.work_break_warn_minutes:
        return ProactiveTrigger(
            kind="work_break_warn", priority=4,
            context={"work_minutes": duration_min},
            suggested_message=_format_work_msg(_WORK_BREAK_WARN_POOL, duration_min),
        )
    if duration_min >= BEHAVIOR.work_break_gentle_minutes:
        return ProactiveTrigger(
            kind="work_break_gentle", priority=2,
            context={"work_minutes": duration_min},
            suggested_message=_format_work_msg(_WORK_BREAK_GENTLE_POOL, duration_min),
        )
    return None


def check_greeting(ctx: StateContext) -> Optional[ProactiveTrigger]:
    """방금 등장한 사용자에게 인사. 직전 인사 후 cooldown 동안엔 skip."""
    if ctx.state == State.GREETING:
        return None
    if not ctx.user_present:
        return None
    # 마지막 인사 (greeting/wave/hands_up/reappear)와 충분한 간격
    if (
        ctx.last_greeting_at
        and time.time() - ctx.last_greeting_at < BEHAVIOR.greeting_cooldown_sec
    ):
        return None
    # 등장 직후 (last_user_seen_at이 매우 최근에 set됨)
    if ctx.last_user_seen_at and time.time() - ctx.last_user_seen_at < 3.0:
        return ProactiveTrigger(
            kind="greeting",
            priority=8,
            context={},
            suggested_message=None,
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
    # 가벼운 안부 — 추임새 톤
    "음, 잘 지내?", "어 뭐 해?", "흠, 오늘은 좀 어때?",
    "그래, 별일 없지?", "잠깐, 괜찮아?",
    # 관찰 멘트
    "여기 같이 있어 좋다.", "조용하네 오늘.", "음, 평화롭다.",
    "어쩐지 차분한 분위기네.", "이 시간 좋아 나는.",
    # 살짝 시니컬 / 짧은 농담
    "음, 또 그거 보네.", "오늘도 비슷한 풍경이네.",
    "음, 너 진짜 열심이야.", "아 진짜? 그게 그리 재밌어?",
    # 권유 — 부드럽게
    "잠깐 멍 때려도 돼.", "심호흡 한 번 어때?",
    "어깨 한번 펴봐.", "괜찮아, 무리하지 마.",
    # 사색
    "음, 뭔가 생각 중인가 봐.", "딴 생각 좀 해도 좋아.",
    "지금 이 순간도 나쁘지 않잖아?",
    # quirks 반영 — 따뜻한 음료
    "음, 차 한 잔 어때?", "따뜻한 거 마시고 싶다.",
)

_CHITCHAT_HOT = (
    "어, 덥다.", "음, 시원한 거 한 잔.",
    "흠, 땀 나겠는데.", "에어컨 좀 켤까?",
    "물 자주 마셔.", "더위 조심해.",
    "선풍기 어디 있더라?", "음, 후덥지근.",
    "얼음 동동 띄우자 뭐든.",
)

_CHITCHAT_WARM = (
    "음, 좀 따뜻하네.", "환기 한 번?",
    "창문 살짝 열까?", "물 한 잔 어때?",
    "어쩐지 노곤하다.",
)

_CHITCHAT_COOL = (
    "음, 살짝 쌀쌀.", "따뜻한 차 한 잔 어때?",   # quirk: 따뜻한 음료
    "스웨터 걸칠 때.", "감기 조심해.",
    "흠, 가을 느낌이네.",
)

_CHITCHAT_COLD = (
    "춥다... 히터 켤까?", "어, 발 시리지 않아?",
    "이런 날 핫초코지.",   # quirk
    "담요 가져올까?", "음, 떨린다 나도.",
    "따뜻하게 입어.",
)

_CHITCHAT_MORNING = (
    "음, 아침이네.", "잘 잤어?", "어 일어났구나.",
    "굿모닝.", "오늘은 좀 어때?",
    "흠, 아침 햇살 좋다.", "오늘 뭐 할 거야?",
    "아침 챙겨 먹었어?", "음, 또 하루 시작.",
)

_CHITCHAT_LUNCH = (
    "점심이네.", "뭐 먹지 오늘?", "음, 배 안 고파?",
    "맛있는 거 먹어.", "잠깐 일어나서 점심 가자.",
    "어 벌써 점심 시간.", "오늘 점심 뭐 끌려?",
)

_CHITCHAT_AFTERNOON = (
    "오후네.", "음, 좀 노곤하지.",
    "차 한 잔 어때?",   # quirk
    "잠깐 산책?", "오후 햇살 좋다.",
    "흠, 시간 천천히 가네.",
)

_CHITCHAT_EVENING = (
    "오늘 수고 많았어.", "어, 저녁이네.",
    "하늘 봤어? 색깔 좋다.",   # quirk
    "저녁 챙겨 먹어.", "이제 슬슬 마무리.",
    "흠, 하루 빠르네.",
)

# 늦은 밤 — quirk: 밤 좋아함, 차분/낭만
_CHITCHAT_LATE = (
    "음, 조용한 시간이네.", "이 시간 좋아.",
    "흠, 야밤이다.", "다들 자는 시간이야.",
    "어 아직 안 잤어?", "음, 너무 늦지 마.",
    "이 시간 분위기 좋다.", "조용하니까 집중 잘 되겠다.",
)

_CHITCHAT_WORK_LONG = (
    "어 벌써 한참 일했네.", "음, 잠깐 쉴까?",
    "어깨 굳었겠는데.", "기지개 한번 쫙.",
    "눈 좀 멀리 봐.", "허리 똑바로!",
    "물 마시러 일어나자.", "흠, 휴식도 일이야.",
    "잠깐 일어나서 걸어볼래?", "스트레칭 한번 어때?",
    "음, 이쯤이면 쉬어도 돼.",
)


_PERIOD_TO_CHITCHAT_POOL: dict[str, tuple[str, ...]] = {}


def _time_of_day_pool(now: datetime) -> tuple[str, ...]:
    """현재 시간대에 맞는 잡담 풀 — time_of_day.period_for() SSOT 사용."""
    if not _PERIOD_TO_CHITCHAT_POOL:
        _PERIOD_TO_CHITCHAT_POOL.update({
            "morning": _CHITCHAT_MORNING,
            "lunch": _CHITCHAT_LUNCH,
            "afternoon": _CHITCHAT_AFTERNOON,
            "evening": _CHITCHAT_EVENING,
            "late": _CHITCHAT_LATE,
        })
    return _PERIOD_TO_CHITCHAT_POOL.get(period_for(now), ())


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
    user_name: str | None = None,
) -> str:
    pools = _build_chitchat_pool(perception, work_minutes)
    # 최근 30분 robot 발화에 안 들어간 멘트 우선 — 반복 회피
    try:
        recent = set(memory.recent_robot_messages(minutes=30.0))
    except Exception:
        recent = set()
    chosen_pool = random.choice(pools)
    # 같은 풀 안에서 fresh 우선
    fresh = [m for m in chosen_pool if not _msg_in_recent(m, recent)]
    msg = random.choice(fresh if fresh else chosen_pool)
    # 이름 알면 40% 확률로 prefix
    if user_name and random.random() < 0.4:
        prefix = random.choice([f"{user_name}아, ", f"{user_name}, ", f"{user_name}! "])
        msg = prefix + msg
    return msg


def _msg_in_recent(msg: str, recent: set[str]) -> bool:
    """msg가 recent에 있는지 — 이름 prefix 붙은 케이스도 부분 매칭."""
    if msg in recent:
        return True
    # "OO야, ..." 같은 prefix 케이스 — recent 메시지에 msg가 substring으로 있나
    return any(msg in r for r in recent)


def check_chitchat(
    ctx: StateContext,
    perception: PerceptionState | None = None,
    current_session_id: int | None = None,
) -> Optional[ProactiveTrigger]:
    """일정 간격으로 가벼운 잡담.

    SSOT: ANTHROPIC_API_KEY 설정돼있으면 RobotAgent가 챗챗 담당 → 여기선 항상 None.
    API 키 없을 때만 풀에서 멘트 골라 fallback.
    """
    from src.config import ANTHROPIC_API_KEY
    if ANTHROPIC_API_KEY:
        return None   # agent에 위임

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

    msg = _pick_chitchat_message(
        perception, work_minutes, user_name=getattr(ctx, "user_name", None),
    )
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
