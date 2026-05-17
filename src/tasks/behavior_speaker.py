"""행동 관찰 멘트 — vision/sensor 이벤트 보고 자연스럽게 한 마디.

각 행동 종류별 멘트 풀 + 종류별 쿨다운. quiet hours 동안엔 모두 skip.
chitchat trigger보다 더 자주, 더 즉각적으로 발동 (이벤트 즉시 반응).
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime

from src.audio.fake_tts import speak as fake_speak
from src.brain import memory
from src.brain.state_machine import State, StateContext
from src.brain.triggers import _is_quiet_hours
from src.face.expressions import Expression
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("behavior_speaker")


# 종류별 마지막 발화 시각 + 쿨다운 (모듈 레벨 — task 간 공유)
_LAST_AT: dict[str, float] = {}
DEFAULT_COOLDOWN_SEC = 30.0


def _busy_state(ctx: StateContext) -> bool:
    return ctx.state in (State.TALKING, State.LISTENING, State.GREETING)


_GREETING_KINDS = {"reappear", "face_recognize"}


def say(
    face: FaceState,
    ctx: StateContext,
    text: str,
    *,
    kind: str,
    cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
    expression: Expression | None = None,
) -> bool:
    """행동 멘트 발화 시도. 발화하면 True.

    - text가 빈 문자열이면 skip
    - quiet hours / busy state면 skip
    - kind별 쿨다운
    - 인사류(reappear/face_recognize)는 마지막 인사 후 5분 안엔 skip
    """
    if not text:
        return False
    if _busy_state(ctx):
        return False
    if _is_quiet_hours():
        return False
    now = time.time()
    # 인사류 전역 cooldown (greeting trigger와 공유 — 인사 중복 방지)
    if kind in _GREETING_KINDS:
        if ctx.last_greeting_at and now - ctx.last_greeting_at < 300.0:
            return False
    last = _LAST_AT.get(kind, 0.0)
    if now - last < cooldown_sec:
        return False
    _LAST_AT[kind] = now

    if expression is not None:
        face.apply_expression(expression)
    log.info(f"🗣️  [{kind}] {text}")
    asyncio.create_task(fake_speak(face, text))
    ctx.last_proactive_at = now
    if kind in _GREETING_KINDS:
        ctx.last_greeting_at = now
    try:
        memory.log_robot(text, kind=kind)
    except Exception as e:
        log.debug(f"conversation log 실패: {e}")
    return True


# ─── 행동별 멘트 풀 ───

def _now_period() -> str:
    h = datetime.now().hour
    if 5 <= h < 11:
        return "morning"
    if 11 <= h < 14:
        return "lunch"
    if 14 <= h < 18:
        return "afternoon"
    if 18 <= h < 22:
        return "evening"
    return "late"


def _name_prefix(ctx: StateContext) -> str:
    """이름 알면 '{이름}아, '나 '{이름}! ' 같이 prefix."""
    name = getattr(ctx, "user_name", None)
    if not name:
        return ""
    suffix = random.choice([f"{name}아, ", f"{name}, ", f"{name}! "])
    return suffix


def _pick_fresh(pool: tuple[str, ...], minutes: float = 30.0) -> str:
    """풀에서 최근 N분 동안 안 한 멘트 우선 선택."""
    try:
        recent = set(memory.recent_robot_messages(minutes=minutes))
    except Exception:
        recent = set()
    fresh = [m for m in pool if not any(m in r for r in recent)]
    return random.choice(fresh if fresh else pool)


# ─── 짧은 부재 후 재등장 (60초 미만) ───
REAPPEAR_SHORT = (
    "금방 왔네!", "어디 갔다 왔어?", "다녀왔어?", "왔어왔어.",
    "휙 갔다 오네.", "잠깐 어디 갔었어?", "다시 봐서 반가워.",
    "벌써 왔어?", "오, 빨리 왔네.",
)

# ─── 긴 부재 후 재등장 (1분~10분) ───
REAPPEAR_MEDIUM = (
    "어! 왔구나.", "오, 다시 왔네.", "잘 다녀왔어?", "어서 와.",
    "기다리고 있었어.", "왔네!", "어디 다녀와?",
)

# ─── 매우 긴 부재 (10분+) ───
REAPPEAR_LONG = (
    "오랜만이야!", "한참 안 보였네, 잘 있었어?", "오, 돌아왔구나.",
    "오랜만에 봐서 반가워.", "어디 갔다 왔어, 한참 만에.",
    "보고 싶었어.", "잘 지냈어?",
)

# 시간대별 인사 ── 등장/wave 다양화용
TIME_GREETINGS = {
    "morning": (
        "좋은 아침!", "아침이네, 잘 잤어?", "굿모닝!",
        "오늘 컨디션 어때?", "아침은 챙겨 먹었어?",
        "오늘도 화이팅!", "안녕, 오늘 잘 부탁해.",
    ),
    "lunch": (
        "점심 시간이네!", "점심 먹었어?", "오늘 점심 뭐였어?",
        "안녕, 배 안 고파?", "잘 챙겨 먹어야 해.",
    ),
    "afternoon": (
        "안녕!", "오후엔 좀 노곤하지.", "오후도 화이팅이야.",
        "왔어, 반가워.", "잘 지내고 있어?",
    ),
    "evening": (
        "안녕, 오늘 수고했어.", "저녁이네!", "퇴근했어?",
        "오늘 어땠어?", "저녁 먹을 시간이지.", "잘 지내, 늦지 마.",
    ),
    "late": (
        "아직 안 자?", "늦은 시간에 보네.", "안녕, 잘 지내?",
        "이 시간에 깨어있구나.",
    ),
}

# 거리 변화 — 시간대 무관
GOT_CLOSER = (
    "어, 가까이 왔네?", "응? 왜?", "뭐 보여줄 거 있어?",
    "오, 가깝다.", "잘 보이게 왔어?", "더 가까이서 보고 싶었구나.",
    "여기 있어, 봐줘서 좋아.",
)

GOT_FARTHER = (
    "어디 가?", "왜 멀어져?", "어... 가지 마.",
    "멀어졌네.", "잠깐, 어디 가?", "다시 와줘.",
)

# 얼굴 인식 후 이름 부르기
FACE_GREETING_TEMPLATES = (
    "{name}이다! 안녕!", "어, {name} 왔네.",
    "{name}, 반가워!", "{name} 오랜만이야.",
    "왔네, {name}.", "{name}! 보고 싶었어.",
    "안녕 {name}, 오늘 어때?",
)


def reappear_message(absence_sec: float, ctx: StateContext) -> str:
    """부재 시간 + 시간대 + 이름 조합으로 멘트 선택. 최근 발화 안 반복."""
    name_pre = _name_prefix(ctx)
    period = _now_period()
    if absence_sec >= 600:
        pool = REAPPEAR_LONG
    elif absence_sec >= 60:
        pool = (TIME_GREETINGS.get(period, ()) + REAPPEAR_MEDIUM
                if random.random() < 0.5 else REAPPEAR_MEDIUM)
    else:
        pool = REAPPEAR_SHORT
    base = _pick_fresh(pool)
    return name_pre + base


def closer_message() -> str:
    return _pick_fresh(GOT_CLOSER)


def farther_message() -> str:
    return _pick_fresh(GOT_FARTHER)


def face_greeting_message(name: str) -> str:
    template = _pick_fresh(FACE_GREETING_TEMPLATES)
    return template.format(name=name)


_WAVE_SHORT = (
    "안녕?", "어, 안녕!", "반가워!", "오, 손 흔들어줬네!",
    "안녕 안녕!", "헤이~", "와, 봐줘서 좋아.",
    "오, 너구나!", "보고 싶었어.",
)


def wave_back_message(ctx: StateContext) -> str:
    """손 흔들기 답례 — 시간대 + 이름 인식 + 최근 발화 안 반복."""
    period = _now_period()
    name_pre = _name_prefix(ctx)
    if random.random() < 0.5:
        pool = TIME_GREETINGS.get(period, _WAVE_SHORT)
    else:
        pool = _WAVE_SHORT
    base = _pick_fresh(pool)
    return name_pre + base
