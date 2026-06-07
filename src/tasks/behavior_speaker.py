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
from src.brain.time_of_day import period_for
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


# 인사류 — 5분 cooldown 공유 (greeting trigger와 공통).
# **수동적 인사**만 포함 (사용자 등장/인식 등). 사용자가 능동적으로 손 흔들거나
# 만세 한 건 5분 cooldown 적용하면 너무 답답 — 거기는 kind별 짧은 cooldown만.
_GREETING_KINDS = {"reappear", "face_recognize"}


def say(
    face: FaceState,
    ctx: StateContext,
    text: str,
    *,
    kind: str,
    cooldown_sec: float = DEFAULT_COOLDOWN_SEC,
    expression: Expression | None = None,
    bypass_quiet: bool = False,
) -> "asyncio.Task | None":
    """행동 멘트 발화 시도. 발화 시 speech task 반환, skip 시 None.

    Gates:
    - text 빈 문자열
    - busy state (TALKING/LISTENING/GREETING)
    - quiet hours (bypass_quiet=True면 우회 — 사용자 능동 제스처 응답용)
    - kind별 cooldown
    - 인사류(_GREETING_KINDS): 마지막 인사 후 5분 안엔 skip
    """
    if not text:
        return None
    if _busy_state(ctx):
        log.debug(f"say skip [{kind}]: busy state ({ctx.state})")
        return None
    if _is_quiet_hours() and not bypass_quiet:
        log.debug(f"say skip [{kind}]: quiet hours")
        return None
    now = time.time()
    if kind in _GREETING_KINDS:
        if ctx.last_greeting_at and now - ctx.last_greeting_at < 300.0:
            remain = 300.0 - (now - ctx.last_greeting_at)
            log.debug(f"say skip [{kind}]: greeting cooldown {remain:.0f}s 남음")
            return None
    last = _LAST_AT.get(kind, 0.0)
    if now - last < cooldown_sec:
        remain = cooldown_sec - (now - last)
        log.debug(f"say skip [{kind}]: kind cooldown {remain:.1f}s 남음")
        return None
    _LAST_AT[kind] = now

    if expression is not None:
        face.apply_expression(expression)
    log.info(f"🗣️  [{kind}] {text}")
    task = asyncio.create_task(fake_speak(face, text))
    ctx.last_proactive_at = now
    if kind in _GREETING_KINDS:
        ctx.last_greeting_at = now
    try:
        memory.log_robot(text, kind=kind)
    except Exception as e:
        log.debug(f"conversation log 실패: {e}")
    return task


# ─── 행동별 멘트 풀 ───


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
    "어 왔어?", "음, 금방이네.", "오케이 다시 보네.",
    "응? 빨리도 왔다.", "휙 다녀오는구나.",
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
        "어, 아침이다.", "음, 일어났네.", "오늘은 좀 어때?",
        "흠, 아침이 또 왔어.", "잘 잤어 너?",
    ),
    "lunch": (
        "점심 시간이네!", "점심 먹었어?", "오늘 점심 뭐였어?",
        "흠, 배 안 고파?", "어 점심이네.", "뭐 먹지 오늘?",
    ),
    "afternoon": (
        "안녕!", "오후엔 좀 노곤하지.", "흠, 오후네.",
        "왔어, 반가워.", "어 왔구나.", "음, 슬슬 졸린 시간.",
        "오후 한 잔 어때?",   # 따뜻한 음료 quirk
    ),
    "evening": (
        "어, 저녁이네.", "오늘 수고 많았어.", "퇴근했어?",
        "음, 어땠어 오늘?", "저녁 시간이다.", "슬슬 마무리할 시간.",
        "하늘 봤어? 색깔 좋잖아.",
    ),
    "late": (
        "아직 안 자네.", "이 시간에 보네.", "음, 늦었다.",
        "흠, 깨어있구나.", "조용한 시간이네 — 좋아.",   # 밤 좋아함 quirk
        "야밤이다 또.",
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


def _claude_situational(event_kind: str, ctx: StateContext, extra: dict | None = None) -> str:
    """행동 멘트는 템플릿 풀만 사용 — 항상 "" 반환(caller가 풀 fallback).

    비용 절감: 이전엔 매 reappear/closer/farther/face_greeting마다
    generate_situational(Claude)를 먼저 호출 → 거리 변화·재등장이 잦아 시간당
    수십~수백 호출(숨은 비용원)이었음. 풀(REAPPEAR/TIME_GREETINGS/GOT_* 등)이
    충분히 다양해 체감 차이 작음. Claude 멘트가 다시 필요하면 여기에 BEHAVIOR
    토글을 추가해 되살릴 수 있음.
    """
    return ""


def reappear_message(absence_sec: float, ctx: StateContext) -> str:
    """부재 시간 + 시간대 + 이름 조합. Claude가 있으면 매번 새로 생성, 없으면 풀."""
    if absence_sec >= 600:
        desc = f"사용자가 {int(absence_sec/60)}분 자리를 비웠다 돌아옴 (오랜만)"
    elif absence_sec >= 60:
        desc = f"사용자가 {int(absence_sec/60)}분 정도 자리를 비웠다 돌아옴"
    else:
        desc = f"사용자가 잠깐({int(absence_sec)}초) 자리를 비웠다 돌아옴"
    msg = _claude_situational(desc, ctx, extra={"absence_sec": int(absence_sec)})
    if msg:
        return msg
    # Fallback: 풀
    name_pre = _name_prefix(ctx)
    period = period_for()
    if absence_sec >= 600:
        pool = REAPPEAR_LONG
    elif absence_sec >= 60:
        pool = (TIME_GREETINGS.get(period, ()) + REAPPEAR_MEDIUM
                if random.random() < 0.5 else REAPPEAR_MEDIUM)
    else:
        pool = REAPPEAR_SHORT
    base = _pick_fresh(pool)
    return name_pre + base


def closer_message(ctx: StateContext | None = None) -> str:
    desc = "사용자가 로봇 쪽으로 가까이 다가옴"
    msg = _claude_situational(desc, ctx) if ctx else _claude_situational_noctx(desc)
    if msg:
        return msg
    return _pick_fresh(GOT_CLOSER)


def farther_message(ctx: StateContext | None = None) -> str:
    desc = "사용자가 로봇에서 멀어짐"
    msg = _claude_situational(desc, ctx) if ctx else _claude_situational_noctx(desc)
    if msg:
        return msg
    return _pick_fresh(GOT_FARTHER)


def _claude_situational_noctx(event_kind: str) -> str:
    """행동 멘트는 풀만 사용 — 항상 "" 반환 (비용 절감, _claude_situational 참조)."""
    return ""


def face_greeting_message(name: str) -> str:
    # 비용 절감 — 템플릿만 사용 (Claude generate_situational 호출 제거)
    template = _pick_fresh(FACE_GREETING_TEMPLATES)
    return template.format(name=name)


_WAVE_SHORT = (
    "안녕?", "어, 안녕!", "반가워!", "오, 손 흔들어줬네!",
    "안녕 안녕!", "헤이~", "와, 봐줘서 좋아.",
    "오, 너구나!", "보고 싶었어.",
    # 추임새 + 자연스러운 톤 추가
    "어!", "음, 안녕.", "오 왔어?", "그래그래 안녕.",
    "헤이, 잘 지냈어?", "오케이, 봤어.", "흠, 손 인사구나.",
    "어어 안녕.", "야, 안녕.", "오 손 보인다.",
)


def wave_back_message(ctx: StateContext) -> str:
    """손 흔들기 답례 — 풀에서 즉시.

    Claude API call(1~3초)은 wave 응답 즉시성을 깨므로 사용 X.
    풀 자체가 다양해서 반복감 적음.
    """
    period = period_for()
    name_pre = _name_prefix(ctx)
    if random.random() < 0.5:
        pool = TIME_GREETINGS.get(period, _WAVE_SHORT)
    else:
        pool = _WAVE_SHORT
    base = _pick_fresh(pool)
    return name_pre + base
