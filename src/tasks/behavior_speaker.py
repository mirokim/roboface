"""행동 관찰 멘트 — vision/sensor 이벤트 보고 자연스럽게 한 마디.

각 행동 종류별 멘트 풀 + 종류별 쿨다운. quiet hours 동안엔 모두 skip.
chitchat trigger보다 더 자주, 더 즉각적으로 발동 (이벤트 즉시 반응).
"""

from __future__ import annotations

import asyncio
import random
import time

from src.audio.fake_tts import speak as fake_speak
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
    - kind별 쿨다운 (이전 동일 kind 멘트가 cooldown_sec 안에 있으면 skip)
    - 전체 proactive 쿨다운은 무시 — 사용자 행동 즉각 반응이 목적
    """
    if not text:
        return False
    if _busy_state(ctx):
        return False
    if _is_quiet_hours():
        return False
    now = time.time()
    last = _LAST_AT.get(kind, 0.0)
    if now - last < cooldown_sec:
        return False
    _LAST_AT[kind] = now

    if expression is not None:
        face.apply_expression(expression)
    log.info(f"🗣️  [{kind}] {text}")
    asyncio.create_task(fake_speak(face, text))
    ctx.last_proactive_at = now
    return True


# ─── 행동별 멘트 풀 ───

REAPPEAR_SHORT = (
    "금방 왔네!",
    "어디 갔다 왔어?",
    "왔어왔어.",
    "다시 봐서 반가워.",
)

REAPPEAR_LONG = (
    "오랜만이야!",
    "한참 안 보였네, 잘 있었어?",
    "어, 돌아왔구나.",
    "오랜만에 보네.",
)

GOT_CLOSER = (
    "어, 가까이 왔네?",
    "뭐 보여줄 거 있어?",
    "응? 왜?",
    "더 가까이서 보고 싶었구나.",
)

GOT_FARTHER = (
    "어디 가?",
    "왜 멀어져?",
    "어... 가지 마.",
    "멀어졌네.",
)

FACE_GREETING_TEMPLATES = (
    "{name}이다! 안녕!",
    "어, {name} 왔네.",
    "{name}, 반가워!",
    "{name} 오랜만이야.",
)


def reappear_message(absence_sec: float) -> str:
    """부재 시간에 따른 재등장 멘트."""
    if absence_sec < 60:
        return random.choice(REAPPEAR_SHORT)
    return random.choice(REAPPEAR_LONG)


def closer_message() -> str:
    return random.choice(GOT_CLOSER)


def farther_message() -> str:
    return random.choice(GOT_FARTHER)


def face_greeting_message(name: str) -> str:
    return random.choice(FACE_GREETING_TEMPLATES).format(name=name)
