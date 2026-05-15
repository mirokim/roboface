"""트리거 평가기 → LLM 멘트 생성 → fake_speak (또는 실제 TTS).

상태 머신과 통합되어, 능동 개입 사이클을 자동화.
"""

from __future__ import annotations

import asyncio
import time

from src.audio.fake_tts import speak as fake_speak
from src.brain import conversation, memory
from src.brain.state_machine import State, StateContext
from src.brain.triggers import ProactiveTrigger, evaluate_all
from src.face.expressions import HAPPY, NEUTRAL, WORRIED
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.utils.logger import get_logger

log = get_logger("proactive")

# 트리거 종류별 표정 매핑
_TRIGGER_EXPR = {
    "greeting": HAPPY,
    "work_break_warn": NEUTRAL,
    "work_break_strong": WORRIED,
    "work_break_alarm": WORRIED,
    "long_silence": NEUTRAL,
}


async def fire_trigger(
    trig: ProactiveTrigger,
    ctx: StateContext,
    face: FaceState,
    servos: ServoController | None = None,
) -> None:
    """단일 트리거 처리: 표정 변경 + 머리 동작 + LLM 멘트 + 발화."""
    log.info(f"트리거 처리: {trig.kind} (priority={trig.priority})")

    # 표정
    expr = _TRIGGER_EXPR.get(trig.kind, NEUTRAL)
    face.apply_expression(expr)
    if trig.kind.startswith("work_break"):
        ctx.transition(State.ALERTING, face)

    # 머리 동작 (트리거별 — 발화와 병렬로)
    motion_task: asyncio.Task | None = None
    if servos is not None:
        if trig.kind == "greeting":
            ctx.transition(State.GREETING, face)
            motion_task = asyncio.create_task(poses.greeting(servos))
        elif trig.kind.startswith("work_break"):
            motion_task = asyncio.create_task(poses.shake(servos, times=1))

    # 멘트 결정
    if trig.suggested_message:
        message = trig.suggested_message
    else:
        message = conversation.generate_proactive_message(trig.kind, trig.context)
    if not message:
        log.debug(f"{trig.kind}: 빈 멘트, skip")
        if motion_task is not None:
            await motion_task
        return

    log.info(f"🗣️  {message}")

    # 발화 시뮬레이션 + 상태
    ctx.transition(State.TALKING, face)
    await fake_speak(face, message)

    # 머리 동작 완료 대기
    if motion_task is not None:
        try:
            await motion_task
        except Exception as e:
            log.debug(f"motion task 에러: {e}")

    memory.log_proactive(trig.kind, message)
    ctx.last_proactive_at = time.time()

    # 잠시 후 복귀
    await asyncio.sleep(0.5)
    if ctx.user_present:
        ctx.transition(State.WATCHING, face)
    else:
        ctx.transition(State.IDLE, face)


async def run_loop(
    ctx: StateContext,
    face: FaceState,
    get_session_id,
    servos: ServoController | None = None,
) -> None:
    """1초마다 트리거 평가."""
    while True:
        await asyncio.sleep(1.0)
        if ctx.state in (State.TALKING, State.LISTENING):
            continue
        triggers_list = evaluate_all(ctx, current_session_id=get_session_id())
        if not triggers_list:
            continue
        await fire_trigger(triggers_list[0], ctx, face, servos=servos)
