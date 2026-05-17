"""Roboface 메인 진입점 — simulator 또는 robot 모드.

모든 핵심 task가 백그라운드로 동시에 실행:
- sensor manager (mmWave/DHT22 폴링)
- idle gaze animation
- work tracker (작업시간 추적)
- posture monitor (자세 감시)
- ambient listener (주변 청취 → 일정/저널 핸들러)
- schedule sync (ThinkTank Calendar)
- offline queue flusher
- proactive speaker (트리거 → LLM 멘트 → 발화)

키:
  1~9, 0, -, =, c, p, h, d, t, y, u, w  표정
  SPACE  mmWave 사용자 등장 시뮬
  B      강제 깜빡임
  M      가짜 발화
  J      ThinkTank PoC
  Q/ESC  종료
"""

from __future__ import annotations

import asyncio
import sys
import time

import pygame

from src.audio.fake_tts import speak as fake_speak
from src.brain import memory
from src.brain.state_machine import State, StateContext
from src.config import is_simulator
from src.face.expressions import (
    ANGRY, CONTENT, CURIOUS, DIZZY, EXCITED, FOCUSED, HAPPY, LOVE, NEUTRAL,
    PROUD, SAD, SHOCKED, SLEEPY, STARSTRUCK, SURPRISED, THINKING, WINK,
    WINK_R, WORRIED, YAWN,
)
from src.face.eyes import trigger_blink
from src.face.renderer import FaceState, PygameRenderer
from src.integrations.thinktank.offline_queue import run_flusher as run_queue_flusher
from src.integrations.thinktank.poc import run_poc as thinktank_poc
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks import command_executor, journal_writer, schedule_extractor
from src.tasks.ambient_listener import AmbientListener
from src.tasks.idle_animation import run_idle_gaze
from src.tasks.posture_monitor import PostureMonitor
from src.tasks.proactive_speaker import run_loop as run_proactive
from src.tasks.work_tracker import WorkTracker
from src.utils.logger import get_logger

log = get_logger("main")


KEY_EXPRESSIONS = {
    pygame.K_1: NEUTRAL, pygame.K_2: HAPPY, pygame.K_3: EXCITED,
    pygame.K_4: SLEEPY, pygame.K_5: SURPRISED, pygame.K_6: WORRIED,
    pygame.K_7: FOCUSED, pygame.K_8: LOVE, pygame.K_9: THINKING,
    pygame.K_0: WINK, pygame.K_MINUS: SAD, pygame.K_EQUALS: ANGRY,
    pygame.K_c: CONTENT, pygame.K_p: PROUD, pygame.K_h: SHOCKED,
    pygame.K_d: DIZZY, pygame.K_t: STARSTRUCK, pygame.K_y: YAWN,
    pygame.K_u: CURIOUS, pygame.K_w: WINK_R,
}


async def run_simulator() -> None:
    log.info("=== Roboface Simulator 시작 ===")
    log.info(
        "키: 1-9/0/-/= 표정 / 알파벳 추가표정 / "
        "SPACE 사용자등장 / B 깜빡임 / M 발화 / J PoC / ESC 종료",
    )

    memory.init_db()

    renderer = PygameRenderer(scale=3)
    face = FaceState(expression=NEUTRAL)
    ctx = StateContext()

    sensors = SensorManager()
    sensors.register_default()

    work_tracker = WorkTracker()
    posture = PostureMonitor()
    ambient = AmbientListener()
    ambient.add_handler(schedule_extractor.handle_transcript)
    ambient.add_handler(journal_writer.handle_transcript)

    servos = create_servos()  # mock 서보 (실제 모드면 PCA9685)

    # 백그라운드 태스크들
    bg_tasks = [
        asyncio.create_task(sensors.run(), name="sensors"),
        asyncio.create_task(run_idle_gaze(face), name="idle_gaze"),
        asyncio.create_task(work_tracker.run(ctx), name="work_tracker"),
        asyncio.create_task(posture.run(ctx, face), name="posture"),
        asyncio.create_task(ambient.run(), name="ambient"),
        asyncio.create_task(schedule_extractor.sync_pending_to_thinktank(),
                            name="schedule_sync"),
        asyncio.create_task(run_queue_flusher(), name="queue_flusher"),
        asyncio.create_task(
            run_proactive(ctx, face, lambda: work_tracker.current_session_id),
            name="proactive",
        ),
        asyncio.create_task(
            command_executor.run(
                face, ctx, servos=servos,
                emit_event=lambda ev: sensors.events.append(ev),
            ),
            name="command_executor",
        ),
    ]

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = _handle_key(event, face, ctx, sensors) and running

            for ev in sensors.drain_events():
                _handle_sensor_event(ev, ctx, face, work_tracker)

            renderer.render(face)
            await asyncio.sleep(0)
    finally:
        log.info("종료 중...")
        for t in bg_tasks:
            t.cancel()
        for t in bg_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
        sensors.close()
        renderer.close()


def _handle_key(
    event, face: FaceState, ctx: StateContext, sensors: SensorManager,
) -> bool:
    """키 입력 처리. False 반환 시 종료."""
    if event.key in (pygame.K_ESCAPE, pygame.K_q):
        return False
    if event.key == pygame.K_b:
        trigger_blink(face.eye_state, time.time())
    elif event.key == pygame.K_SPACE:
        for s in sensors.sensors:
            if hasattr(s, "trigger_arrival"):
                ev = s.trigger_arrival()
                sensors.events.append(ev)
    elif event.key == pygame.K_m:
        asyncio.create_task(fake_speak(
            face, "안녕하세요. 오늘 날씨가 좋네요. 잘 지내고 계신가요?",
        ))
    elif event.key == pygame.K_j:
        asyncio.create_task(_thinktank_poc_with_feedback(face))
    elif event.key in KEY_EXPRESSIONS:
        face.apply_expression(KEY_EXPRESSIONS[event.key])
        log.info(f"표정 → {face.expression.name}")
    return True


async def _thinktank_poc_with_feedback(face: FaceState) -> None:
    face.apply_expression(THINKING)
    result = await thinktank_poc()
    if result["journal_ok"]:
        face.apply_expression(LOVE)
        log.info("PoC 성공!")
    elif result["healthcheck"]:
        face.apply_expression(WORRIED)
        log.info(f"헬스체크 OK 했지만 저널 실패: {result.get('error')}")
    else:
        face.apply_expression(SAD)
        log.info(f"PoC 실패: {result.get('error')}")


def _handle_sensor_event(
    ev,
    ctx: StateContext,
    face: FaceState,
    work_tracker: WorkTracker,
) -> None:
    """센서 이벤트 → 상태 머신 + work tracker 반영."""
    work_tracker.on_event(ev, ctx)

    if ev.type == SensorEventType.PRESENCE_NEW:
        ctx.user_present = True
        ctx.last_user_seen_at = time.time()
        if ctx.state == State.IDLE:
            ctx.transition(State.WATCHING, face)
    elif ev.type == SensorEventType.PRESENCE_LEFT:
        ctx.user_present = False
        if ctx.state != State.IDLE:
            ctx.transition(State.IDLE, face)
    elif ev.type == SensorEventType.ENV_TEMP:
        memory.log_env(ev.data["value"], 0.0)


def main() -> None:
    if not is_simulator():
        log.error("이 진입점은 simulator 모드 전용입니다. "
                  "ROBOFACE_MODE=simulator로 실행하세요.")
        sys.exit(1)
    try:
        asyncio.run(run_simulator())
    except KeyboardInterrupt:
        log.info("사용자 중단")


if __name__ == "__main__":
    main()
