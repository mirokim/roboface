"""Roboface 메인 진입점 — simulator 또는 robot 모드.

simulator 모드:
- Pygame 창에 얼굴 표시
- Mock 센서로 이벤트 흐름 테스트
- 키보드 단축키:
    SPACE  : mmWave 사용자 등장 트리거
    1~9    : 표정 전환
    B      : 강제 깜빡임
    Q/ESC  : 종료
"""

from __future__ import annotations

import asyncio
import sys
import time

import pygame

from src.audio.fake_tts import speak as fake_speak
from src.brain import memory, triggers
from src.brain.state_machine import State, StateContext
from src.config import is_simulator
from src.face.expressions import (
    ANGRY, CONTENT, CURIOUS, DIZZY, EXCITED, FOCUSED, HAPPY, LOVE, NEUTRAL,
    PROUD, SAD, SHOCKED, SLEEPY, STARSTRUCK, SURPRISED, THINKING, WINK,
    WINK_R, WORRIED, YAWN,
)
from src.face.eyes import trigger_blink
from src.face.renderer import FaceState, PygameRenderer
from src.integrations.thinktank.poc import run_poc as thinktank_poc
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks.idle_animation import run_idle_gaze
from src.utils.logger import get_logger

log = get_logger("main")


# 키보드 → 표정 매핑
KEY_EXPRESSIONS = {
    # 숫자키 — 기본 12종
    pygame.K_1: NEUTRAL,
    pygame.K_2: HAPPY,
    pygame.K_3: EXCITED,
    pygame.K_4: SLEEPY,
    pygame.K_5: SURPRISED,
    pygame.K_6: WORRIED,
    pygame.K_7: FOCUSED,
    pygame.K_8: LOVE,
    pygame.K_9: THINKING,
    pygame.K_0: WINK,
    pygame.K_MINUS: SAD,
    pygame.K_EQUALS: ANGRY,
    # 알파벳 — 추가 8종
    pygame.K_c: CONTENT,
    pygame.K_p: PROUD,
    pygame.K_h: SHOCKED,    # sHocked
    pygame.K_d: DIZZY,
    pygame.K_t: STARSTRUCK, # sTar
    pygame.K_y: YAWN,
    pygame.K_u: CURIOUS,
    pygame.K_w: WINK_R,
}


async def run_simulator() -> None:
    log.info("=== Roboface Simulator 시작 ===")
    log.info("키: 1-9 표정 / 알파벳 추가표정 / SPACE 사용자등장 / B 깜빡임 / "
             "M 발화시뮬 / J ThinkTank PoC / ESC 종료")

    memory.init_db()

    renderer = PygameRenderer(scale=2)
    face = FaceState(expression=NEUTRAL)
    ctx = StateContext()

    sensors = SensorManager()
    sensors.register_default()
    sensor_task = asyncio.create_task(sensors.run())
    idle_task = asyncio.create_task(run_idle_gaze(face))
    bg_tasks = [sensor_task, idle_task]

    servos = create_servos()
    current_session_id: int | None = None
    last_trigger_check = 0.0

    running = True
    try:
        while running:
            # === 이벤트 ===
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_b:
                        trigger_blink(face.eye_state, time.time())
                    elif event.key == pygame.K_SPACE:
                        for s in sensors.sensors:
                            if hasattr(s, "trigger_arrival"):
                                ev = s.trigger_arrival()
                                sensors.events.append(ev)
                    elif event.key == pygame.K_m:
                        # M키: 가짜 발화 시뮬레이션
                        asyncio.create_task(fake_speak(
                            face, "안녕하세요. 오늘 날씨가 좋네요. 잘 지내고 계신가요?"
                        ))
                    elif event.key == pygame.K_j:
                        # J키: ThinkTank PoC (헬스체크 + 저널 POST)
                        async def _run_poc() -> None:
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
                        asyncio.create_task(_run_poc())
                    elif event.key in KEY_EXPRESSIONS:
                        face.apply_expression(KEY_EXPRESSIONS[event.key])
                        log.info(f"표정 → {face.expression.name}")

            # === 센서 이벤트 처리 ===
            for ev in sensors.drain_events():
                _handle_sensor_event(ev, ctx, face, current_session_id)
                # 작업 세션 관리
                if ev.type == SensorEventType.PRESENCE_NEW:
                    if current_session_id is None:
                        current_session_id = memory.start_work_session()
                        log.info(f"작업 세션 시작 #{current_session_id}")
                elif ev.type == SensorEventType.PRESENCE_LEFT:
                    if current_session_id is not None:
                        memory.end_work_session(current_session_id)
                        log.info(f"작업 세션 종료 #{current_session_id}")
                        current_session_id = None

            # === 트리거 평가 (1Hz) ===
            now = time.time()
            if now - last_trigger_check > 1.0:
                last_trigger_check = now
                for trig in triggers.evaluate_all(ctx, current_session_id):
                    log.info(f"트리거 발생: {trig.kind} (priority={trig.priority})")
                    if trig.kind == "greeting" and ctx.state != State.GREETING:
                        ctx.transition(State.GREETING, face)
                        face.apply_expression(HAPPY)
                    elif trig.kind.startswith("work_break"):
                        ctx.transition(State.ALERTING, face)
                    if trig.suggested_message:
                        log.info(f"  멘트: {trig.suggested_message}")
                    ctx.last_proactive_at = now
                    memory.log_proactive(trig.kind, trig.suggested_message or "")
                    break  # 한 사이클에 하나만

                # 상태 자동 복귀: GREETING → WATCHING 3초 후
                if ctx.state == State.GREETING and now - ctx.entered_at > 3.0:
                    ctx.transition(State.WATCHING, face)

            # === 렌더 ===
            renderer.render(face)
            await asyncio.sleep(0)  # 다른 코루틴에 양보

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


def _handle_sensor_event(
    ev,
    ctx: StateContext,
    face: FaceState,
    session_id: int | None,
) -> None:
    """센서 이벤트를 상태/표정에 반영."""
    if ev.type == SensorEventType.PRESENCE_NEW:
        ctx.user_present = True
        ctx.last_user_seen_at = time.time()
        if ctx.state == State.IDLE:
            ctx.transition(State.WATCHING, face)
            face.apply_expression(SURPRISED)
    elif ev.type == SensorEventType.PRESENCE_LEFT:
        ctx.user_present = False
        if ctx.state != State.IDLE:
            ctx.transition(State.IDLE, face)
    elif ev.type == SensorEventType.ENV_TEMP:
        memory.log_env(ev.data["value"], 0.0)  # 임시
    elif ev.type == SensorEventType.ENV_HUMIDITY:
        # 두 이벤트가 함께 옴 — 마지막 env_log를 update해도 되지만 단순화
        pass


def main() -> None:
    if not is_simulator():
        log.error("이 진입점은 simulator 모드 전용입니다. ROBOFACE_MODE=simulator로 실행하세요.")
        sys.exit(1)
    try:
        asyncio.run(run_simulator())
    except KeyboardInterrupt:
        log.info("사용자 중단")


if __name__ == "__main__":
    main()
