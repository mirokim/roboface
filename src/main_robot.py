"""Roboface robot 모드 진입점 — Pi 5 (헤드리스).

pygame 없음. LCD 렌더러는 추후 통합. 지금은 백그라운드 task만 모두 가동.

키보드 입력 없음. 상태는 로그로만 관측.
종료: Ctrl+C 또는 SIGTERM.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from src.audio.mic import Microphone, MicCaptureError
from src.brain import memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.config import AUDIO_INPUT_DEVICE, is_robot
from src.face.expressions import NEUTRAL
from src.face.renderer import FaceState
from src.integrations.thinktank.offline_queue import run_flusher as run_queue_flusher
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks import journal_writer, schedule_extractor
from src.tasks.ambient_listener import AmbientListener
from src.tasks.audio_reactive import run_audio_reactive
from src.tasks.eye_tracker import run_eye_tracker
from src.tasks.head_tracker import run_head_tracker
from src.tasks.idle_animation import run_ambient_motion, run_idle_gaze
from src.tasks.mood_drift import run_mood_drift
from src.tasks.posture_monitor import PostureMonitor
from src.tasks.proactive_speaker import run_loop as run_proactive
from src.tasks.reactive_face import flash_expression
from src.tasks.vision_task import run_vision
from src.tasks.voice_assistant import run_voice_assistant
from src.tasks.work_tracker import WorkTracker
from src.utils.logger import get_logger
import time

log = get_logger("main_robot")


async def run_robot() -> None:
    log.info("=== Roboface (robot mode) 시작 ===")

    memory.init_db()

    face = FaceState(expression=NEUTRAL)
    ctx = StateContext()

    # LCD 렌더러 시도 — 실패하면 헤드리스 모드
    lcd = None
    try:
        from src.face.lcd_renderer import LCDRenderer
        lcd = LCDRenderer()
    except Exception as e:
        log.warning(f"LCD 초기화 실패 (헤드리스 모드로 전환): {e}")
        lcd = None

    sensors = SensorManager()
    sensors.register_default()

    perception = PerceptionState()
    servos = create_servos()
    # 시작 시 머리 중앙 정렬
    try:
        servos.home()
    except Exception as e:
        log.warning(f"서보 home 실패: {e}")

    work_tracker = WorkTracker()
    posture = PostureMonitor()
    ambient = AmbientListener()
    ambient.add_handler(schedule_extractor.handle_transcript)
    ambient.add_handler(journal_writer.handle_transcript)

    # 공유 마이크 — voice_assistant + audio_reactive 둘 다 subscribe
    shared_mic: Microphone | None = None
    try:
        shared_mic = Microphone(device=AUDIO_INPUT_DEVICE)
        shared_mic.__enter__()  # 시작 (callback 활성)
    except MicCaptureError as e:
        log.warning(f"마이크 사용 불가 — 음성/박수/음악 기능 비활성: {e}")
        shared_mic = None

    bg_tasks = [
        asyncio.create_task(sensors.run(), name="sensors"),
        asyncio.create_task(run_idle_gaze(face, perception), name="idle_gaze"),
        asyncio.create_task(run_eye_tracker(face, perception, ctx), name="eye_tracker"),
        asyncio.create_task(run_ambient_motion(servos, ctx), name="ambient_motion"),
        asyncio.create_task(run_mood_drift(face, ctx), name="mood_drift"),
        asyncio.create_task(work_tracker.run(ctx), name="work_tracker"),
        asyncio.create_task(posture.run(ctx, face), name="posture"),
        asyncio.create_task(ambient.run(), name="ambient"),
        asyncio.create_task(schedule_extractor.sync_pending_to_thinktank(),
                            name="schedule_sync"),
        asyncio.create_task(run_queue_flusher(), name="queue_flusher"),
        asyncio.create_task(
            run_proactive(ctx, face, lambda: work_tracker.current_session_id, servos=servos),
            name="proactive",
        ),
        # AI Camera person detection + perception + 표정 거울 + 얼굴 인식
        asyncio.create_task(
            run_vision(lambda ev: sensors.events.append(ev), perception,
                       face=face, ctx=ctx),
            name="vision",
        ),
        # 얼굴 추적 — 서보로 머리 회전
        asyncio.create_task(
            run_head_tracker(servos, perception, ctx),
            name="head_tracker",
        ),
        # 음성 어시스턴트 — wake word → STT → Claude → TTS
        asyncio.create_task(
            run_voice_assistant(ctx, face, servos=servos, mic=shared_mic),
            name="voice_assistant",
        ),
    ]
    if shared_mic is not None:
        # 박수 + 음악 비트 → 표정/모션 반응
        bg_tasks.append(asyncio.create_task(
            run_audio_reactive(shared_mic, face, ctx,
                               perception=perception, servos=servos),
            name="audio_reactive",
        ))

    # 시그널 핸들러로 graceful shutdown
    stop_event = asyncio.Event()

    def _request_stop(_sig=None, _frm=None) -> None:
        log.info("종료 시그널 수신")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            # Windows 등 일부 환경
            signal.signal(sig, _request_stop)

    try:
        # 메인 루프: 센서 이벤트 처리 + LCD 렌더 + 상태 머신
        while not stop_event.is_set():
            for ev in sensors.drain_events():
                _handle_sensor_event(ev, ctx, face, work_tracker, servos)
            if lcd is not None:
                lcd.render(face)
            await asyncio.sleep(0.033)  # ~30 FPS
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
        if lcd is not None:
            lcd.close()
        if shared_mic is not None:
            try:
                shared_mic.__exit__(None, None, None)
            except Exception:
                pass
        log.info("종료 완료")


def _handle_sensor_event(
    ev,
    ctx: StateContext,
    face: FaceState,
    work_tracker: WorkTracker,
    servos=None,
) -> None:
    work_tracker.on_event(ev, ctx)
    if ev.type == SensorEventType.PRESENCE_NEW:
        ctx.user_present = True
        ctx.last_user_seen_at = time.time()
        if ctx.state == State.IDLE:
            ctx.transition(State.WATCHING, face)
        # 사용자가 갑자기 들어옴 → 잠깐 놀란 표정
        from src.face.expressions import SURPRISED
        flash_expression(face, SURPRISED, 0.45)
    elif ev.type == SensorEventType.PRESENCE_LEFT:
        ctx.user_present = False
        if ctx.state != State.IDLE:
            ctx.transition(State.IDLE, face)
    elif ev.type == SensorEventType.ENV_TEMP:
        memory.log_env(ev.data["value"], 0.0)
    elif ev.type == SensorEventType.GESTURE_WAVE:
        log.info("👋 wave 응답 시작")
        asyncio.create_task(_wave_back(ctx, face, servos))


async def _wave_back(ctx: StateContext, face: FaceState, servos) -> None:
    """손 흔들기에 대한 응답 — HAPPY 표정 + 짧은 댄스로 답례."""
    # 이미 다른 인터랙션 중이면 양보
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    prev_state = ctx.state
    from src.face.expressions import HAPPY
    face.apply_expression(HAPPY)
    ctx.transition(State.GREETING, face)
    try:
        if servos is not None:
            from src.motion import poses
            await poses.dance(servos, face, bpm=140, beats=4)
        else:
            await asyncio.sleep(1.5)
    finally:
        # 사용자 보이면 watching, 아니면 idle
        ctx.transition(
            State.WATCHING if ctx.user_present else State.IDLE,
            face,
        )


def main() -> None:
    if not is_robot():
        log.error("이 진입점은 robot 모드 전용입니다. "
                  "ROBOFACE_MODE=robot으로 실행하세요.")
        sys.exit(1)
    try:
        asyncio.run(run_robot())
    except KeyboardInterrupt:
        log.info("사용자 중단")


if __name__ == "__main__":
    main()
