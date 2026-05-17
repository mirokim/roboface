"""Roboface robot 모드 진입점 — Pi 5 (헤드리스).

pygame 없음. LCD 렌더러는 추후 통합. 지금은 백그라운드 task만 모두 가동.

키보드 입력 없음. 상태는 로그로만 관측.
종료: Ctrl+C 또는 SIGTERM.
"""

from __future__ import annotations

import asyncio
import random
import signal
import sys
import time

from src.audio.fake_tts import speak as fake_speak
from src.audio.mic import Microphone, MicCaptureError
from src.brain import memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.config import AUDIO_INPUT_DEVICE, is_robot
from src.face.expressions import HAPPY, NEUTRAL, SURPRISED
from src.face.renderer import FaceState
from src.integrations.thinktank.offline_queue import run_flusher as run_queue_flusher
from src.motion import poses
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks import behavior_speaker, journal_writer, schedule_extractor
from src.tasks.ambient_listener import AmbientListener
from src.tasks.audio_reactive import run_audio_reactive
from src.tasks.eye_tracker import run_eye_tracker
from src.tasks.head_tracker import run_head_tracker
from src.tasks.idle_animation import run_ambient_motion, run_idle_gaze
from src.tasks.mood_drift import run_mood_drift
from src.tasks.posture_monitor import PostureMonitor
from src.tasks.proactive_speaker import run_loop as run_proactive
from src.tasks.reactive_face import flash_expression
from src.tasks.thermal_state import run_thermal_state
from src.tasks.vision_task import run_vision
from src.tasks.voice_assistant import run_voice_assistant
from src.tasks.work_tracker import WorkTracker
from src.utils.logger import get_logger

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
        asyncio.create_task(run_thermal_state(face, perception), name="thermal"),
        asyncio.create_task(work_tracker.run(ctx), name="work_tracker"),
        asyncio.create_task(posture.run(ctx, face), name="posture"),
        asyncio.create_task(ambient.run(), name="ambient"),
        asyncio.create_task(schedule_extractor.sync_pending_to_thinktank(),
                            name="schedule_sync"),
        asyncio.create_task(run_queue_flusher(), name="queue_flusher"),
        asyncio.create_task(
            run_proactive(
                ctx, face, lambda: work_tracker.current_session_id,
                servos=servos, perception=perception,
            ),
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
                _handle_sensor_event(
                    ev, ctx, face, work_tracker, servos, perception,
                )
            if lcd is not None:
                # SPI 전송 동안 다른 async task 진행 (dance, fake_speak 등)
                await lcd.render_async(face)
                # ILI9341은 vsync 없어 너무 빨리 갱신하면 tearing/flicker.
                # ~20 FPS로 제한 (1/20 - 평균 SPI 30ms = 20ms 추가 sleep).
                await asyncio.sleep(0.02)
            else:
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
    perception: PerceptionState | None = None,
) -> None:
    work_tracker.on_event(ev, ctx)
    if ev.type == SensorEventType.PRESENCE_NEW:
        now = time.time()
        absence_sec = (
            now - ctx.last_user_seen_at if ctx.last_user_seen_at else 99999.0
        )
        ctx.user_present = True
        ctx.last_user_seen_at = now
        if ctx.state == State.IDLE:
            ctx.transition(State.WATCHING, face)
        # 사용자가 갑자기 들어옴 → 잠깐 놀란 표정 + 부재 시간 기반 멘트
        flash_expression(face, SURPRISED, 0.45)
        # 30초 미만이면 재등장 멘트 굳이 X (계속 있는 것과 구분 안 됨)
        if absence_sec > 30:
            behavior_speaker.say(
                face, ctx,
                behavior_speaker.reappear_message(absence_sec, ctx),
                kind="reappear",
                cooldown_sec=20.0,
            )
    elif ev.type == SensorEventType.PRESENCE_LEFT:
        ctx.user_present = False
        if ctx.state != State.IDLE:
            ctx.transition(State.IDLE, face)
    elif ev.type == SensorEventType.ENV_TEMP:
        temp = ev.data.get("value")
        if temp is not None:
            memory.log_env(temp, 0.0)
            if perception is not None:
                perception.temperature_c = float(temp)
    elif ev.type == SensorEventType.GESTURE_WAVE:
        log.info("👋 wave 응답 시작")
        asyncio.create_task(_wave_back(ctx, face, servos))
    elif ev.type == SensorEventType.GESTURE_HANDS_UP:
        log.info("🙌 hands up 응답 시작")
        asyncio.create_task(_hands_up_back(ctx, face, servos))
    elif ev.type == SensorEventType.GESTURE_HEAD_NOD:
        log.info("👍 head nod 응답 시작")
        asyncio.create_task(_simple_reply(ctx, face, _HEAD_NOD_REPLIES, "nod"))
    elif ev.type == SensorEventType.GESTURE_HEAD_SHAKE:
        log.info("🙅 head shake 응답 시작")
        asyncio.create_task(_simple_reply(ctx, face, _HEAD_SHAKE_REPLIES, "shake"))


_WAVE_GREETINGS = (
    "안녕?",
    "어, 안녕!",
    "반가워!",
    "오, 손 흔들어줬네!",
    "안녕 안녕!",
    "헤이~",
)

_HANDS_UP_REPLIES = (
    "와! 만세!",
    "야호!",
    "신난다!",
    "오, 뭐가 좋은 일 있어?",
    "축하해!",
)

_HEAD_NOD_REPLIES = (
    "응응.",
    "그래!",
    "오케이!",
    "알겠어.",
    "좋아.",
)

_HEAD_SHAKE_REPLIES = (
    "안돼?",
    "왜?",
    "음... 알겠어.",
    "아냐?",
    "그래, 그러지 말자.",
)


async def _simple_reply(
    ctx: StateContext, face: FaceState, replies: tuple[str, ...],
    kind: str = "gesture_reply",
) -> None:
    """짧은 발화만 — 표정/머리는 그대로. 끄덕임/도리도리 응답용."""
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    msg = random.choice(replies)
    log.info(f"🗣️  {msg}")
    asyncio.create_task(fake_speak(face, msg))
    memory.log_robot(msg, kind=kind)


async def _hands_up_back(ctx: StateContext, face: FaceState, servos) -> None:
    """양손 만세 응답 — STARSTRUCK + 짧은 댄스 + 신난 멘트."""
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    from src.face.expressions import STARSTRUCK
    face.apply_expression(STARSTRUCK)
    ctx.transition(State.GREETING, face)
    msg = random.choice(_HANDS_UP_REPLIES)
    log.info(f"🗣️  {msg}")
    ctx.last_greeting_at = time.time()
    memory.log_robot(msg, kind="hands_up_reply")
    speech_task = asyncio.create_task(fake_speak(face, msg))
    await asyncio.sleep(0)
    try:
        if servos is not None:
            await poses.dance(servos, face, bpm=150, beats=4)
        else:
            await asyncio.sleep(1.5)
        try:
            await speech_task
        except Exception as e:
            log.debug(f"hands_up 멘트 에러: {e}")
    finally:
        if not speech_task.done():
            speech_task.cancel()
        ctx.transition(
            State.WATCHING if ctx.user_present else State.IDLE, face,
        )


async def _wave_back(ctx: StateContext, face: FaceState, servos) -> None:
    """손 흔들기에 대한 응답 — HAPPY 표정 + 짧은 댄스 + 인사 멘트."""
    # 이미 다른 인터랙션 중이면 양보
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    face.apply_expression(HAPPY)
    ctx.transition(State.GREETING, face)
    greeting = behavior_speaker.wave_back_message(ctx)
    log.info(f"🗣️  {greeting}")
    ctx.last_greeting_at = time.time()
    memory.log_robot(greeting, kind="wave_reply")
    # fake_speak가 내부에서 face.show_speech 호출. 첫 await 대기 없이
    # 즉시 노출되도록 task 생성 직후 한 번 yield해서 task가 실행되게 함.
    speech_task = asyncio.create_task(fake_speak(face, greeting))
    await asyncio.sleep(0)
    try:
        if servos is not None:
            await poses.dance(servos, face, bpm=140, beats=4)
        else:
            await asyncio.sleep(1.5)
        # 멘트가 모션보다 길면 끝까지 기다림
        try:
            await speech_task
        except Exception as e:
            log.debug(f"wave 멘트 에러: {e}")
    finally:
        if not speech_task.done():
            speech_task.cancel()
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
