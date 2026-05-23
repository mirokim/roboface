"""Roboface robot 모드 진입점 — Pi 5 (헤드리스).

pygame 없음. LCD 렌더러는 추후 통합. 지금은 백그라운드 task만 모두 가동.

키보드 입력 없음. 상태는 로그로만 관측.
종료: Ctrl+C 또는 SIGTERM.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from datetime import datetime

from src.audio.mic import Microphone, MicCaptureError
from src.brain import conversation_templates, memory, stats as robot_stats
from src.brain.agent import RobotAgent
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.config import AUDIO_INPUT_DEVICE, is_robot
from src.face.expressions import HAPPY, NEUTRAL, SURPRISED
from src.face.renderer import FaceState
from src.integrations.thinktank.offline_queue import run_flusher as run_queue_flusher
from src.motion import poses
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks import behavior_speaker, command_executor, journal_writer, schedule_extractor
from src.tasks.activity_monitor import run_activity_monitor
from src.tasks.ambient_listener import AmbientListener
from src.tasks.audio_reactive import run_audio_reactive
from src.tasks.eye_tracker import run_eye_tracker
from src.tasks.head_tracker import run_head_tracker
from src.tasks.idle_animation import run_ambient_motion, run_idle_gaze
from src.tasks.mood_drift import run_mood_drift
from src.tasks.posture_monitor import PostureMonitor
from src.tasks.proactive_speaker import run_loop as run_proactive
from src.tasks.reactive_face import flash_expression
from src.tasks.daily_recap import run_daily_recap
from src.tasks.db_cleanup import run_db_cleanup
from src.tasks.stats_tick import run_stats_tick
from src.web.server import run_web_server
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
    posture = PostureMonitor(
        keypoints_provider=lambda: perception.last_pose_keypoints,
        perception=perception,
    )
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
        asyncio.create_task(run_stats_tick(ctx), name="stats_tick"),
        asyncio.create_task(run_daily_recap(face, ctx), name="daily_recap"),
        asyncio.create_task(run_db_cleanup(), name="db_cleanup"),
        # Web UI — WEB_UI_PASSWORD 있을 때만 활성. 0.0.0.0:8080 (LAN).
        asyncio.create_task(
            run_web_server(face, ctx, perception), name="web",
        ),
        asyncio.create_task(work_tracker.run(ctx), name="work_tracker"),
        asyncio.create_task(posture.run(ctx, face), name="posture"),
        asyncio.create_task(run_activity_monitor(perception), name="activity"),
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
        # Claude tool-use 자율 에이전트 — 15초마다 결정 (API 키 있을 때만)
        asyncio.create_task(
            RobotAgent(
                face, ctx, perception, servos=servos,
                get_session_id=lambda: work_tracker.current_session_id,
            ).run(),
            name="agent",
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
        # 외부 명령 큐 — scripts/robot_cli.py에서 INSERT한 명령 처리.
        # gesture 명령은 sensors.events에 그대로 emit해 vision 우회 가능.
        asyncio.create_task(
            command_executor.run(
                face, ctx, servos=servos,
                emit_event=lambda ev: sensors.events.append(ev),
            ),
            name="command_executor",
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
                # 보수적으로 ~15 FPS (SPI ~38ms + sleep 30ms).
                await asyncio.sleep(0.03)
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
        # last_user_seen_at이 None이면 부팅 후 첫 등장 — "오랜만" 발동되면 어색.
        # 보수적으로 60초 부재로 취급 (짧은 인사 풀에 매핑됨)
        first_time = ctx.last_user_seen_at is None
        absence_sec = (now - ctx.last_user_seen_at) if not first_time else 60.0
        ctx.user_present = True
        ctx.last_user_seen_at = now
        # 오늘 처음 본 거면 first_seen_today_at 기록 (자정 지나면 초기화)
        today_str = datetime.now().strftime("%Y-%m-%d")
        if ctx.first_seen_today_date != today_str:
            ctx.first_seen_today_at = now
            ctx.first_seen_today_date = today_str
        if ctx.state == State.IDLE:
            ctx.transition(State.WATCHING, face)
        flash_expression(face, SURPRISED, 0.45)
        memory.log_user(
            f"(등장 — 부재 {int(absence_sec)}초"
            f"{', 부팅 직후' if first_time else ''})",
            kind="presence_new",
        )
        robot_stats.on_event("presence_new")
        if absence_sec > 30 or first_time:
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
        memory.log_user("(자리 비움)", kind="presence_left")
    elif ev.type == SensorEventType.ENV_TEMP:
        temp = ev.data.get("value")
        if temp is not None:
            memory.log_env(temp, 0.0)
            if perception is not None:
                perception.temperature_c = float(temp)
    elif ev.type == SensorEventType.GESTURE_WAVE:
        log.info("👋 wave 응답 시작")
        memory.log_user("(손 흔듦)", kind="gesture_wave")
        robot_stats.on_event("wave")
        asyncio.create_task(_wave_back(ctx, face, servos))
    elif ev.type == SensorEventType.GESTURE_HANDS_UP:
        log.info("🙌 hands up 응답 시작")
        memory.log_user("(양손 만세)", kind="gesture_hands_up")
        robot_stats.on_event("hands_up")
        asyncio.create_task(_hands_up_back(ctx, face, servos))
    elif ev.type == SensorEventType.GESTURE_HEAD_NOD:
        log.info("👍 head nod 응답 시작")
        memory.log_user("(끄덕임 — yes)", kind="gesture_nod")
        robot_stats.on_event("nod")
        asyncio.create_task(_simple_reply(ctx, face, "nod"))
    elif ev.type == SensorEventType.GESTURE_HEAD_SHAKE:
        log.info("🙅 head shake 응답 시작")
        memory.log_user("(도리도리 — no)", kind="gesture_shake")
        asyncio.create_task(_simple_reply(ctx, face, "shake"))
    elif ev.type == SensorEventType.GAZE_AT_ME:
        log.info("👀 정면 응시 응답 시작")
        memory.log_user("(사용자가 나를 쳐다봄)", kind="gaze_at_me")
        robot_stats.on_event("gaze")
        asyncio.create_task(_simple_reply(ctx, face, "gaze"))
    # MediaPipe Hands 기반 손 제스처 — 카테고리별 단순 reply + stat
    elif ev.type == SensorEventType.HAND_THUMB_UP:
        memory.log_user("(엄지척 👍)", kind="hand_thumb_up")
        robot_stats.on_event("thumb_up")
        asyncio.create_task(_simple_reply(ctx, face, "thumb_up"))
    elif ev.type == SensorEventType.HAND_THUMB_DOWN:
        memory.log_user("(엄지 다운 👎)", kind="hand_thumb_down")
        robot_stats.on_event("thumb_down")
        asyncio.create_task(_simple_reply(ctx, face, "thumb_down"))
    elif ev.type == SensorEventType.HAND_VICTORY:
        memory.log_user("(V사인 ✌️)", kind="hand_victory")
        robot_stats.on_event("victory")
        asyncio.create_task(_simple_reply(ctx, face, "victory"))
    elif ev.type == SensorEventType.HAND_OPEN_PALM:
        memory.log_user("(손바닥 🖐️)", kind="hand_open_palm")
        asyncio.create_task(_simple_reply(ctx, face, "open_palm"))
    elif ev.type == SensorEventType.HAND_FIST:
        memory.log_user("(주먹 👊)", kind="hand_fist")
        asyncio.create_task(_simple_reply(ctx, face, "fist"))
    elif ev.type == SensorEventType.HAND_POINTING:
        memory.log_user("(검지 ☝️)", kind="hand_pointing")
        asyncio.create_task(_simple_reply(ctx, face, "pointing"))
    elif ev.type == SensorEventType.HAND_ILOVEYOU:
        memory.log_user("(사랑해 🤟)", kind="hand_iloveyou")
        robot_stats.on_event("iloveyou")
        asyncio.create_task(_simple_reply(ctx, face, "iloveyou"))


async def _simple_reply(
    ctx: StateContext, face: FaceState, gesture_kind: str,
) -> None:
    """짧은 발화만 — 표정/머리는 그대로. 끄덕임/도리도리 응답용.

    gesture_kind: conversation_templates.GESTURE_POOLS 의 키.
    """
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    loop = asyncio.get_running_loop()
    msg = await loop.run_in_executor(
        None, lambda: conversation_templates.pick(gesture_kind, ctx),
    )
    # say() 단일 통로 — quiet hours / cooldown 자동 적용.
    # gesture류는 _GREETING_KINDS 밖이라 5분 인사 cooldown 영향 X.
    behavior_speaker.say(
        face, ctx, msg,
        kind=f"gesture_{gesture_kind}",
        cooldown_sec=15.0,
    )


async def _hands_up_back(ctx: StateContext, face: FaceState, servos) -> None:
    """양손 만세 응답 — STARSTRUCK + 짧은 댄스 + 신난 멘트.

    say() 단일 통로 — wave_back과 동일 패턴.
    """
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    from src.face.expressions import STARSTRUCK
    loop = asyncio.get_running_loop()
    msg = await loop.run_in_executor(
        None, lambda: conversation_templates.pick("hands_up", ctx),
    )
    # say() 먼저 호출 — busy_state(GREETING)이면 자기 자신을 막아버림.
    # 성공 시 그때서야 GREETING 전이.
    speech_task = behavior_speaker.say(
        face, ctx, msg,
        kind="hands_up_reply",
        cooldown_sec=3.0,
        expression=STARSTRUCK,
        bypass_quiet=True,   # 사용자 능동 제스처 — 새벽이라도 응답
    )
    if speech_task is None:
        flash_expression(face, STARSTRUCK, 0.6)
        return
    ctx.transition(State.GREETING, face)
    await asyncio.sleep(0)
    try:
        if servos is not None:
            async with motion_busy_scope(ctx):
                await poses.dance(servos, face, bpm=110, beats=4)
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
    """손 흔들기에 대한 응답 — HAPPY 표정 + 짧은 댄스 + 인사 멘트.

    behavior_speaker.say()를 단일 통로로 사용 — quiet hours / cooldown 통일 적용.
    사용자가 의도적으로 손 흔든 거라 _GREETING_KINDS 5분 cooldown은 적용 X.
    kind별 짧은 cooldown(10초)만.
    """
    if ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
        return
    # Claude로 멘트 생성 (sync — executor로)
    loop = asyncio.get_running_loop()
    greeting = await loop.run_in_executor(
        None, behavior_speaker.wave_back_message, ctx,
    )
    # say() 먼저 — busy_state(GREETING) gate가 자기 자신을 막아버리므로
    # state 전이는 say() 성공 후에. (이전 버전이 이 순서가 뒤바뀌어 있어 wave가
    # 매번 응답 없이 즉시 종료됨 — 실제 발화/댄스 한 번도 실행 X.)
    speech_task = behavior_speaker.say(
        face, ctx, greeting,
        kind="wave_reply",
        cooldown_sec=3.0,     # 짧게 — 연속 wave에 빠른 반응
        expression=HAPPY,
        bypass_quiet=True,    # 사용자 능동 제스처 — 새벽이라도 응답
    )
    if speech_task is None:
        # cooldown으로 막혔어도 무시 신호는 X — 짧은 표정 flash로 "봤어" 인디케이션
        flash_expression(face, HAPPY, 0.6)
        return
    ctx.transition(State.GREETING, face)
    await asyncio.sleep(0)  # bubble 즉시 노출
    try:
        if servos is not None:
            async with motion_busy_scope(ctx):
                await poses.dance(servos, face, bpm=100, beats=4)
        else:
            await asyncio.sleep(1.5)
        try:
            await speech_task
        except Exception as e:
            log.debug(f"wave 멘트 에러: {e}")
    finally:
        if not speech_task.done():
            speech_task.cancel()
        ctx.transition(
            State.WATCHING if ctx.user_present else State.IDLE, face,
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
