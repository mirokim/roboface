"""Roboface robot 모드 진입점 — Pi 5 (헤드리스).

pygame 없음. LCD 렌더러는 추후 통합. 지금은 백그라운드 task만 모두 가동.

키보드 입력 없음. 상태는 로그로만 관측.
종료: Ctrl+C 또는 SIGTERM.
"""

from __future__ import annotations

import asyncio
import signal
import sys

from src.brain import memory
from src.brain.state_machine import State, StateContext
from src.config import is_robot
from src.face.expressions import NEUTRAL
from src.face.renderer import FaceState   # state container만 사용
from src.integrations.thinktank.offline_queue import run_flusher as run_queue_flusher
from src.motion.servos import create_controller as create_servos
from src.sensors.base import SensorEventType
from src.sensors.manager import SensorManager
from src.tasks import journal_writer, schedule_extractor
from src.tasks.ambient_listener import AmbientListener
from src.tasks.idle_animation import run_idle_gaze
from src.tasks.posture_monitor import PostureMonitor
from src.tasks.proactive_speaker import run_loop as run_proactive
from src.tasks.work_tracker import WorkTracker
from src.utils.logger import get_logger
import time

log = get_logger("main_robot")


async def run_robot() -> None:
    log.info("=== Roboface (robot mode) 시작 ===")
    log.info("LCD 렌더러 미통합 — 표정 상태는 로그로만 출력")

    memory.init_db()

    face = FaceState(expression=NEUTRAL)
    ctx = StateContext()

    sensors = SensorManager()
    sensors.register_default()

    work_tracker = WorkTracker()
    posture = PostureMonitor()
    ambient = AmbientListener()
    ambient.add_handler(schedule_extractor.handle_transcript)
    ambient.add_handler(journal_writer.handle_transcript)

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
    ]

    _ = create_servos()  # PCA9685 (실패 시 Mock 폴백)

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
        # 메인 루프: 센서 이벤트 처리 + 상태 머신 운영
        while not stop_event.is_set():
            for ev in sensors.drain_events():
                _handle_sensor_event(ev, ctx, face, work_tracker)
            await asyncio.sleep(0.1)
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
        log.info("종료 완료")


def _handle_sensor_event(
    ev,
    ctx: StateContext,
    face: FaceState,
    work_tracker: WorkTracker,
) -> None:
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
