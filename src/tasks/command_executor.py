"""외부 명령 큐 실행기.

scripts/robot_cli.py가 SQLite command_queue에 INSERT한 명령을 메인 프로세스가
1초마다 폴링해 실행. 명령은 짧고 즉시 적용되는 동작만 허용.

지원 명령:
- speak {text, expression?}        — fake_speak로 발화 (말풍선 + 입 모양)
- expression {name}                — 표정만 변경
- dance {beats?, bpm?}             — 짧은 댄스 (서보 있을 때만)
- pose {kind}                      — nod / shake / greeting / tilt_curious
- transition {state}               — 상태 머신 강제 전이
- blink                            — 즉시 깜빡임
- status                           — 현재 상태 반환 (result 컬럼)

마이크 / API 키 없어도 동작 — 모든 명령은 face/servo만 사용.
"""

from __future__ import annotations

import asyncio
import json
import time

from collections.abc import Callable

from src.audio.fake_tts import speak as fake_speak
from src.brain import memory
from src.brain.state_machine import State, StateContext
from src.config import BEHAVIOR
from src.face import expressions as expr
from src.face.eyes import trigger_blink
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.sensors.base import SensorEvent, SensorEventType
from src.utils.logger import get_logger

log = get_logger("command_executor")


_POSE_MAP = {
    "nod": poses.nod,
    "shake": poses.shake,
    "greeting": poses.greeting,
    "tilt_curious": poses.tilt_curious,
    "look_around": poses.look_around,
}


# CLI `gesture <kind>` → SensorEventType 매핑.
# vision 없이도 downstream(메모리/표정/멘트)이 잘 도는지 검증 가능.
_GESTURE_MAP = {
    "wave": SensorEventType.GESTURE_WAVE,
    "hands_up": SensorEventType.GESTURE_HANDS_UP,
    "nod": SensorEventType.GESTURE_HEAD_NOD,
    "shake": SensorEventType.GESTURE_HEAD_SHAKE,
    "gaze": SensorEventType.GAZE_AT_ME,
    "presence_new": SensorEventType.PRESENCE_NEW,
    "presence_left": SensorEventType.PRESENCE_LEFT,
}


async def _execute(
    cmd: str,
    args: dict,
    face: FaceState,
    ctx: StateContext,
    servos: ServoController | None,
    emit_event: Callable[[SensorEvent], None] | None = None,
) -> str:
    """단일 명령 실행. 성공 시 result 문자열 반환, 실패 시 raise."""
    if cmd == "speak":
        text = (args.get("text") or "").strip()
        if not text:
            raise ValueError("text 비어있음")
        expr_name = args.get("expression")
        if expr_name:
            ex = expr.EXPRESSIONS_BY_NAME.get(expr_name.lower())
            if ex is not None:
                face.apply_expression(ex)
        asyncio.create_task(fake_speak(face, text))
        try:
            memory.log_robot(text, kind="cli_speak")
        except Exception:
            pass
        return f"speak: {text[:40]}"

    if cmd == "expression":
        name = (args.get("name") or "").lower()
        ex = expr.EXPRESSIONS_BY_NAME.get(name)
        if ex is None:
            raise ValueError(f"알 수 없는 표정: {name}")
        face.apply_expression(ex)
        return f"expression: {name}"

    if cmd == "dance":
        if servos is None:
            raise RuntimeError("서보 없음")
        beats = int(args.get("beats", 4))
        bpm = int(args.get("bpm", 120))
        asyncio.create_task(poses.dance(servos, face, bpm=bpm, beats=beats))
        return f"dance: {beats}@{bpm}bpm"

    if cmd == "pose":
        if servos is None:
            raise RuntimeError("서보 없음")
        kind = args.get("kind", "")
        fn = _POSE_MAP.get(kind)
        if fn is None:
            raise ValueError(f"알 수 없는 pose: {kind}. 가능: {list(_POSE_MAP)}")
        asyncio.create_task(fn(servos))
        return f"pose: {kind}"

    if cmd == "transition":
        sname = (args.get("state") or "").upper()
        try:
            new_state = State[sname]
        except KeyError as e:
            raise ValueError(
                f"알 수 없는 state: {sname}. 가능: {[s.name for s in State]}",
            ) from e
        ctx.transition(new_state, face)
        return f"transition: {sname}"

    if cmd == "blink":
        trigger_blink(face.eye_state, time.time())
        return "blink"

    if cmd == "status":
        return json.dumps({
            "state": ctx.state.value,
            "user_present": ctx.user_present,
            "expression": face.expression.name,
            "user_name": ctx.user_name,
            "brightness": face.brightness,
        }, ensure_ascii=False)

    if cmd == "gesture":
        kind = args.get("kind", "")
        ev_type = _GESTURE_MAP.get(kind)
        if ev_type is None:
            raise ValueError(
                f"알 수 없는 gesture: {kind}. 가능: {list(_GESTURE_MAP)}",
            )
        if emit_event is None:
            raise RuntimeError("emit_event 콜백 미주입 (main에서 등록 필요)")
        emit_event(SensorEvent(type=ev_type, data={"source": "cli"}))
        return f"gesture: {kind} → {ev_type.value}"

    raise ValueError(f"알 수 없는 명령: {cmd}")


async def run(
    face: FaceState,
    ctx: StateContext,
    servos: ServoController | None = None,
    poll_interval_sec: float | None = None,
    emit_event: Callable[[SensorEvent], None] | None = None,
) -> None:
    """주기적으로 pending 명령 처리.

    emit_event: gesture 명령에서 SensorEvent를 SensorManager.events 등으로 보낼 콜백.
    """
    if poll_interval_sec is None:
        poll_interval_sec = BEHAVIOR.proactive_eval_interval_sec  # 1s 기본 재사용
    log.info(f"command_executor 시작 (poll={poll_interval_sec}s)")
    while True:
        await asyncio.sleep(poll_interval_sec)
        try:
            pending = memory.fetch_pending_commands()
        except Exception as e:
            log.debug(f"명령 조회 실패: {e}")
            continue
        for c in pending:
            cmd_id = c["id"]
            try:
                result = await _execute(
                    c["cmd"], c["args"], face, ctx, servos,
                    emit_event=emit_event,
                )
                memory.mark_command_done(cmd_id, result)
                log.info(f"명령 #{cmd_id} 완료: {c['cmd']} → {result[:60]}")
            except Exception as e:
                memory.mark_command_failed(cmd_id, str(e))
                log.warning(f"명령 #{cmd_id} 실패: {c['cmd']} → {e}")
