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
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.config import BEHAVIOR
from src.face import expressions as expr
from src.face.eyes import trigger_blink
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.sensors.base import SensorEvent, SensorEventType
from src.utils.logger import get_logger
from src.vision import debug_snapshot

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
    perception: PerceptionState | None = None,
    get_session_id: Callable[[], int | None] | None = None,
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

        async def _do_dance():
            async with motion_busy_scope(ctx):
                await poses.dance(servos, face, bpm=bpm, beats=beats)

        asyncio.create_task(_do_dance())
        return f"dance: {beats}@{bpm}bpm"

    if cmd == "pose":
        if servos is None:
            raise RuntimeError("서보 없음")
        kind = args.get("kind", "")
        fn = _POSE_MAP.get(kind)
        if fn is None:
            raise ValueError(f"알 수 없는 pose: {kind}. 가능: {list(_POSE_MAP)}")

        async def _do_pose():
            async with motion_busy_scope(ctx):
                await fn(servos)

        asyncio.create_task(_do_pose())
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
        # 원격 세션이 이미지 없이 1초 단위로 읽는 perception 요약.
        # 값이 오래됐으면(stale_sec 초과) None — 원격이 stale 신호에 반응 안 하게.
        now = time.time()
        st: dict = {
            "state": ctx.state.value,
            "user_present": ctx.user_present,
            "expression": face.expression.name,
            "user_name": ctx.user_name,
            "brightness": face.brightness,
            "speech": face.speech_text if now < face.speech_until else None,
        }
        if perception is not None:
            def _fresh(val, at, stale_sec=30.0):
                return val if (val is not None and now - at < stale_sec) else None
            st.update({
                "person_present": perception.person_present,
                "person_seen_ago_sec": (
                    int(now - perception.last_person_seen_at)
                    if perception.last_person_seen_at else None
                ),
                "distance_cm": (
                    int(perception.person_distance_cm)
                    if perception.person_distance_cm > 0 else None
                ),
                "emotion": _fresh(perception.current_emotion, perception.current_emotion_at),
                "gaze": _fresh(perception.gaze_target, perception.gaze_target_at),
                "activity": _fresh(perception.activity_level, perception.activity_level_at, 90.0),
                "posture": _fresh(perception.posture_category, perception.posture_category_at, 90.0),
                "head_pan": perception.head_pan_deg,
                "head_tilt": perception.head_tilt_deg,
                "temp_c": perception.temperature_c,
                "user_spoke_ago_sec": (
                    int(now - perception.last_user_speech_at)
                    if getattr(perception, "last_user_speech_at", 0) else None
                ),
            })
        sid = get_session_id() if get_session_id else None
        if sid is not None:
            try:
                st["work_minutes"] = int(memory.current_work_duration(sid) / 60)
            except Exception:
                st["work_minutes"] = None
        try:
            recent = memory.recent_conversation(minutes=30.0, limit=4)
            st["recent"] = [
                {"role": r.get("speaker"), "kind": r.get("kind"),
                 "text": (r.get("text") or "")[:60], "ago_sec": int(now - float(r.get("ts", now)))}
                for r in recent
            ]
        except Exception:
            pass
        return json.dumps(st, ensure_ascii=False)

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

    if cmd == "snapshot":
        # vision_task가 다음 프레임에서 처리 — bbox/keypoint 어노테이트 후 저장
        note = args.get("note", "")
        debug_snapshot.request_snapshot(note)
        return f"snapshot requested → {debug_snapshot.DEBUG_SNAPSHOT_PATH}"

    raise ValueError(f"알 수 없는 명령: {cmd}")


async def run(
    face: FaceState,
    ctx: StateContext,
    servos: ServoController | None = None,
    poll_interval_sec: float | None = None,
    emit_event: Callable[[SensorEvent], None] | None = None,
    perception: PerceptionState | None = None,
    get_session_id: Callable[[], int | None] | None = None,
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
                    emit_event=emit_event, perception=perception,
                    get_session_id=get_session_id,
                )
                memory.mark_command_done(cmd_id, result)
                log.info(f"명령 #{cmd_id} 완료: {c['cmd']} → {result[:60]}")
            except Exception as e:
                memory.mark_command_failed(cmd_id, str(e))
                log.warning(f"명령 #{cmd_id} 실패: {c['cmd']} → {e}")
