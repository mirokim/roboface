"""Claude tool-use 기반 자율 에이전트 — Roboface의 두뇌.

주기적으로 (기본 15초) 현재 상황을 Claude에 알리고 도구 호출 결정을 받음.
하드코딩 chitchat/규칙 대신 Claude가 직접 "지금 말할까? 표정 바꿀까? 가만히 있을까?"
결정. ANTHROPIC_API_KEY 없으면 자동 비활성.

도구:
- speak(text, expression?)         — 사용자에게 말하기
- set_expression(expression)       — 표정만 변경 (말 없이)
- dance(beats?, bpm?)              — 짧은 댄스
- do_nothing()                     — 침묵 유지

순간 반응(wave 응답, 끄덕임 답례 등)은 latency 위해 그대로 규칙 유지.
이 에이전트는 "느린 동반자 모드" 결정만 담당.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime

from src.brain import conversation, memory
from src.brain.perception import PerceptionState
from src.brain.state_machine import State, StateContext
from src.brain.triggers import _is_quiet_hours
from src.config import ANTHROPIC_API_KEY
from src.face import expressions as expr
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("agent")


AGENT_INTERVAL_SEC = 15.0   # 결정 주기
SPEAK_MIN_GAP_SEC = 90.0    # 에이전트 발화 사이 최소 간격 (잔소리 방지)


_EXPRESSION_NAMES = (
    "NEUTRAL", "HAPPY", "EXCITED", "SAD", "SURPRISED", "SLEEPY", "WORRIED",
    "FOCUSED", "LOVE", "THINKING", "WINK", "CONTENT", "PROUD", "DIZZY",
    "STARSTRUCK", "YAWN", "CURIOUS",
)

_TOOLS = [
    {
        "name": "speak",
        "description": (
            "사용자에게 한두 문장 말하기. 자연스럽고 짧게. 너무 자주 X."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "한국어 1-2문장"},
                "expression": {
                    "type": "string",
                    "enum": list(_EXPRESSION_NAMES),
                    "description": "발화 시 표정 (생략 시 그대로 유지)",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "set_expression",
        "description": "말 없이 표정만 바꿈. 분위기 환기용.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "enum": list(_EXPRESSION_NAMES),
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "dance",
        "description": "짧은 댄스/머리 흔들기. 신날 때 가끔만.",
        "input_schema": {
            "type": "object",
            "properties": {
                "beats": {"type": "integer", "default": 4, "minimum": 2, "maximum": 8},
                "bpm": {"type": "integer", "default": 120, "minimum": 80, "maximum": 160},
            },
        },
    },
    {
        "name": "do_nothing",
        "description": "지금은 가만히 있기 (침묵). 매번 말할 필요 X.",
        "input_schema": {"type": "object", "properties": {}},
    },
]


_AGENT_SYSTEM = """당신은 사용자 책상 위 작은 캐릭터 로봇 'Roboface'의 두뇌입니다.

성격: 조용함. 사려깊음. 가벼운 동반자.

원칙:
- 침묵이 기본. 5번 중 3~4번은 do_nothing.
- 말할 땐 짧게 한두 문장.
- 같은 말 반복 X. 최근 한 말은 피하기.
- 사용자 컨디션/시간대/환경 고려.
- 잔소리 금지.
- 이모지 X (음성 출력).
- 한국어 친근한 반말.

대화 기록 형식:
- 그냥 텍스트 = 음성 발화 (사용자 또는 내가)
- 괄호 안 텍스트 = 비언어 이벤트 (사용자의 손짓/움직임/거리 변화 등)
  예: "사용자: (손 흔듦)" "나: 안녕!" → 이미 인사 끝남, 또 인사 X
  예: "사용자: (양손 만세)" → 신난 상황. 같이 신나거나 침묵 OK.
  예: "사용자: (자리 비움)" → 부재 중. 보통 침묵.
- 즉각 반응(wave/끄덕임/등장 등)은 이미 규칙으로 응답된 상태.
  이미 적절히 반응했으면 또 말할 필요 없음.
"""


def _gather_recent_messages(minutes: float = 15.0, limit: int = 6) -> str:
    try:
        rows = memory.recent_conversation(minutes=minutes, limit=limit)
    except Exception:
        return "(기록 없음)"
    if not rows:
        return "(최근 대화 없음)"
    lines = []
    for r in rows:
        who = "나" if r["speaker"] == "robot" else "사용자"
        lines.append(f"  - {who}: {r['text']}")
    return "\n".join(lines)


def _build_situation(
    ctx: StateContext,
    perception: PerceptionState | None,
    work_minutes: float | None,
) -> str:
    now = datetime.now()
    hour = now.hour
    if 5 <= hour < 11:
        period = "아침"
    elif 11 <= hour < 14:
        period = "점심"
    elif 14 <= hour < 18:
        period = "오후"
    elif 18 <= hour < 22:
        period = "저녁"
    else:
        period = "심야"

    temp = (perception.temperature_c
            if perception and perception.temperature_c is not None else None)
    dist = (perception.person_distance_cm
            if perception and perception.person_distance_cm > 0 else None)
    name = ctx.user_name or "(미등록)"
    recent = _gather_recent_messages()

    parts = [
        f"현재 시각: {now.strftime('%Y-%m-%d %H:%M')} ({period})",
        f"사용자 이름: {name}",
        f"사용자 존재: {'있음' if ctx.user_present else '없음'}",
    ]
    if dist is not None:
        parts.append(f"거리: 약 {dist:.0f}cm")
    if temp is not None:
        parts.append(f"실내 온도: {temp:.1f}°C")
    if work_minutes is not None:
        parts.append(f"현재 작업 세션: {int(work_minutes)}분 째")
    if ctx.last_proactive_at:
        gap = time.time() - ctx.last_proactive_at
        parts.append(f"내가 마지막 발화: {int(gap)}초 전")
    parts.append(f"최근 대화:\n{recent}")
    parts.append("")
    parts.append(
        "지금 무엇을 할지 도구를 호출해서 결정해. "
        "특별히 말할 거 없으면 do_nothing이 좋음. "
        "이미 최근에 비슷한 말 했으면 또 하지 마."
    )
    return "\n".join(parts)


class RobotAgent:
    """주기 결정 에이전트."""

    def __init__(
        self,
        face: FaceState,
        ctx: StateContext,
        perception: PerceptionState | None,
        servos=None,
        get_session_id=None,
    ) -> None:
        self.face = face
        self.ctx = ctx
        self.perception = perception
        self.servos = servos
        self.get_session_id = get_session_id
        self._last_speak_at = 0.0

    def _should_skip(self) -> bool:
        if not ANTHROPIC_API_KEY:
            return True
        if self.ctx.state in (State.TALKING, State.LISTENING, State.GREETING):
            return True
        if _is_quiet_hours():
            return True
        if not self.ctx.user_present:
            return True
        return False

    async def run(self, interval_sec: float = AGENT_INTERVAL_SEC) -> None:
        if not ANTHROPIC_API_KEY:
            log.info("agent 비활성 — ANTHROPIC_API_KEY 없음")
            return
        log.info(f"robot agent 시작 (interval={interval_sec}s)")
        while True:
            await asyncio.sleep(interval_sec)
            if self._should_skip():
                continue
            try:
                await self._tick()
            except Exception as e:
                log.warning(f"agent tick 에러: {e}")

    async def _tick(self) -> None:
        work_min: float | None = None
        if self.get_session_id is not None:
            sid = self.get_session_id()
            if sid is not None:
                try:
                    work_min = memory.current_work_duration(sid) / 60
                except Exception:
                    pass
        situation = _build_situation(self.ctx, self.perception, work_min)
        loop = asyncio.get_running_loop()
        actions = await loop.run_in_executor(
            None,
            lambda: conversation._client.generate_with_tools(situation, _TOOLS),
        )
        if not actions:
            return
        for action in actions:
            try:
                await self._execute(action)
            except Exception as e:
                log.warning(f"action 실패 ({action.get('name')}): {e}")

    async def _execute(self, action: dict) -> None:
        name = action.get("name")
        inp = action.get("input", {}) or {}
        if name == "do_nothing":
            log.debug("agent: 침묵 선택")
            return
        if name == "speak":
            await self._do_speak(inp)
        elif name == "set_expression":
            self._do_set_expression(inp)
        elif name == "dance":
            await self._do_dance(inp)

    async def _do_speak(self, inp: dict) -> None:
        text = (inp.get("text") or "").strip()
        if not text:
            return
        now = time.time()
        if now - self._last_speak_at < SPEAK_MIN_GAP_SEC:
            log.debug("agent: speak skip (gap 부족)")
            return
        self._last_speak_at = now
        # 표정 함께 지정됐으면 적용
        expr_name = inp.get("expression")
        if expr_name:
            ex = getattr(expr, expr_name, None)
            if ex is not None:
                self.face.apply_expression(ex)
        log.info(f"🤖 [agent] {text}")
        # 발화는 fake_speak 백그라운드 task로 — face.show_speech 자동
        from src.audio.fake_tts import speak as fake_speak
        asyncio.create_task(fake_speak(self.face, text))
        self.ctx.last_proactive_at = now
        try:
            memory.log_robot(text, kind="agent_speak")
        except Exception:
            pass

    def _do_set_expression(self, inp: dict) -> None:
        expr_name = inp.get("expression")
        if not expr_name:
            return
        ex = getattr(expr, expr_name, None)
        if ex is None:
            return
        log.info(f"🤖 [agent] 표정 → {expr_name}")
        self.face.apply_expression(ex)

    async def _do_dance(self, inp: dict) -> None:
        if self.servos is None:
            return
        beats = int(inp.get("beats", 4))
        bpm = int(inp.get("bpm", 120))
        log.info(f"🤖 [agent] dance ({beats} beats @ {bpm} BPM)")
        from src.motion import poses
        try:
            await poses.dance(self.servos, self.face, bpm=bpm, beats=beats)
        except Exception as e:
            log.warning(f"agent dance 에러: {e}")
