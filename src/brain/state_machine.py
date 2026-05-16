"""상태 머신 — IDLE / WATCHING / GREETING / TALKING / ALERTING / LISTENING."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from src.face import expressions as expr
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("state_machine")


class State(Enum):
    IDLE = "idle"           # 사용자 부재 — 절전
    WATCHING = "watching"   # 사용자 있음, 관찰 중 (기본)
    GREETING = "greeting"   # 인사 중
    TALKING = "talking"     # 음성 출력 중
    ALERTING = "alerting"   # 자세/휴식 알림
    LISTENING = "listening" # 수동 청취 (음성 분석 중)


@dataclass
class StateContext:
    state: State = State.IDLE
    entered_at: float = field(default_factory=time.time)
    last_user_seen_at: float | None = None
    last_proactive_at: float | None = None
    user_present: bool = False
    # ambient_motion이 서보를 점유 중일 때 True — head_tracker가 양보.
    ambient_motion_active: bool = False
    # 현재 인식된 사용자 이름 (face_memory). None이면 모르는 사람 / 미등록.
    user_name: str | None = None
    # 다음 사용자 face crop을 이 이름으로 등록 (voice_assistant가 set)
    pending_register_name: str | None = None

    def transition(self, new_state: State, face: FaceState) -> None:
        if new_state == self.state:
            return
        log.info(f"상태 전이: {self.state.value} → {new_state.value}")
        self.state = new_state
        self.entered_at = time.time()
        # 상태별 기본 표정 적용
        face.apply_expression(_DEFAULT_EXPRESSIONS[new_state])
        # 절전 처리
        face.brightness = 0.3 if new_state == State.IDLE else 1.0


_DEFAULT_EXPRESSIONS = {
    State.IDLE: expr.SLEEPY,
    State.WATCHING: expr.NEUTRAL,
    State.GREETING: expr.HAPPY,
    State.TALKING: expr.NEUTRAL,  # 입 모양은 별도 제어
    State.ALERTING: expr.WORRIED,
    State.LISTENING: expr.FOCUSED,
}


def time_in_state(ctx: StateContext) -> float:
    return time.time() - ctx.entered_at
