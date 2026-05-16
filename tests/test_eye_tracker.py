"""eye_tracker + idle_gaze 양보 테스트."""

from __future__ import annotations

import asyncio

import pytest

from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext
from src.face.expressions import NEUTRAL
from src.face.renderer import FaceState
from src.tasks import eye_tracker, idle_animation
from src.tasks.head_tracker import PAN_INVERT


def _run_eye_tracker_for(face, perception, ctx, seconds: float) -> None:
    async def go():
        task = asyncio.create_task(eye_tracker.run_eye_tracker(face, perception, ctx))
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    asyncio.run(go())


def test_eye_tracker_moves_to_user_on_right():
    """사람이 화면 오른쪽 (cx=0.9)에 있으면 gaze_x가 그쪽으로 이동."""
    face = FaceState(expression=NEUTRAL)
    perception = PerceptionState()
    # bbox 중앙이 화면 오른쪽 끝
    perception.update_person(bbox=(0.8, 0.4, 1.0, 0.6))
    ctx = StateContext()

    _run_eye_tracker_for(face, perception, ctx, 0.6)

    # cx=0.9 → ox=0.4 → PAN_INVERT 적용 후 ±, 부호는 환경에 따라
    expected_sign = -1 if PAN_INVERT else 1
    assert face.eye_state.gaze_x * expected_sign > 0.1
    # 진폭 한계 초과 X
    assert abs(face.eye_state.gaze_x) <= eye_tracker.MAX_GAZE_X + 0.01


def test_eye_tracker_returns_to_center_when_no_user():
    """사람 없으면 gaze가 0 근처로 복귀."""
    face = FaceState(expression=NEUTRAL)
    face.eye_state.gaze_x = 0.3
    face.eye_state.gaze_y = 0.15
    perception = PerceptionState()  # person_present=False
    ctx = StateContext()

    _run_eye_tracker_for(face, perception, ctx, 2.0)

    assert abs(face.eye_state.gaze_x) < 0.1
    assert abs(face.eye_state.gaze_y) < 0.05


def test_eye_tracker_respects_amplitude_clamp():
    """bbox가 극단(cx=0/1)이라도 진폭 한계 안에서만 움직임."""
    face = FaceState(expression=NEUTRAL)
    perception = PerceptionState()
    perception.update_person(bbox=(0.0, 0.0, 0.01, 0.01))  # 좌상단 끝
    ctx = StateContext()

    _run_eye_tracker_for(face, perception, ctx, 1.0)

    assert abs(face.eye_state.gaze_x) <= eye_tracker.MAX_GAZE_X + 0.01
    assert abs(face.eye_state.gaze_y) <= eye_tracker.MAX_GAZE_Y + 0.01


def test_idle_gaze_yields_when_person_present(monkeypatch):
    """사람이 보이면 idle_gaze가 시선을 건드리지 않아야 함."""
    face = FaceState(expression=NEUTRAL)
    perception = PerceptionState()
    perception.update_person(bbox=(0.4, 0.4, 0.6, 0.6))

    # gaze 보존 확인용 마커
    face.eye_state.gaze_x = 0.25
    face.eye_state.gaze_y = -0.1

    # 짧게 만들어 즉시 실행 — random.uniform 결과를 작은 값으로 고정
    monkeypatch.setattr(
        idle_animation.random, "uniform", lambda lo, hi: 0.01,
    )

    async def go():
        task = asyncio.create_task(idle_animation.run_idle_gaze(face, perception))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())
    # idle_gaze가 건드리지 않았으니 그대로
    assert face.eye_state.gaze_x == 0.25
    assert face.eye_state.gaze_y == -0.1


def test_idle_gaze_runs_when_no_perception():
    """perception=None이면 기존 동작 유지."""
    face = FaceState(expression=NEUTRAL)

    async def go():
        task = asyncio.create_task(idle_animation.run_idle_gaze(face, None))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(go())  # 에러만 안 나면 OK
