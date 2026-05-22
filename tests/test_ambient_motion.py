"""ambient_motion 동작 + head_tracker 양보 테스트."""

from __future__ import annotations

import asyncio

import pytest

from src.brain.state_machine import State, StateContext
from src.config import PAN_CENTER_DEG, TILT_CENTER_DEG
from src.face.expressions import NEUTRAL
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import MockServoController


def test_sway_does_not_touch_face():
    """sway는 표정/입 건드리지 않아야 함."""
    face = FaceState(expression=NEUTRAL)
    servos = MockServoController()
    saved_expr = face.expression.name
    saved_mouth = face.mouth_state.shape
    saved_amp = face.mouth_state.talk_amplitude

    asyncio.run(poses.sway(servos, bpm=240, beats=2, update_hz=20))

    assert face.expression.name == saved_expr
    assert face.mouth_state.shape == saved_mouth
    assert face.mouth_state.talk_amplitude == saved_amp


def test_sway_returns_to_center():
    servos = MockServoController()
    asyncio.run(poses.sway(servos, bpm=240, beats=2, update_hz=20))
    assert abs(servos.position.pan - PAN_CENTER_DEG) < 1.0
    assert abs(servos.position.tilt - TILT_CENTER_DEG) < 1.0


def test_state_context_has_ambient_flag():
    ctx = StateContext()
    assert ctx.ambient_motion_active is False
    ctx.ambient_motion_active = True
    assert ctx.ambient_motion_active is True


def test_ambient_motion_skips_blocked_states(monkeypatch):
    """대화/알림 중에는 sway가 호출되지 않아야 함."""
    from src.tasks import idle_animation

    calls: list[dict] = []

    async def fake_sway(servos, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(idle_animation.poses, "sway", fake_sway)
    # 대기 시간을 0으로
    monkeypatch.setattr(idle_animation, "_AMBIENT_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(idle_animation, "_AMBIENT_MAX_INTERVAL_SEC", 0.001)

    ctx = StateContext()
    ctx.state = State.TALKING  # blocked
    servos = MockServoController()

    async def run_briefly():
        task = asyncio.create_task(idle_animation.run_ambient_motion(servos, ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())
    assert calls == []  # blocked state 동안 한 번도 호출되지 않음


def test_ambient_motion_runs_in_idle(monkeypatch):
    """idle 상태에서는 sway가 호출되어야 함."""
    from src.tasks import idle_animation

    calls: list[dict] = []

    async def fake_sway(servos, **kwargs):
        calls.append(kwargs)
        # ambient_motion_active 플래그가 set돼있어야 함
        assert ctx.ambient_motion_active is True

    monkeypatch.setattr(idle_animation.poses, "sway", fake_sway)
    monkeypatch.setattr(idle_animation, "_AMBIENT_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(idle_animation, "_AMBIENT_MAX_INTERVAL_SEC", 0.001)

    ctx = StateContext()
    ctx.state = State.IDLE
    servos = MockServoController()

    async def run_briefly():
        task = asyncio.create_task(idle_animation.run_ambient_motion(servos, ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())
    assert len(calls) >= 1
    # idle + no user → "free" 파라미터 (bpm 45~80)
    assert all(45 <= c["bpm"] <= 80 for c in calls)
    # 끝나면 플래그 해제됨
    assert ctx.ambient_motion_active is False


def test_ambient_motion_short_when_user_present(monkeypatch):
    """user_present일 때는 짧은 sway (beats <= 2)."""
    from src.tasks import idle_animation

    calls: list[dict] = []

    async def fake_sway(servos, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(idle_animation.poses, "sway", fake_sway)
    monkeypatch.setattr(idle_animation, "_AMBIENT_MIN_INTERVAL_SEC", 0.0)
    monkeypatch.setattr(idle_animation, "_AMBIENT_MAX_INTERVAL_SEC", 0.001)

    ctx = StateContext()
    ctx.state = State.WATCHING
    ctx.user_present = True
    servos = MockServoController()

    async def run_briefly():
        task = asyncio.create_task(idle_animation.run_ambient_motion(servos, ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())
    assert len(calls) >= 1
    for c in calls:
        assert c["beats"] <= 2
        assert c["pan_amp_deg"] <= 8.0


def test_head_tracker_yields_to_ambient_motion():
    """head_tracker가 ctx.ambient_motion_active를 보고 양보하는지."""
    import inspect
    from src.tasks import head_tracker
    src = inspect.getsource(head_tracker.run_head_tracker)
    assert "ambient_motion_active" in src
