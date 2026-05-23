"""Breathing + micro-saccades + mood drift 테스트."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.brain.state_machine import State, StateContext
from src.face import eyes
from src.face.expressions import NEUTRAL, MouthShape
from src.face.renderer import FaceState
from src.tasks import head_tracker, mood_drift


# === Breathing ===

def _reset_breath_state():
    head_tracker._breath_state["cycle_start"] = None
    head_tracker._breath_state["next_cycle_at"] = 0.0


def test_breathing_oscillates_during_cycle():
    """사이클 진행 중에는 tilt가 시간에 따라 변하고 진폭 한계 안."""
    _reset_breath_state()
    samples = [head_tracker._breathing_offsets(t) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    for pan, tilt in samples:
        assert abs(pan) <= head_tracker.BREATH_PAN_AMP_DEG + 1e-6
        assert abs(tilt) <= head_tracker.BREATH_TILT_AMP_MAX_DEG + 1e-6
    tilt_vals = [t for _, t in samples]
    assert len(set(round(t, 3) for t in tilt_vals)) > 1


def test_breathing_period_completes():
    """tilt가 한 주기 후 0으로 떨어짐 (사이클 종료)."""
    _reset_breath_state()
    head_tracker._breathing_offsets(0.0)  # 사이클 시작
    _, t_period = head_tracker._breathing_offsets(head_tracker.BREATH_TILT_PERIOD_SEC)
    assert abs(t_period) < 1e-6


def test_breathing_silent_between_cycles():
    """사이클 끝난 직후부터 다음 1시간까지는 (0, 0)."""
    _reset_breath_state()
    head_tracker._breathing_offsets(0.0)  # 사이클 시작
    # 사이클 끝 (60초)
    head_tracker._breathing_offsets(head_tracker.BREATH_TILT_PERIOD_SEC)
    # 사이클 끝 직후부터 다음 인터벌 직전까지 휴면
    for t in (61.0, 600.0, 1800.0, 3599.0):
        assert head_tracker._breathing_offsets(t) == (0.0, 0.0)


def test_breathing_restarts_after_interval():
    """1시간 지나면 새 사이클이 다시 시작 (tilt 다시 nonzero)."""
    _reset_breath_state()
    head_tracker._breathing_offsets(0.0)
    head_tracker._breathing_offsets(head_tracker.BREATH_TILT_PERIOD_SEC)  # 끝
    # 1시간 뒤
    t_next = head_tracker.BREATH_INTERVAL_SEC
    # 사이클 시작 직후 sin(0)=0이라 1초 뒤로 샘플
    head_tracker._breathing_offsets(t_next)
    _, tilt = head_tracker._breathing_offsets(t_next + 15.0)
    assert abs(tilt) > 0.01


# === Micro-saccades ===

def test_saccade_updates_within_amplitude():
    state = eyes.EyeState()
    now = time.time()
    for i in range(50):
        eyes.update_saccade(state, now + i * 0.05)
    # 진폭 안
    assert abs(state.saccade_x) <= eyes.SACCADE_AMP_X + 0.01
    assert abs(state.saccade_y) <= eyes.SACCADE_AMP_Y + 0.01


def test_saccade_actually_moves():
    """100번 업데이트하면 saccade_x가 실제로 변화."""
    state = eyes.EyeState()
    samples = []
    now = time.time()
    for i in range(100):
        eyes.update_saccade(state, now + i * 0.05)
        samples.append(state.saccade_x)
    # 모두 같은 값은 아님
    unique = len({round(s, 4) for s in samples})
    assert unique > 5


def test_saccade_independent_of_gaze():
    """gaze_x를 0.5로 둬도 saccade_x는 ±amp 안에서만 흔들려야 함."""
    state = eyes.EyeState(gaze_x=0.5, gaze_y=-0.3)
    now = time.time()
    for i in range(30):
        eyes.update_saccade(state, now + i * 0.1)
    assert abs(state.saccade_x) <= eyes.SACCADE_AMP_X + 0.01
    assert state.gaze_x == 0.5  # gaze 자체는 안 건드림


def test_renderer_calls_update_saccade(monkeypatch):
    """draw_face_to_surface가 update_saccade를 호출하는지."""
    import pygame
    from src.face import renderer

    pygame.init()
    surface = pygame.Surface((320, 240))
    face = FaceState(expression=NEUTRAL)

    calls = []
    orig = eyes.update_saccade

    def tracking(state, now):
        calls.append(now)
        orig(state, now)

    monkeypatch.setattr(renderer.eyes, "update_saccade", tracking)
    renderer.draw_face_to_surface(surface, face)
    assert len(calls) == 1


# === Mood drift ===

def test_mood_drift_default_at_noon():
    ctx = StateContext()
    ctx.user_present = False
    ctx.last_user_seen_at = time.time()  # 방금 봄
    mood = mood_drift._select_mood(ctx, time.time(), 12)
    assert mood.name in {"neutral", "content", "thinking", "happy"}


def test_mood_drift_sleepy_at_night():
    ctx = StateContext()
    mood = mood_drift._select_mood(ctx, time.time(), 2)  # 새벽 2시
    assert mood.name in {"sleepy", "content"}


def test_mood_drift_yawn_after_long_absence():
    ctx = StateContext()
    ctx.user_present = False
    now = time.time()
    ctx.last_user_seen_at = now - 2 * 3600  # 2시간 전
    moods = {mood_drift._select_mood(ctx, now, 14).name for _ in range(50)}
    # 1시간+ 부재 → sleepy 또는 yawn
    assert moods.issubset({"sleepy", "yawn"})


def test_mood_drift_greeting_immediately_after_user_appears():
    ctx = StateContext()
    ctx.user_present = True
    ctx.last_user_seen_at = time.time()  # 방금 봄
    moods = {mood_drift._select_mood(ctx, time.time(), 14).name for _ in range(30)}
    assert moods.issubset({"happy", "content", "starstruck"})


def test_mood_drift_skips_non_eligible_states(monkeypatch):
    """TALKING 등에서는 표정 변경 안 함."""
    face = FaceState(expression=NEUTRAL)
    ctx = StateContext()
    ctx.state = State.TALKING

    monkeypatch.setattr(mood_drift, "CHECK_INTERVAL_SEC", 0.01)

    async def go():
        task = asyncio.create_task(mood_drift.run_mood_drift(face, ctx))
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    saved = face.expression.name
    asyncio.run(go())
    assert face.expression.name == saved
