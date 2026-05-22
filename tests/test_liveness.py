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

def test_breathing_oscillates():
    """breath offset이 시간에 따라 변하는지 + 진폭 한계 안에 들어오는지."""
    samples = [head_tracker._breathing_offsets(t) for t in (0.0, 1.0, 2.0, 3.0, 4.0)]
    # 진폭 안
    for pan, tilt in samples:
        assert abs(pan) <= head_tracker.BREATH_PAN_AMP_DEG + 1e-6
        assert abs(tilt) <= head_tracker.BREATH_TILT_AMP_DEG + 1e-6
    # 시간 흐름에 따라 다른 값 (정지 아님) — tilt 기준 (PAN amp는 0일 수 있음)
    tilt_vals = [t for _, t in samples]
    assert len(set(round(t, 3) for t in tilt_vals)) > 1


def test_breathing_period_completes():
    """tilt가 한 주기 후 다시 비슷한 값으로 돌아옴."""
    _, t0 = head_tracker._breathing_offsets(0.0)
    _, t_period = head_tracker._breathing_offsets(head_tracker.BREATH_TILT_PERIOD_SEC)
    assert abs(t0 - t_period) < 0.05


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
