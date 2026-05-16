"""extras (땀/떨림/말풍선) + thermal_state 단위 테스트."""

from __future__ import annotations

import time

import pygame
import pytest

from src.face import extras
from src.face.expressions import HAPPY, NEUTRAL
from src.face.renderer import FaceState, draw_face_to_surface
from src.tasks.thermal_state import _targets_for


pygame.init()


# === shiver ===

def test_shiver_offset_zero_when_intensity_low():
    assert extras.shiver_offset(0.0, 0.0) == (0, 0)
    assert extras.shiver_offset(0.03, 1.23) == (0, 0)


def test_shiver_offset_within_amp():
    dx, dy = extras.shiver_offset(1.0, 0.123)
    assert abs(dx) <= extras.SHIVER_MAX_PX
    assert abs(dy) <= extras.SHIVER_MAX_PX


def test_shiver_offset_changes_over_time():
    samples = {extras.shiver_offset(1.0, t) for t in (0.0, 0.05, 0.1, 0.15, 0.2)}
    assert len(samples) >= 3


# === thermal mapping ===

@pytest.mark.parametrize("temp,sweat,shiver", [
    (35.0, 1.0, 0.0),
    (30.0, 1.0, 0.0),
    (29.0, 0.5, 0.0),
    (28.0, 0.0, 0.0),
    (22.0, 0.0, 0.0),
    (18.0, 0.0, 0.0),
    (16.5, 0.0, 0.5),
    (15.0, 0.0, 1.0),
    (10.0, 0.0, 1.0),
])
def test_thermal_targets(temp, sweat, shiver):
    s, sh = _targets_for(temp)
    assert abs(s - sweat) < 0.01
    assert abs(sh - shiver) < 0.01


def test_thermal_none_temperature():
    assert _targets_for(None) == (0.0, 0.0)


# === speech bubble ===

def test_wrap_text_single_line():
    font = extras.get_font(13)
    lines = extras._wrap_text("Hello", font, 200)
    assert lines == ["Hello"]


def test_wrap_text_breaks_at_max_width():
    font = extras.get_font(13)
    text = "이것은 좀 긴 한국어 문장으로 자동 줄바꿈을 테스트하기 위한 예시입니다"
    lines = extras._wrap_text(text, font, 200)
    assert len(lines) >= 2
    # 각 줄이 max_width 안에 들어가야 함
    for ln in lines:
        assert font.size(ln)[0] <= 200 + 4


def test_draw_speech_bubble_no_crash():
    canvas = pygame.Surface((320, 240))
    extras.draw_speech_bubble(canvas, "안녕 미로!")
    extras.draw_speech_bubble(canvas, "")  # 빈 텍스트 — no-op
    # 긴 텍스트 — truncate 처리
    extras.draw_speech_bubble(canvas, "긴 " * 50)


# === FaceState integration ===

def test_show_speech_sets_until_future():
    face = FaceState(expression=NEUTRAL)
    before = time.time()
    face.show_speech("hi", 2.0)
    assert face.speech_text == "hi"
    # +1.0 buffer (TTS speak에서 추가)
    assert face.speech_until >= before + 2.0


def test_clear_speech():
    face = FaceState(expression=NEUTRAL)
    face.show_speech("hi", 5.0)
    face.clear_speech()
    assert face.speech_text is None
    assert face.speech_until == 0.0


def test_renderer_auto_clears_expired_speech():
    face = FaceState(expression=HAPPY)
    face.speech_text = "old message"
    face.speech_until = time.time() - 5.0  # 이미 만료
    canvas = pygame.Surface((320, 240))
    draw_face_to_surface(canvas, face)
    assert face.speech_text is None


def test_renderer_full_face_with_extras():
    """말풍선 + 땀 + 떨림 동시 — 크래시 안 나면 OK."""
    face = FaceState(expression=HAPPY)
    face.show_speech("덥다 진짜", 3.0)
    face.sweat_intensity = 0.9
    face.shiver_intensity = 0.5
    canvas = pygame.Surface((320, 240))
    draw_face_to_surface(canvas, face)
