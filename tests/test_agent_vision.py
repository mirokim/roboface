"""Phase vision — image_encoding + agent vision attach 조건부 로직."""

from __future__ import annotations

import base64
import time
from dataclasses import replace

import numpy as np
import pytest

from src.brain import agent, image_encoding
from src.brain.perception import PerceptionState
from src.brain.state_machine import StateContext
from src.config import BEHAVIOR
from src.face.renderer import FaceState


def _patch_behavior(monkeypatch, **kwargs):
    """frozen BehaviorConfig를 replace로 새로 만들고 agent 모듈에 주입."""
    new = replace(BEHAVIOR, **kwargs)
    monkeypatch.setattr(agent, "BEHAVIOR", new)
    return new


# === image_encoding ===

def test_encode_jpeg_b64_basic():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    b64 = image_encoding.encode_jpeg_b64(frame, quality=70, max_side_px=480)
    assert b64 is not None
    # decode 시 JPEG SOI 마커(FF D8)로 시작
    raw = base64.b64decode(b64)
    assert raw[0] == 0xFF and raw[1] == 0xD8


def test_encode_jpeg_downscales():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    b64 = image_encoding.encode_jpeg_b64(frame, quality=70, max_side_px=480)
    assert b64 is not None
    # 다운샘플로 데이터 크기 작아짐
    assert len(b64) < 30_000


def test_encode_jpeg_returns_none_for_invalid():
    assert image_encoding.encode_jpeg_b64(None) is None
    assert image_encoding.encode_jpeg_b64(np.zeros((10, 10), dtype=np.uint8)) is None


def test_encode_jpeg_quality_affects_size():
    """품질 낮으면 데이터 작아짐."""
    frame = (np.random.rand(240, 320, 3) * 255).astype(np.uint8)
    big = image_encoding.encode_jpeg_b64(frame, quality=95, max_side_px=480)
    small = image_encoding.encode_jpeg_b64(frame, quality=30, max_side_px=480)
    assert big is not None and small is not None
    assert len(small) < len(big)


# === RobotAgent._maybe_encode_frame ===

def _make_agent():
    face = FaceState()
    ctx = StateContext()
    perception = PerceptionState()
    return agent.RobotAgent(face, ctx, perception)


def test_vision_returns_none_when_disabled(monkeypatch):
    _patch_behavior(monkeypatch, agent_vision_enabled=False)
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time()
    assert a._maybe_encode_frame() is None


def test_vision_returns_none_without_frame(monkeypatch):
    _patch_behavior(monkeypatch, agent_vision_enabled=True)
    a = _make_agent()
    assert a._maybe_encode_frame() is None


def test_vision_returns_none_for_stale_frame(monkeypatch):
    _patch_behavior(monkeypatch, agent_vision_enabled=True)
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time() - 60   # 60초 전 — stale
    assert a._maybe_encode_frame() is None


def test_vision_attaches_on_first_call(monkeypatch):
    _patch_behavior(
        monkeypatch, agent_vision_enabled=True, agent_vision_min_interval_sec=0.0,
    )
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time()
    a.perception.current_emotion = "smile"
    b64 = a._maybe_encode_frame()
    assert b64 is not None


def test_vision_skips_when_no_change_and_recent(monkeypatch):
    _patch_behavior(
        monkeypatch,
        agent_vision_enabled=True,
        agent_vision_min_interval_sec=0.0,
        agent_vision_max_interval_sec=600.0,
    )
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time()
    assert a._maybe_encode_frame() is not None
    assert a._maybe_encode_frame() is None


def test_vision_reattaches_on_emotion_change(monkeypatch):
    _patch_behavior(
        monkeypatch, agent_vision_enabled=True, agent_vision_min_interval_sec=0.0,
    )
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time()
    a.perception.current_emotion = "neutral"
    assert a._maybe_encode_frame() is not None
    a.perception.current_emotion = "smile"
    a.perception.last_frame_at = time.time()
    assert a._maybe_encode_frame() is not None


def test_vision_respects_min_interval(monkeypatch):
    _patch_behavior(
        monkeypatch, agent_vision_enabled=True, agent_vision_min_interval_sec=60.0,
    )
    a = _make_agent()
    a.perception.last_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    a.perception.last_frame_at = time.time()
    a.perception.current_emotion = "neutral"
    assert a._maybe_encode_frame() is not None
    a.perception.current_emotion = "smile"
    a.perception.last_frame_at = time.time()
    assert a._maybe_encode_frame() is None
