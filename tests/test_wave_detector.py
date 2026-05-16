"""WaveDetector 단위 테스트 — 가짜 frame 시퀀스로 oscillation 인식 검증."""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.vision.wave_detector import WaveDetector


def _frame_with_blob(w: int, h: int, blob_x: int, blob_y: int,
                     blob_w: int = 30, blob_h: int = 50) -> np.ndarray:
    """검은 배경에 흰색 사각형 한 개. blob_x/y가 사각형 중심."""
    img = np.full((h, w, 3), 30, dtype=np.uint8)  # 어두운 회색 배경
    x0 = max(0, blob_x - blob_w // 2)
    x1 = min(w, blob_x + blob_w // 2)
    y0 = max(0, blob_y - blob_h // 2)
    y1 = min(h, blob_y + blob_h // 2)
    img[y0:y1, x0:x1] = 220
    return img


def _feed_oscillation(det: WaveDetector, bbox, *, frames: int,
                      amp_px: int = 40, w: int = 160, h: int = 240,
                      hz: float = 2.0) -> int:
    """좌우 oscillation 시퀀스 push. 감지된 횟수 반환."""
    fps = det.fps
    detections = 0
    center_x = w // 2
    blob_y = int(h * 0.3)  # bbox 위쪽
    for i in range(frames):
        # sin 으로 좌우 흔들기
        phase = (i / fps) * 2 * np.pi * hz
        blob_x = int(center_x + np.sin(phase) * amp_px)
        frame = _frame_with_blob(w, h, blob_x, blob_y)
        if det.process(frame, bbox):
            detections += 1
    return detections


def test_detects_clear_oscillation():
    det = WaveDetector(fps=10, history_sec=2.0, cooldown_sec=0.1,
                       min_motion_pixels=20)
    bbox = (0.1, 0.05, 0.9, 0.95)
    hits = _feed_oscillation(det, bbox, frames=40, amp_px=45, hz=2.0)
    assert hits >= 1


def test_ignores_static_scene():
    det = WaveDetector(fps=10, history_sec=2.0)
    bbox = (0.1, 0.05, 0.9, 0.95)
    w, h = 160, 240
    static_frame = _frame_with_blob(w, h, w // 2, int(h * 0.3))
    hits = 0
    for _ in range(40):
        if det.process(static_frame, bbox):
            hits += 1
    assert hits == 0


def test_ignores_linear_drift():
    """blob이 한 방향으로만 천천히 이동 — wave 아님."""
    det = WaveDetector(fps=10, history_sec=2.0)
    bbox = (0.1, 0.05, 0.9, 0.95)
    w, h = 160, 240
    blob_y = int(h * 0.3)
    hits = 0
    for i in range(40):
        blob_x = 30 + i * 2  # 한쪽으로만 drift
        frame = _frame_with_blob(w, h, blob_x, blob_y)
        if det.process(frame, bbox):
            hits += 1
    assert hits == 0


def test_reset_on_no_bbox():
    det = WaveDetector(fps=10)
    bbox = (0.1, 0.05, 0.9, 0.95)
    w, h = 160, 240
    # 몇 프레임 push
    for i in range(5):
        det.process(_frame_with_blob(w, h, 80 + i * 5, 50), bbox)
    assert len(det.centroid_history) > 0
    # bbox=None 호출 → 초기화
    det.process(_frame_with_blob(w, h, 80, 50), None)
    assert len(det.centroid_history) == 0
    assert det._prev_gray is None


def test_cooldown_prevents_immediate_redetection():
    det = WaveDetector(fps=10, history_sec=1.5, cooldown_sec=5.0,
                       min_motion_pixels=20)
    bbox = (0.1, 0.05, 0.9, 0.95)
    hits_1 = _feed_oscillation(det, bbox, frames=30, amp_px=45, hz=2.0)
    assert hits_1 >= 1
    # 즉시 한 번 더 — cooldown으로 차단되어야 함
    hits_2 = _feed_oscillation(det, bbox, frames=30, amp_px=45, hz=2.0)
    assert hits_2 == 0


def test_handles_none_frame():
    det = WaveDetector()
    assert det.process(None, (0.1, 0.1, 0.9, 0.9)) is False


def test_bbox_jump_resets_history():
    det = WaveDetector(fps=10)
    bbox_a = (0.1, 0.05, 0.4, 0.95)
    bbox_b = (0.6, 0.05, 0.9, 0.95)
    w, h = 160, 240
    for i in range(5):
        det.process(_frame_with_blob(w, h, 50 + i * 3, 50), bbox_a)
    # bbox가 크게 점프 → reset
    det.process(_frame_with_blob(w, h, 130, 50), bbox_b)
    assert len(det.centroid_history) == 0
