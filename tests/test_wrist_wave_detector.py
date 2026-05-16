"""WristWaveDetector — pose keypoint 기반 wave 감지 테스트."""

from __future__ import annotations

import math

import numpy as np

from src.vision.camera import (
    KP_L_SHOULDER, KP_L_WRIST, KP_R_SHOULDER, KP_R_WRIST,
)
from src.vision.wrist_wave_detector import WristWaveDetector


def _build_keypoints(
    wrist_x: float,
    wrist_y: float = 0.3,
    shoulder_l_x: float = 0.4,
    shoulder_r_x: float = 0.6,
    shoulder_y: float = 0.4,
    side: str = "right",
    conf: float = 0.9,
) -> np.ndarray:
    """필수 keypoint만 채운 (17, 3) 배열. 나머지는 confidence 0."""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[KP_L_SHOULDER] = [shoulder_l_x, shoulder_y, conf]
    kps[KP_R_SHOULDER] = [shoulder_r_x, shoulder_y, conf]
    if side == "right":
        kps[KP_R_WRIST] = [wrist_x, wrist_y, conf]
    else:
        kps[KP_L_WRIST] = [wrist_x, wrist_y, conf]
    return kps


def test_oscillating_wrist_detected():
    """손목이 좌우로 큰 진폭으로 움직이면 wave 감지."""
    det = WristWaveDetector(fps=10, history_sec=1.5)
    hits = 0
    for i in range(30):
        # 어깨 위(y=0.3, shoulder_y=0.4)에서 좌우 진동 — 어깨너비 0.2의 60% 진폭
        wrist_x = 0.5 + math.sin(i * math.pi / 3) * 0.12
        kps = _build_keypoints(wrist_x, wrist_y=0.3)
        if det.process(kps):
            hits += 1
    assert hits >= 1


def test_static_wrist_not_detected():
    """손목이 어깨 위에 있지만 움직이지 않으면 wave 아님."""
    det = WristWaveDetector(fps=10, history_sec=1.5)
    for _ in range(30):
        kps = _build_keypoints(0.5, wrist_y=0.3)
        assert not det.process(kps)


def test_wrist_at_leg_level_ignored():
    """손목이 다리 높이(어깨에서 너무 멀리 아래)면 wave 후보 X."""
    det = WristWaveDetector(fps=10, history_sec=1.5)
    for i in range(30):
        wrist_x = 0.5 + math.sin(i * math.pi / 3) * 0.12
        # 어깨 y=0.4, 손목 y=0.95 → 0.55 아래 (다리 높이) — 무시되어야
        kps = _build_keypoints(wrist_x, wrist_y=0.95)
        assert not det.process(kps)


def test_low_confidence_keypoint_ignored():
    """confidence 낮은 keypoint는 무시 — history 안 쌓임."""
    det = WristWaveDetector(fps=10, history_sec=1.5)
    for i in range(30):
        wrist_x = 0.5 + math.sin(i * math.pi / 3) * 0.12
        kps = _build_keypoints(wrist_x, wrist_y=0.3, conf=0.05)
        assert not det.process(kps)


def test_amplitude_normalization_by_shoulder_width():
    """진폭이 어깨너비에 비해 작으면 감지 안 됨 (멀리 있어도 동일 기준)."""
    det = WristWaveDetector(fps=10, history_sec=1.5)
    # 어깨너비 0.4 (큼), 진폭 0.05 — ratio 0.125 너무 작음
    hits = 0
    for i in range(30):
        wrist_x = 0.5 + math.sin(i * math.pi / 3) * 0.05
        kps = _build_keypoints(
            wrist_x, wrist_y=0.3,
            shoulder_l_x=0.3, shoulder_r_x=0.7,
        )
        if det.process(kps):
            hits += 1
    assert hits == 0


def test_cooldown_after_detection():
    """감지 후 cooldown_sec 동안엔 재감지 차단."""
    det = WristWaveDetector(fps=10, history_sec=1.5, cooldown_sec=5.0)
    hits = 0
    for i in range(60):
        wrist_x = 0.5 + math.sin(i * math.pi / 3) * 0.12
        kps = _build_keypoints(wrist_x, wrist_y=0.3)
        if det.process(kps):
            hits += 1
    # 30프레임 안에 1회 감지, cooldown 5s = 50프레임 차단 → 이후 또 가능
    assert hits == 1


def test_none_keypoints_no_crash():
    det = WristWaveDetector(fps=10)
    assert not det.process(None)


def test_reset_clears_history():
    det = WristWaveDetector(fps=10)
    for i in range(5):
        kps = _build_keypoints(0.5 + i * 0.01, wrist_y=0.3)
        det.process(kps)
    det.reset()
    assert len(det.left_history) == 0
    assert len(det.right_history) == 0
