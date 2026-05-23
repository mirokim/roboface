"""Phase F-H 추가 — gaze_target / activity_level / posture_category."""

from __future__ import annotations

import time

import pytest

from src.tasks import activity_monitor
from src.tasks.posture_monitor import PostureReading
from src.vision.pose_gestures import GazeTargetClassifier


# === GazeTargetClassifier ===

# COCO 17 keypoint 순서: nose(0), l_eye(1), r_eye(2), l_ear(3), r_ear(4),
# l_shoulder(5), r_shoulder(6), ...
def _make_kp(nose=(0.5, 0.5, 0.9), l_eye=None, r_eye=None,
             l_ear=None, r_ear=None, l_sh=None, r_sh=None):
    kp = [(0.0, 0.0, 0.0)] * 17
    kp[0] = nose
    kp[1] = l_eye or (0.0, 0.0, 0.0)
    kp[2] = r_eye or (0.0, 0.0, 0.0)
    kp[3] = l_ear or (0.0, 0.0, 0.0)
    kp[4] = r_ear or (0.0, 0.0, 0.0)
    kp[5] = l_sh or (0.0, 0.0, 0.0)
    kp[6] = r_sh or (0.0, 0.0, 0.0)
    return kp


def test_gaze_target_front_when_eyes_balanced():
    """양 눈 보이고 nose가 eye보다 살짝 아래 → front."""
    clf = GazeTargetClassifier(fps=5, window_sec=2.5)
    # eye_dist = 0.1, nose가 eye보다 0.02 아래 → pitch = 0.2 (front 범위)
    kp = _make_kp(
        nose=(0.5, 0.52, 0.9),
        l_eye=(0.45, 0.50, 0.9), r_eye=(0.55, 0.50, 0.9),
    )
    # 윈도우 채우기
    for _ in range(10):
        result = clf.process(kp)
    assert result == "front"


def test_gaze_target_down_when_nose_far_below_eyes():
    """nose가 eye보다 크게 아래 → down (핸드폰/책상)."""
    clf = GazeTargetClassifier(fps=5, window_sec=2.5)
    # eye_dist = 0.1, nose가 eye보다 0.10 아래 → pitch = 1.0 (down 범위)
    kp = _make_kp(
        nose=(0.5, 0.60, 0.9),
        l_eye=(0.45, 0.50, 0.9), r_eye=(0.55, 0.50, 0.9),
    )
    for _ in range(10):
        result = clf.process(kp)
    assert result == "down"


def test_gaze_target_side_when_only_one_eye_visible():
    """한쪽 눈만 신뢰도 높음 → side."""
    clf = GazeTargetClassifier(fps=5, window_sec=2.5)
    kp = _make_kp(
        nose=(0.5, 0.5, 0.9),
        l_eye=(0.45, 0.50, 0.9),
        r_eye=(0.55, 0.50, 0.05),   # 신뢰도 낮음 — 안 보임
    )
    for _ in range(10):
        result = clf.process(kp)
    assert result == "side"


def test_gaze_target_none_when_no_keypoints():
    clf = GazeTargetClassifier()
    assert clf.process(None) is None


def test_gaze_target_window_majority_vote():
    """대다수가 front면 1~2 down 노이즈 있어도 front 유지."""
    clf = GazeTargetClassifier(fps=5, window_sec=2.5)
    front_kp = _make_kp(
        nose=(0.5, 0.52, 0.9),
        l_eye=(0.45, 0.50, 0.9), r_eye=(0.55, 0.50, 0.9),
    )
    down_kp = _make_kp(
        nose=(0.5, 0.60, 0.9),
        l_eye=(0.45, 0.50, 0.9), r_eye=(0.55, 0.50, 0.9),
    )
    # front 8회, down 2회 — 윈도우는 12 frame
    for _ in range(8):
        clf.process(front_kp)
    for _ in range(2):
        clf.process(down_kp)
    # majority는 front
    result = clf.process(front_kp)
    assert result == "front"


# === activity_monitor ===

def test_activity_classify_still():
    assert activity_monitor._classify(0.001) == "still"


def test_activity_classify_focused():
    assert activity_monitor._classify(0.008) == "focused"


def test_activity_classify_normal():
    assert activity_monitor._classify(0.02) == "normal"


def test_activity_classify_restless():
    assert activity_monitor._classify(0.06) == "restless"


def test_activity_sample_returns_none_for_low_conf():
    """confidence 낮으면 None."""
    kp = _make_kp(
        nose=(0.5, 0.5, 0.05),   # low conf
        l_sh=(0.4, 0.7, 0.9), r_sh=(0.6, 0.7, 0.9),
    )
    assert activity_monitor._sample(kp) is None


def test_activity_sample_returns_tuple_for_valid_kp():
    kp = _make_kp(
        nose=(0.5, 0.5, 0.9),
        l_sh=(0.4, 0.7, 0.9), r_sh=(0.6, 0.7, 0.9),
    )
    sample = activity_monitor._sample(kp)
    assert sample is not None
    assert sample == (0.5, 0.5, 0.5, 0.7)


def test_activity_compute_std_mean_constant_is_zero():
    samples = [(0.5, 0.5, 0.5, 0.7)] * 6
    assert activity_monitor._compute_std_mean(samples) < 1e-10


def test_activity_compute_std_mean_varies():
    """변동 있는 샘플은 양수."""
    samples = [
        (0.50, 0.50, 0.50, 0.70),
        (0.55, 0.55, 0.55, 0.75),
        (0.45, 0.45, 0.45, 0.65),
        (0.50, 0.50, 0.50, 0.70),
    ]
    std = activity_monitor._compute_std_mean(samples)
    assert std > 0.01


# === PostureReading.category ===

def test_posture_category_upright():
    r = PostureReading(neck_angle_deg=10, shoulder_tilt_deg=5, timestamp=time.time())
    assert r.category == "upright"


def test_posture_category_slouched():
    r = PostureReading(neck_angle_deg=30, shoulder_tilt_deg=5, timestamp=time.time())
    assert r.category == "slouched"


def test_posture_category_leaning():
    r = PostureReading(neck_angle_deg=10, shoulder_tilt_deg=20, timestamp=time.time())
    assert r.category == "leaning"


def test_posture_slouched_dominates_leaning():
    """둘 다 안 좋으면 slouched 우선 (목 문제 더 중요)."""
    r = PostureReading(neck_angle_deg=30, shoulder_tilt_deg=20, timestamp=time.time())
    assert r.category == "slouched"
