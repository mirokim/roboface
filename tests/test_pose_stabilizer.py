"""PoseStabilizer — 스무딩 + lock 컨센서스 테스트."""

from __future__ import annotations

import numpy as np

from src.vision.pose_stabilizer import PoseStabilizer


def _kps(x: float, y: float, conf: float = 0.5) -> np.ndarray:
    """간단 keypoints — 모든 관절을 같은 위치/신뢰도로."""
    arr = np.zeros((17, 3), dtype=np.float32)
    arr[:, 0] = x
    arr[:, 1] = y
    arr[:, 2] = conf
    return arr


# ─── Smoothing ───

def test_smoothing_averages_position():
    stab = PoseStabilizer(smoothing_frames=3, fps=10.0)
    stab.update(_kps(0.0, 0.0), 0.5)
    stab.update(_kps(0.2, 0.2), 0.5)
    out = stab.update(_kps(0.4, 0.4), 0.5)
    # 평균 (0+0.2+0.4)/3 = 0.2
    assert abs(out[0, 0] - 0.2) < 0.01
    assert abs(out[0, 1] - 0.2) < 0.01


def test_smoothing_weighted_by_confidence():
    stab = PoseStabilizer(smoothing_frames=3, fps=10.0)
    # 신뢰도 0.1 짜리 끼어들면 거의 무시되어야
    stab.update(_kps(0.0, 0.0, conf=0.9), 0.9)
    stab.update(_kps(0.5, 0.5, conf=0.1), 0.5)   # 거의 무시
    stab.update(_kps(0.0, 0.0, conf=0.9), 0.9)
    out = stab.smoothed
    # 0.5 쪽 가중치 매우 낮으므로 결과 ~0.0에 가까움
    assert out[0, 0] < 0.1


def test_single_frame_returns_as_is():
    stab = PoseStabilizer(smoothing_frames=3, fps=10.0)
    kps = _kps(0.3, 0.5, conf=0.8)
    out = stab.update(kps, 0.5)
    assert np.array_equal(out, kps)


def test_none_keypoints_returns_last_smoothed():
    stab = PoseStabilizer(smoothing_frames=3, fps=10.0)
    stab.update(_kps(0.0, 0.0), 0.5)
    last = stab.update(_kps(0.2, 0.2), 0.5)
    out = stab.update(None, 0.0)
    assert np.array_equal(out, last)


# ─── Lock ───

def test_lock_requires_enough_samples():
    stab = PoseStabilizer(lock_window_sec=1.0, fps=10.0)  # 10샘플 버퍼
    # 1~2개로는 lock 안 됨
    stab.update(_kps(0.5, 0.5), 0.5)
    assert not stab.is_locked


def test_lock_true_when_avg_above_threshold():
    stab = PoseStabilizer(lock_window_sec=1.0, lock_min_score=0.15, fps=10.0)
    # 10프레임 모두 score 0.3 → 평균 0.3 ≥ 0.15
    for _ in range(10):
        stab.update(_kps(0.5, 0.5), 0.3)
    assert stab.is_locked


def test_lock_false_when_avg_below_threshold():
    stab = PoseStabilizer(lock_window_sec=1.0, lock_min_score=0.20, fps=10.0)
    for _ in range(10):
        stab.update(_kps(0.5, 0.5), 0.10)
    assert not stab.is_locked


def test_lock_drops_when_person_leaves():
    stab = PoseStabilizer(lock_window_sec=1.0, lock_min_score=0.15, fps=10.0)
    # 잠깐 잡힘
    for _ in range(10):
        stab.update(_kps(0.5, 0.5), 0.3)
    assert stab.is_locked
    # 이후 10프레임 사람 없음 → 평균이 0으로 수렴
    for _ in range(15):
        stab.update(None, 0.0)
    assert not stab.is_locked


def test_reset_clears_lock_and_history():
    stab = PoseStabilizer(fps=10.0)
    for _ in range(10):
        stab.update(_kps(0.5, 0.5), 0.5)
    stab.reset()
    assert not stab.is_locked
    assert stab.smoothed is None
