"""Pose 안정화 — keypoint 스무딩 + 사람 lock 컨센서스.

HigherHRNet 출력이 신뢰도 낮고 프레임마다 흔들림 — 그대로 쓰면 제스처
감지가 노이즈에 약함. 두 층으로 안정화:

1. 스무딩: 최근 N프레임 keypoint를 신뢰도 가중 평균. 떨림 완화.
2. Lock: 최근 1초간 평균 detection score가 threshold 이상 → 사람 lock.
   lock된 상태에서만 제스처 emit (false positive 차단).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger

log = get_logger("pose_stab")


def _get_numpy():
    if _get_numpy._cached is not None:
        return _get_numpy._cached
    try:
        import numpy as np
        _get_numpy._cached = np
        return np
    except ImportError:
        return None


_get_numpy._cached = None  # type: ignore[attr-defined]


class PoseStabilizer:
    """매 프레임 keypoints/score 받아 스무딩 + lock 상태 관리."""

    def __init__(
        self,
        smoothing_frames: int = 3,
        lock_window_sec: float = 1.0,
        lock_min_score: float = 0.12,
        fps: float = 10.0,
    ) -> None:
        self.smoothing_frames = max(1, smoothing_frames)
        self.lock_min_score = lock_min_score
        score_buf = max(2, int(lock_window_sec * fps))
        self._kp_history: deque[Any] = deque(maxlen=self.smoothing_frames)
        self._score_history: deque[float] = deque(maxlen=score_buf)
        self._last_smoothed: Any = None

    def reset(self) -> None:
        self._kp_history.clear()
        self._score_history.clear()
        self._last_smoothed = None

    def update(self, keypoints: Any, score: float) -> Any:
        """이번 프레임 데이터 push 후 스무딩된 keypoints 반환 (없으면 None)."""
        if keypoints is None:
            self._score_history.append(0.0)
            return self._last_smoothed
        self._kp_history.append(keypoints)
        self._score_history.append(float(score))
        self._last_smoothed = self._smooth()
        return self._last_smoothed

    @property
    def is_locked(self) -> bool:
        """최근 lock window 평균 score가 threshold 이상일 때 True.

        충분한 샘플 수(maxlen 절반 이상) 모인 후에만 판정.
        """
        if len(self._score_history) < max(2, self._score_history.maxlen // 2):
            return False
        avg = sum(self._score_history) / len(self._score_history)
        return avg >= self.lock_min_score

    @property
    def smoothed(self) -> Any:
        return self._last_smoothed

    def _smooth(self) -> Any:
        np = _get_numpy()
        if np is None or len(self._kp_history) == 0:
            return None
        if len(self._kp_history) == 1:
            return self._kp_history[0]
        # (N, 17, 3) — 신뢰도를 가중치로 x, y 가중평균
        stack = np.stack(list(self._kp_history), axis=0).astype(np.float32)
        weights = stack[:, :, 2]   # (N, 17)
        weight_sum = weights.sum(axis=0)
        safe = np.maximum(weight_sum, 1e-6)
        wx = (stack[:, :, 0] * weights).sum(axis=0) / safe
        wy = (stack[:, :, 1] * weights).sum(axis=0) / safe
        mean_conf = weights.mean(axis=0)
        smoothed = np.stack([wx, wy, mean_conf], axis=1).astype(np.float32)
        return smoothed
