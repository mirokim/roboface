"""손 흔들기 감지 — bbox 위쪽 영역의 motion centroid가 좌우로 oscillation.

알고리즘:
1. 매 프레임 사람 bbox의 위쪽 60% (얼굴 + 어깨 + 팔) crop
2. grayscale로 변환 + 다운샘플
3. 이전 프레임과의 |diff| → motion mask
4. 컬럼별 motion 합 → motion centroid x (가중평균)
5. centroid x를 시계열에 push (history_sec 분량)
6. 시계열의 (a) 진폭 ≥ min_amplitude + (b) zero crossings가 범위 안 → wave!

False positive 방지:
- 감지 후 cooldown_sec 동안 push/평가 둘 다 skip (잔류 노이즈 차단)
- bbox 크게 바뀌면 (예: 사람이 들어옴) prev 초기화
- motion 너무 적으면 push 안 함 (정적 frame이 평균 끌어내림 방지)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger

log = get_logger("wave_detector")


def _get_numpy():
    """numpy lazy import — robot 모드에서만 필요, 한 번 import 후 캐시."""
    if _get_numpy._cached is not None:
        return _get_numpy._cached
    try:
        import numpy as np
        _get_numpy._cached = np
        return np
    except ImportError:
        return None


_get_numpy._cached = None  # type: ignore[attr-defined]


class WaveDetector:
    def __init__(
        self,
        fps: float = 5.0,
        history_sec: float = 1.5,
        cooldown_sec: float = 5.0,
        min_motion_pixels: int = 30,
        diff_threshold: int = 15,
        min_amplitude: float = 0.12,
        min_zero_crossings: int = 3,
        max_zero_crossings: int = 16,
        min_eval_frames: int = 6,
    ) -> None:
        self.fps = fps
        self.history_max = max(min_eval_frames, int(fps * history_sec))
        self.min_eval_frames = min_eval_frames
        self.cooldown_sec = cooldown_sec
        self.min_motion_pixels = min_motion_pixels
        self.diff_threshold = diff_threshold
        self.min_amplitude = min_amplitude
        self.min_zero_crossings = min_zero_crossings
        self.max_zero_crossings = max_zero_crossings

        self._prev_gray: Any = None
        self._prev_bbox: tuple[float, float, float, float] | None = None
        self.centroid_history: deque[float] = deque(maxlen=self.history_max)
        self.last_wave_at = 0.0
        self._last_debug_log_at = 0.0

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_bbox = None
        self.centroid_history.clear()

    def process(
        self,
        frame: Any,
        bbox: tuple[float, float, float, float] | None,
    ) -> bool:
        """frame: numpy HxWx{3 or 4} or HxW. bbox: 정규화 0~1. 감지되면 True."""
        # 프레임 한두 개 빠져도 history는 유지 (vision_task 1.5초 grace로 보완).
        # 정말로 사라지면 vision_task에서 wave_detector.reset() 명시적 호출.
        if frame is None or bbox is None:
            return False

        np = _get_numpy()
        if np is None:
            return False

        # 감지 직후 cooldown — push/평가 모두 skip해 잔류 노이즈 차단
        if time.time() - self.last_wave_at < self.cooldown_sec:
            return False

        # bbox 크게 바뀌면 prev 초기화 (사람 다시 들어옴)
        if self._prev_bbox is not None:
            dx = abs(bbox[0] - self._prev_bbox[0]) + abs(bbox[2] - self._prev_bbox[2])
            if dx > 0.25:
                log.debug(f"wave reset — bbox 점프 dx={dx:.2f}")
                self.reset()
        self._prev_bbox = bbox

        h, w = frame.shape[:2]
        x0, y0, x1, y1 = bbox
        # 위쪽 60% — 얼굴/어깨/팔 영역 (다리는 wave에 관련 없음)
        upper_y1 = y0 + (y1 - y0) * 0.6
        px0, py0 = int(max(0.0, x0) * w), int(max(0.0, y0) * h)
        px1 = int(min(1.0, x1) * w)
        py1 = int(min(1.0, upper_y1) * h)
        if px1 - px0 < 24 or py1 - py0 < 24:
            self.reset()
            return False

        crop = frame[py0:py1, px0:px1]
        if crop.ndim == 3:
            # RGB → grayscale (luminance approximate)
            gray = (
                crop[..., 0].astype(np.int16)
                + crop[..., 1].astype(np.int16) * 2
                + crop[..., 2].astype(np.int16)
            ) // 4
            gray = gray.astype(np.uint8)
        else:
            gray = crop
        # 다운샘플 (속도 + 노이즈 감소)
        gray = gray[::2, ::2]

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return False

        diff = np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))
        self._prev_gray = gray

        mask = diff > self.diff_threshold
        total_motion = int(mask.sum())

        # 진단 로그 — 2초마다 1회 (튜닝 끝나면 LOG_LEVEL=DEBUG로 끔)
        now = time.time()
        if now - self._last_debug_log_at > 2.0:
            self._last_debug_log_at = now
            log.debug(
                f"wave proc motion={total_motion} (≥{self.min_motion_pixels} 필요) "
                f"history={len(self.centroid_history)}/{self.history_max}"
            )

        if total_motion < self.min_motion_pixels:
            # 정적 — push 안 함 (시계열에 noise 끼지 않게)
            return False

        # column-wise motion sum → centroid x (0~1 normalized)
        col_sum = mask.sum(axis=0).astype(np.float32)
        col_total = float(col_sum.sum())
        if col_total <= 0.0:
            return False
        xs = np.arange(col_sum.shape[0], dtype=np.float32)
        centroid = float((xs * col_sum).sum() / col_total / col_sum.shape[0])

        self.centroid_history.append(centroid)

        # 최소 평가 가능 프레임 수 모이면 시도
        if len(self.centroid_history) < self.min_eval_frames:
            return False

        return self._evaluate()

    def _evaluate(self) -> bool:
        np = _get_numpy()
        if np is None:
            return False
        arr = np.fromiter(self.centroid_history, dtype=np.float32)
        amp = float(arr.max() - arr.min())
        median = float(np.median(arr))
        signs = np.sign(arr - median)
        zero_crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

        log.debug(
            f"wave eval amp={amp:.3f} zc={zero_crossings} "
            f"(amp≥{self.min_amplitude}, zc {self.min_zero_crossings}~"
            f"{self.max_zero_crossings})"
        )

        if amp < self.min_amplitude:
            return False
        if self.min_zero_crossings <= zero_crossings <= self.max_zero_crossings:
            log.info(
                f"👋 wave 감지! amp={amp:.2f} zero_crossings={zero_crossings}"
            )
            self.last_wave_at = time.time()
            self.centroid_history.clear()
            self._prev_gray = None  # 다음 프레임은 새 베이스라인
            return True
        return False
