"""손 흔들기 감지 — bbox 위쪽 영역의 motion centroid가 좌우로 oscillation.

알고리즘:
1. 매 프레임 사람 bbox의 위쪽 60% (얼굴 + 어깨 + 팔) crop
2. grayscale로 변환 + 다운샘플
3. 이전 프레임과의 |diff| → motion mask
4. 컬럼별 motion 합 → motion centroid x (가중평균)
5. centroid x를 시계열에 push (~3초 buffer)
6. 시계열의 (a) 진폭 충분히 큼 + (b) zero crossings 4~12회 → wave!

False positive 방지:
- 감지 후 5초 cooldown
- bbox 크게 바뀌면 (예: 사람이 들어옴) prev 초기화
- motion 너무 적으면 push 안 함 (정적 frame이 평균 끌어내림 방지)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger

log = get_logger("wave_detector")


class WaveDetector:
    def __init__(
        self,
        fps: float = 5.0,
        history_sec: float = 1.5,
        cooldown_sec: float = 5.0,
        min_motion_pixels: int = 60,
        diff_threshold: int = 18,
        min_amplitude: float = 0.08,
        min_zero_crossings: int = 2,
        max_zero_crossings: int = 16,
    ) -> None:
        self.fps = fps
        self.history_max = max(8, int(fps * history_sec))
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
        if frame is None or bbox is None:
            self.reset()
            return False

        # numpy는 robot 모드에서만 사용 — 지연 import
        try:
            import numpy as np
        except ImportError:
            return False

        # bbox 크게 바뀌면 prev 초기화 (사람 다시 들어옴)
        if self._prev_bbox is not None:
            dx = abs(bbox[0] - self._prev_bbox[0]) + abs(bbox[2] - self._prev_bbox[2])
            if dx > 0.25:
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

        # 충분히 모이지 않았으면 판단 보류
        if len(self.centroid_history) < self.history_max:
            return False

        # cooldown
        if time.time() - self.last_wave_at < self.cooldown_sec:
            return False

        return self._evaluate(np)

    def _evaluate(self, np) -> bool:
        arr = np.fromiter(self.centroid_history, dtype=np.float32)
        amp = float(arr.max() - arr.min())
        median = float(np.median(arr))
        signs = np.sign(arr - median)
        zero_crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

        # 튜닝용 — 2초마다 현재 측정값 INFO로 찍어줌
        now = time.time()
        if now - self._last_debug_log_at > 2.0:
            self._last_debug_log_at = now
            log.info(
                f"wave 후보 amp={amp:.3f} zero_crossings={zero_crossings} "
                f"(필요: amp≥{self.min_amplitude}, "
                f"zc {self.min_zero_crossings}~{self.max_zero_crossings})"
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
