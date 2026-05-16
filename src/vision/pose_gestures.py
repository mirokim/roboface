"""포즈 keypoint 기반 단순 제스처 감지 — 양손 만세 / 고개 끄덕임 / 도리도리.

각 감지기는 .process(keypoints)로 호출 → 감지 시 True. cooldown 내장.
손 흔들기는 별도 WristWaveDetector 사용 (진폭/zero crossings 로직 더 복잡).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger
from src.vision.camera import (
    KP_L_SHOULDER, KP_L_WRIST,
    KP_NOSE, KP_R_SHOULDER, KP_R_WRIST,
)

log = get_logger("pose_gestures")


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


# ─── 양손 만세 ───

class HandsUpDetector:
    """양 손목이 코보다 위로 올라간 상태로 hold_sec 동안 유지 → 만세 감지.

    축하/신남/요청 등 신호. cooldown_sec 동안 재발동 차단.
    """

    KP_CONF_THRESHOLD = 0.10

    def __init__(
        self,
        fps: float = 5.0,
        hold_sec: float = 0.6,
        cooldown_sec: float = 5.0,
    ) -> None:
        self.required_frames = max(2, int(fps * hold_sec))
        self.cooldown_sec = cooldown_sec
        self._consecutive = 0
        self._last_at = 0.0

    def reset(self) -> None:
        self._consecutive = 0

    def process(self, keypoints: Any) -> bool:
        if keypoints is None:
            self._consecutive = 0
            return False
        if time.time() - self._last_at < self.cooldown_sec:
            return False
        nose = keypoints[KP_NOSE]
        l_wrist = keypoints[KP_L_WRIST]
        r_wrist = keypoints[KP_R_WRIST]
        if (
            nose[2] < self.KP_CONF_THRESHOLD
            or l_wrist[2] < self.KP_CONF_THRESHOLD
            or r_wrist[2] < self.KP_CONF_THRESHOLD
        ):
            self._consecutive = 0
            return False
        # 양 손목이 코보다 위 (y 더 작음)
        if l_wrist[1] < nose[1] and r_wrist[1] < nose[1]:
            self._consecutive += 1
            if self._consecutive >= self.required_frames:
                log.info("🙌 양손 만세 감지!")
                self._last_at = time.time()
                self._consecutive = 0
                return True
        else:
            self._consecutive = 0
        return False


# ─── 고개 끄덕임 / 도리도리 ───

class _HeadOscillationDetector:
    """공통 — 코 좌표 한 축의 진동 감지. 끄덕임(y)/도리도리(x) 공용."""

    KP_CONF_THRESHOLD = 0.15

    def __init__(
        self,
        axis: int,   # 0 = x (도리도리), 1 = y (끄덕임)
        fps: float = 5.0,
        history_sec: float = 1.2,
        cooldown_sec: float = 4.0,
        min_amp: float = 0.025,
        max_amp: float = 0.15,
        min_zc: int = 3,
        max_zc: int = 8,
    ) -> None:
        self.axis = axis
        self.history_max = max(6, int(fps * history_sec))
        self.cooldown_sec = cooldown_sec
        self.min_amp = min_amp
        self.max_amp = max_amp
        self.min_zc = min_zc
        self.max_zc = max_zc
        self.history: deque[float] = deque(maxlen=self.history_max)
        self._last_at = 0.0
        # 어깨너비로 정규화하기 위해 매 프레임 어깨 정보도 참고
        self._shoulder_widths: deque[float] = deque(maxlen=self.history_max)

    def reset(self) -> None:
        self.history.clear()
        self._shoulder_widths.clear()

    def process(self, keypoints: Any) -> bool:
        if keypoints is None:
            return False
        if time.time() - self._last_at < self.cooldown_sec:
            return False
        nose = keypoints[KP_NOSE]
        l_sh = keypoints[KP_L_SHOULDER]
        r_sh = keypoints[KP_R_SHOULDER]
        if (
            nose[2] < self.KP_CONF_THRESHOLD
            or l_sh[2] < self.KP_CONF_THRESHOLD
            or r_sh[2] < self.KP_CONF_THRESHOLD
        ):
            return False
        sw = float(abs(l_sh[0] - r_sh[0]))
        if sw < 0.02:
            return False
        self.history.append(float(nose[self.axis]))
        self._shoulder_widths.append(sw)
        if len(self.history) < self.history_max:
            return False
        np = _get_numpy()
        if np is None:
            return False
        arr = np.fromiter(self.history, dtype=np.float32)
        amp = float(arr.max() - arr.min())
        sw_med = float(np.median(np.fromiter(
            self._shoulder_widths, dtype=np.float32,
        )))
        # 어깨너비 대비로 정규화 (거리 무관)
        amp_ratio = amp / sw_med if sw_med > 0 else 0.0
        if amp_ratio < self.min_amp or amp_ratio > self.max_amp:
            return False
        median = float(np.median(arr))
        signs = np.sign(arr - median)
        zc = int(np.sum(np.abs(np.diff(signs)) > 0))
        if not (self.min_zc <= zc <= self.max_zc):
            return False
        self._last_at = time.time()
        self.history.clear()
        self._shoulder_widths.clear()
        return True


class HeadNodDetector(_HeadOscillationDetector):
    """고개 끄덕임 (y 진동) → yes/긍정."""

    def __init__(self, fps: float = 5.0) -> None:
        super().__init__(axis=1, fps=fps)

    def process(self, keypoints: Any) -> bool:
        if super().process(keypoints):
            log.info("👍 고개 끄덕임 감지!")
            return True
        return False


class HeadShakeDetector(_HeadOscillationDetector):
    """고개 도리도리 (x 진동) → no/부정."""

    def __init__(self, fps: float = 5.0) -> None:
        super().__init__(axis=0, fps=fps)

    def process(self, keypoints: Any) -> bool:
        if super().process(keypoints):
            log.info("🙅 고개 도리도리 감지!")
            return True
        return False
