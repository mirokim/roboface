"""손 흔들기 감지 (PoseNet 기반) — 손목 좌표 x의 좌우 oscillation.

기존 motion centroid 방식보다 훨씬 정확:
- 손목 x좌표 직접 트래킹 (배경/전신 모션 영향 X)
- 어깨 너비로 정규화 — 거리에 무관한 진폭 비교
- 손이 어깨 위로 올라온 경우만 카운트 (인사 wave는 손을 들고 하는 동작)

알고리즘:
1. 매 프레임 keypoints에서 좌/우 손목 (idx 9, 10) + 좌/우 어깨 (5, 6) 추출
2. 손목이 어깨보다 위에 있고 confidence 충분하면 push (정규화 x좌표)
3. 양손 중 더 active한 쪽 (분산 큰 쪽) 시계열로 평가
4. 진폭 ≥ min_amplitude * 어깨너비 + zero crossings 범위 안 → wave!
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger
from src.vision.camera import KP_L_SHOULDER, KP_L_WRIST, KP_R_SHOULDER, KP_R_WRIST

log = get_logger("wrist_wave")


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


class WristWaveDetector:
    """손목 trajectory 기반 wave 감지.

    Pose model이 frame당 인물별 (17, 3) keypoints 반환. process(keypoints)로
    호출. 같은 인스턴스가 좌/우 손목 두 시계열 동시 관리.
    """

    KP_CONF_THRESHOLD = 0.10   # 손목/어깨 keypoint 최소 신뢰도 (HigherHRNet 분포 낮음)

    def __init__(
        self,
        fps: float = 5.0,
        history_sec: float = 1.5,
        cooldown_sec: float = 5.0,
        # 진폭은 어깨너비의 배수 — 거리 무관. 0.5 = 어깨너비 절반만큼 손이 흔들림
        min_amplitude_ratio: float = 0.4,
        min_zero_crossings: int = 3,
        max_zero_crossings: int = 16,
        min_eval_frames: int = 6,
    ) -> None:
        self.fps = fps
        self.history_max = max(min_eval_frames, int(fps * history_sec))
        self.min_eval_frames = min_eval_frames
        self.cooldown_sec = cooldown_sec
        self.min_amplitude_ratio = min_amplitude_ratio
        self.min_zero_crossings = min_zero_crossings
        self.max_zero_crossings = max_zero_crossings

        # 좌/우 손목 각각 별도 시계열
        self.left_history: deque[float] = deque(maxlen=self.history_max)
        self.right_history: deque[float] = deque(maxlen=self.history_max)
        # 평가 시 정규화 기준 (어깨너비 또는 fallback)
        self.shoulder_width_history: deque[float] = deque(maxlen=self.history_max)
        self.last_wave_at = 0.0
        self._last_debug_log_at = 0.0

    def reset(self) -> None:
        self.left_history.clear()
        self.right_history.clear()
        self.shoulder_width_history.clear()

    def diag(self) -> dict:
        """현재 진행 상황 — vision_task가 5초마다 로그로 출력."""
        np = _get_numpy()
        l_amp = r_amp = -1.0
        if np is not None and len(self.left_history) >= self.min_eval_frames:
            arr = np.fromiter(self.left_history, dtype=np.float32)
            l_amp = float(arr.max() - arr.min())
        if np is not None and len(self.right_history) >= self.min_eval_frames:
            arr = np.fromiter(self.right_history, dtype=np.float32)
            r_amp = float(arr.max() - arr.min())
        sw_med = -1.0
        if np is not None and self.shoulder_width_history:
            sw_med = float(np.median(np.fromiter(
                self.shoulder_width_history, dtype=np.float32,
            )))
        return {
            "left": len(self.left_history),
            "right": len(self.right_history),
            "need": self.min_eval_frames,
            "max": self.history_max,
            "l_amp_ratio": l_amp / sw_med if sw_med > 0 and l_amp > 0 else 0.0,
            "r_amp_ratio": r_amp / sw_med if sw_med > 0 and r_amp > 0 else 0.0,
            "need_amp": self.min_amplitude_ratio,
            "cooldown_remain": max(
                0.0, self.cooldown_sec - (time.time() - self.last_wave_at),
            ),
        }

    def process(self, keypoints: Any) -> bool:
        """keypoints: shape (17, 3) — (x, y, conf), 0~1 정규화. 감지되면 True."""
        if keypoints is None:
            return False
        np = _get_numpy()
        if np is None:
            return False

        # 쿨다운 — push도 skip해서 잔류 노이즈 차단
        if time.time() - self.last_wave_at < self.cooldown_sec:
            return False

        l_shoulder = keypoints[KP_L_SHOULDER]
        r_shoulder = keypoints[KP_R_SHOULDER]
        l_wrist = keypoints[KP_L_WRIST]
        r_wrist = keypoints[KP_R_WRIST]

        shoulder_ok = (
            l_shoulder[2] >= self.KP_CONF_THRESHOLD
            and r_shoulder[2] >= self.KP_CONF_THRESHOLD
        )

        # 진단 — 2초마다 keypoint conf + y 좌표 + history (DEBUG)
        now_t = time.time()
        if now_t - self._last_debug_log_at > 2.0:
            self._last_debug_log_at = now_t
            sh_y_show = (
                (l_shoulder[1] + r_shoulder[1]) / 2 if shoulder_ok else -1
            )
            log.debug(
                f"wave kp: shoulder L={l_shoulder[2]:.2f} R={r_shoulder[2]:.2f} "
                f"(y={sh_y_show:.2f}) | "
                f"wrist L={l_wrist[2]:.2f}(y={l_wrist[1]:.2f}) "
                f"R={r_wrist[2]:.2f}(y={r_wrist[1]:.2f}) "
                f"hist L={len(self.left_history)} R={len(self.right_history)}"
            )

        if not shoulder_ok:
            return False

        shoulder_width = float(abs(l_shoulder[0] - r_shoulder[0]))
        if shoulder_width < 0.02:   # 너무 좁음 — 옆모습이거나 노이즈
            return False
        shoulder_y = float((l_shoulder[1] + r_shoulder[1]) / 2)
        # 다리 모션 오인 방지. 어깨~가슴 정도까지 (정규화 0.4 = 화면 40%)는 wave 허용.
        max_below = 0.4

        # 손목 push: confidence 통과 AND 손목이 다리 높이는 아닐 때
        pushed_any = False
        for wrist, history in (
            (l_wrist, self.left_history),
            (r_wrist, self.right_history),
        ):
            if wrist[2] < self.KP_CONF_THRESHOLD:
                continue
            if wrist[1] > shoulder_y + max_below:   # 다리 높이 — 무시
                continue
            history.append(float(wrist[0]))
            pushed_any = True

        if pushed_any:
            self.shoulder_width_history.append(shoulder_width)

        # 둘 중 하나라도 평가 가능 프레임 수 모이면 시도
        candidates = [
            ("left", self.left_history),
            ("right", self.right_history),
        ]
        for side, history in candidates:
            if len(history) >= self.min_eval_frames:
                if self._evaluate(side, history, np):
                    return True
        return False

    def _evaluate(self, side: str, history: deque, np) -> bool:
        if not self.shoulder_width_history:
            return False
        # 어깨너비 최근 median으로 정규화 기준
        sw_median = float(np.median(np.fromiter(
            self.shoulder_width_history, dtype=np.float32,
        )))
        if sw_median < 0.02:
            return False
        arr = np.fromiter(history, dtype=np.float32)
        amp = float(arr.max() - arr.min())
        amp_ratio = amp / sw_median   # 어깨너비 대비 흔들림 폭

        median = float(np.median(arr))
        signs = np.sign(arr - median)
        zero_crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

        # 진단 — 2초마다 한 번
        now = time.time()
        if now - self._last_debug_log_at > 2.0:
            self._last_debug_log_at = now
            log.debug(
                f"wave eval [{side}] amp_ratio={amp_ratio:.2f} zc={zero_crossings} "
                f"(amp_ratio≥{self.min_amplitude_ratio}, "
                f"zc {self.min_zero_crossings}~{self.max_zero_crossings})"
            )

        if amp_ratio < self.min_amplitude_ratio:
            return False
        if self.min_zero_crossings <= zero_crossings <= self.max_zero_crossings:
            log.info(
                f"👋 wave 감지! [{side}] amp_ratio={amp_ratio:.2f} "
                f"zc={zero_crossings}"
            )
            self.last_wave_at = time.time()
            # 한쪽 감지되면 양쪽 history 다 클리어 (다음 wave 베이스라인 새로)
            self.left_history.clear()
            self.right_history.clear()
            self.shoulder_width_history.clear()
            return True
        return False
