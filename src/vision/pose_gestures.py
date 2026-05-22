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
    KP_L_EAR, KP_L_EYE, KP_L_SHOULDER, KP_L_WRIST,
    KP_NOSE, KP_R_EAR, KP_R_EYE, KP_R_SHOULDER, KP_R_WRIST,
)


# ─── 얼굴 방향 판정 ───

def face_orientation(keypoints, kp_conf_threshold: float = 0.20) -> str:
    """keypoints의 좌/우 눈·귀 신뢰도 차이로 얼굴 방향 추정.

    return: "front" / "side_left" / "side_right" / "away" / "unknown"
    - front: 양쪽 눈/귀 비교적 균형. 정면 응시.
    - side_left: 사용자 기준 왼쪽 (카메라 기준 오른쪽 눈/귀만 보임)
    - side_right: 사용자 기준 오른쪽 (왼쪽 눈/귀만 보임)
    - away: 코/눈 자체 신뢰도 낮음 (뒤돌거나 가려짐)
    - unknown: keypoints 없거나 어깨도 안 보임
    """
    if keypoints is None:
        return "unknown"
    nose = keypoints[KP_NOSE]
    l_eye = keypoints[KP_L_EYE]
    r_eye = keypoints[KP_R_EYE]
    l_ear = keypoints[KP_L_EAR]
    r_ear = keypoints[KP_R_EAR]
    l_sh = keypoints[KP_L_SHOULDER]
    r_sh = keypoints[KP_R_SHOULDER]
    if l_sh[2] < kp_conf_threshold and r_sh[2] < kp_conf_threshold:
        return "unknown"

    # 코 + 두 눈 모두 신뢰도 충분 → 정면
    eyes_ok = (l_eye[2] >= kp_conf_threshold and r_eye[2] >= kp_conf_threshold)
    nose_ok = nose[2] >= kp_conf_threshold

    if nose_ok and eyes_ok:
        # 양 눈 사이 거리 vs 눈-귀 거리 비율로도 확인 가능. 일단 단순 판정.
        return "front"

    # 한쪽 눈/귀만 명확 → 옆 모습
    left_side_visible = (l_eye[2] >= kp_conf_threshold or l_ear[2] >= kp_conf_threshold)
    right_side_visible = (r_eye[2] >= kp_conf_threshold or r_ear[2] >= kp_conf_threshold)
    if left_side_visible and not right_side_visible:
        # 카메라가 사용자의 왼쪽 보임 → 사용자가 자기 오른쪽으로 고개 돌림
        return "side_right"
    if right_side_visible and not left_side_visible:
        return "side_left"

    # 어깨는 있지만 얼굴 keypoints 거의 없음 → 뒤돌거나 매우 멀리
    return "away"

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
        hold_sec: float = 0.35,   # 0.6 → 0.35 — 빨리 잡히게
        cooldown_sec: float = 5.0,
    ) -> None:
        self.required_frames = max(2, int(fps * hold_sec))
        self.cooldown_sec = cooldown_sec
        self._consecutive = 0
        self._last_at = 0.0

    def reset(self) -> None:
        self._consecutive = 0

    def diag(self) -> dict:
        return {
            "consecutive": self._consecutive,
            "need": self.required_frames,
            "cooldown_remain": max(
                0.0, self.cooldown_sec - (time.time() - self._last_at),
            ),
        }

    def process(self, keypoints: Any) -> bool:
        if keypoints is None:
            self._consecutive = 0
            return False
        if time.time() - self._last_at < self.cooldown_sec:
            return False
        l_wrist = keypoints[KP_L_WRIST]
        r_wrist = keypoints[KP_R_WRIST]
        l_sh = keypoints[KP_L_SHOULDER]
        r_sh = keypoints[KP_R_SHOULDER]
        if (
            l_wrist[2] < self.KP_CONF_THRESHOLD
            or r_wrist[2] < self.KP_CONF_THRESHOLD
            or l_sh[2] < self.KP_CONF_THRESHOLD
            or r_sh[2] < self.KP_CONF_THRESHOLD
        ):
            self._consecutive = 0
            return False
        # 어깨 기준 좌표 사용 (카메라 각도에 무관). 어깨너비로 정규화한 "위" 거리.
        shoulder_y = float((l_sh[1] + r_sh[1]) / 2)
        shoulder_width = float(abs(l_sh[0] - r_sh[0]))
        if shoulder_width < 0.02:
            self._consecutive = 0
            return False
        # 손목이 어깨보다 위 (y 작음) + 어깨너비의 30% 이상 떨어져 있어야 인정
        threshold = shoulder_y - shoulder_width * 0.3
        if l_wrist[1] < threshold and r_wrist[1] < threshold:
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
    """공통 — 코 좌표 한 축의 진동 감지. 끄덕임(y)/도리도리(x) 공용.

    HigherHRNet keypoint 노이즈가 의도적 머리 움직임 진폭과 비슷해 false
    positive가 자주 발생. 임계값 매우 보수적으로 — 명확한 의도적 동작만
    잡힘. 적당히 흔들면 안 잡힘 (의도적이고 큰 동작 필요).
    """

    KP_CONF_THRESHOLD = 0.20   # keypoint 신뢰도 더 엄격

    def __init__(
        self,
        axis: int,
        fps: float = 5.0,
        history_sec: float = 1.8,    # 더 긴 sustained 동작 필요
        cooldown_sec: float = 10.0,  # 한 번 감지 후 10초 차단
        min_amp: float = 0.06,       # 어깨너비 대비 6% 이상 진폭
        max_amp: float = 0.12,       # 12% 넘으면 일반 머리 회전
        min_zc: int = 5,             # 2.5 사이클 이상 (확실한 패턴)
        max_zc: int = 10,
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

    def diag(self) -> dict:
        return {
            "history": len(self.history),
            "need": self.history_max,
            "cooldown_remain": max(
                0.0, self.cooldown_sec - (time.time() - self._last_at),
            ),
        }

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
        # 어깨 자체가 움직이면 (사람이 위치 이동) 머리 동작이 아님 — track 따로
        # 시계열에 어깨 중심도 push해서 분리 확인
        self.history.append(float(nose[self.axis]))
        self._shoulder_widths.append(sw)
        # 어깨 중심 좌표도 같은 축으로 push
        if not hasattr(self, "_shoulder_axis_history"):
            from collections import deque as _dq
            self._shoulder_axis_history = _dq(maxlen=self.history_max)
        if self.axis == 0:
            sh_axis = float((l_sh[0] + r_sh[0]) / 2)
        else:
            sh_axis = float((l_sh[1] + r_sh[1]) / 2)
        self._shoulder_axis_history.append(sh_axis)
        if len(self.history) < self.history_max:
            return False
        np = _get_numpy()
        if np is None:
            return False
        arr = np.fromiter(self.history, dtype=np.float32)
        sh_arr = np.fromiter(self._shoulder_axis_history, dtype=np.float32)
        amp = float(arr.max() - arr.min())
        sh_amp = float(sh_arr.max() - sh_arr.min())
        sw_med = float(np.median(np.fromiter(
            self._shoulder_widths, dtype=np.float32,
        )))
        amp_ratio = amp / sw_med if sw_med > 0 else 0.0
        sh_amp_ratio = sh_amp / sw_med if sw_med > 0 else 0.0
        # 어깨가 머리와 비슷하게 움직였으면 머리 동작 아님 (몸 전체 흔들림)
        if sh_amp_ratio > amp_ratio * 0.5:
            return False
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
        self._shoulder_axis_history.clear()
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


# ─── 정면 응시 (사용자가 로봇을 쳐다봄) ───

class GazeAtMeDetector:
    """사용자가 다른 곳 보고 있다가 로봇 쪽으로 시선 돌린 순간 감지.

    원리:
      매 프레임 "정면 응시 여부" 측정. nose가 두 눈 중간 가까우면 정면
      (x축 정렬). 카메라가 사용자보다 낮게 책상에 있어 사용자를 올려다보는
      시점이라, 화면(위)을 보면 코가 눈 라인 위 또는 비슷한 y로 보이고
      반대로 로봇(아래)을 내려다보면 코가 눈보다 더 아래(y 큼)로 떨어짐.
      이 vertical 시그널을 함께 써서 "screen 보는 중" vs "robot 보는 중"
      구분.

      시간 윈도우 둘로 나눠:
        - before: hold_sec 이전 before_window_sec 동안 mostly NOT facing (<50%)
        - now: 최근 hold_sec 동안 mostly facing (>80%)
      이 두 조건 동시 만족 → "시선 전환" 감지 (의도적으로 쳐다봄).

    그냥 책상에 앉아 정면(화면) 향한 채 작업하는 건 — 코가 눈보다 위로
    떠 있어 facing 판정 X. 의도적으로 고개 숙여 로봇 쳐다봐야 잡힘.
    """

    KP_CONF_THRESHOLD = 0.20
    OFFSET_RATIO_MAX = 0.25
    # 낮은 카메라 시점에서 "로봇을 보는 중"의 nose-below-eyes 최소 비율.
    # nose_y - eye_y >= eye_dist * 이 값 이어야 facing 인정.
    # 0이면 vertical 무시 (구버전과 동일), 양수면 고개 살짝이라도 숙여야 함.
    PITCH_RATIO_MIN = 0.15

    def __init__(
        self,
        fps: float = 10.0,
        hold_sec: float = 1.2,            # 1.2초 안정적으로 정면
        before_window_sec: float = 2.5,   # 그 직전 2.5초 동안 다른 곳
        cooldown_sec: float = 60.0,
    ) -> None:
        self.hold_frames = max(3, int(fps * hold_sec))
        self.before_frames = max(5, int(fps * before_window_sec))
        total = self.hold_frames + self.before_frames
        self.recent: deque[bool] = deque(maxlen=total)
        self.cooldown_sec = cooldown_sec
        self._last_at = 0.0

    def reset(self) -> None:
        self.recent.clear()

    def diag(self) -> dict:
        seq = list(self.recent)
        return {
            "history": len(seq),
            "need": self.recent.maxlen,
            "facing_ratio": (sum(seq) / len(seq)) if seq else 0.0,
            "cooldown_remain": max(
                0.0, self.cooldown_sec - (time.time() - self._last_at),
            ),
        }

    def _is_facing(self, keypoints: Any) -> bool:
        nose = keypoints[KP_NOSE]
        l_eye = keypoints[KP_L_EYE]
        r_eye = keypoints[KP_R_EYE]
        if (
            nose[2] < self.KP_CONF_THRESHOLD
            or l_eye[2] < self.KP_CONF_THRESHOLD
            or r_eye[2] < self.KP_CONF_THRESHOLD
        ):
            return False
        eye_dist = float(abs(l_eye[0] - r_eye[0]))
        if eye_dist < 0.015:
            return False
        # x축 정렬 (양 눈 중간에 코가 있어야 정면)
        mid_eye_x = float((l_eye[0] + r_eye[0]) / 2)
        offset_ratio = abs(float(nose[0]) - mid_eye_x) / eye_dist
        if offset_ratio >= self.OFFSET_RATIO_MAX:
            return False
        # y축 pitch — 낮은 카메라이므로 로봇 보면 nose가 eye보다 아래(y 큼)
        mid_eye_y = float((l_eye[1] + r_eye[1]) / 2)
        pitch_ratio = (float(nose[1]) - mid_eye_y) / eye_dist
        return pitch_ratio >= self.PITCH_RATIO_MIN

    def process(self, keypoints: Any) -> bool:
        if keypoints is None:
            self.recent.clear()
            return False
        if time.time() - self._last_at < self.cooldown_sec:
            return False
        self.recent.append(self._is_facing(keypoints))
        if len(self.recent) < self.recent.maxlen:
            return False
        seq = list(self.recent)
        before = seq[:self.before_frames]
        now = seq[self.before_frames:]
        before_ratio = sum(before) / len(before)
        now_ratio = sum(now) / len(now)
        if now_ratio > 0.80 and before_ratio < 0.50:
            log.info(
                f"👀 시선 전환 감지! "
                f"(before={before_ratio:.2f}, now={now_ratio:.2f})"
            )
            self._last_at = time.time()
            self.recent.clear()
            return True
        return False
