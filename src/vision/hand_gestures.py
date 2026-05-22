"""MediaPipe Gesture Recognizer 기반 손 제스처 인식 — CPU 처리.

IMX500 NPU는 사람/포즈를 담당. 그와 병렬로 CPU에서 MediaPipe가 손을 처리:
- 21개 손 keypoint
- 7가지 내장 제스처: 👍 👎 ✌️ 🖐️ 👊 ☝️ 🤟
- palm center x로 wave 감지 (HigherHRNet 손목보다 정확)

Pi 5에서 ~14 FPS, 80% 정확도 (참고: github.com/mvipin/gesturebot).

모델 파일: gesture_recognizer.task. 첫 실행 시 자동 다운로드해서
DATA_DIR/models/에 캐시.
"""

from __future__ import annotations

import time
import urllib.request
from collections import deque
from typing import Any

from src.config import DATA_DIR
from src.utils.logger import get_logger

log = get_logger("hand_gestures")


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
MODEL_PATH = DATA_DIR / "models" / "gesture_recognizer.task"


# MediaPipe 내장 카테고리 → 우리 이벤트 이름 매핑
GESTURE_MAPPING = {
    "Thumb_Up":    "thumb_up",
    "Thumb_Down":  "thumb_down",
    "Victory":     "victory",      # ✌️
    "Open_Palm":   "open_palm",    # 🖐️
    "Closed_Fist": "fist",
    "Pointing_Up": "pointing_up",
    "ILoveYou":    "iloveyou",
    "None":        None,
}

# 손목/검지 등 주요 landmark 인덱스
LM_WRIST = 0
LM_INDEX_MCP = 5    # 검지 손가락 뿌리
LM_MIDDLE_MCP = 9   # 중지 손가락 뿌리 (손바닥 중심에 가까움)
LM_PINKY_MCP = 17   # 새끼 손가락 뿌리


def _ensure_model() -> bool:
    """모델 파일 없으면 다운로드. 성공 시 True."""
    if MODEL_PATH.exists():
        return True
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"MediaPipe 제스처 모델 다운로드: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info(f"  → 저장: {MODEL_PATH}")
        return True
    except Exception as e:
        log.warning(f"모델 다운로드 실패: {e}")
        return False


class HandGestureDetector:
    """MediaPipe Gesture Recognizer wrapper.

    process(frame_rgb, ts_ms) → 매 프레임 호출.
    잡힌 손이 있고 신뢰도 충분하면 (gesture_name, gesture_score) 또는 None.
    별도로 wave 감지 — 손바닥 중심 x 시계열의 oscillation.
    """

    def __init__(
        self,
        min_confidence: float = 0.5,
        fps: float = 10.0,
        wave_history_sec: float = 1.5,
        wave_min_amp_ratio: float = 0.5,   # 손폭의 50% 이상 움직임
        wave_cooldown_sec: float = 5.0,
        gesture_cooldown_sec: float = 4.0,
    ) -> None:
        self.min_confidence = min_confidence
        self._recognizer = None
        self._init_ok = self._init_mediapipe()
        # wave 트래킹
        self._palm_history: deque[float] = deque(
            maxlen=max(6, int(fps * wave_history_sec)),
        )
        self._hand_width_history: deque[float] = deque(maxlen=self._palm_history.maxlen)
        self.wave_min_amp_ratio = wave_min_amp_ratio
        self.wave_cooldown_sec = wave_cooldown_sec
        self._last_wave_at = 0.0
        # 카테고리 제스처 cooldown (kind별)
        self.gesture_cooldown_sec = gesture_cooldown_sec
        self._last_gesture_at: dict[str, float] = {}
        # 정적 제스처는 sustained 필요 — 같은 제스처 N프레임 연속이어야 emit
        self._sustained_target = 3
        self._last_seen: tuple[str, int] = ("", 0)

    def _init_mediapipe(self) -> bool:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            log.warning(f"mediapipe 미설치 — 손 제스처 비활성: {e}")
            return False
        if not _ensure_model():
            return False
        try:
            options = mp_vision.GestureRecognizerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=self.min_confidence,
                min_hand_presence_confidence=self.min_confidence,
                min_tracking_confidence=self.min_confidence,
            )
            self._recognizer = mp_vision.GestureRecognizer.create_from_options(options)
            self._mp = mp
            log.info("MediaPipe 제스처 인식기 초기화 완료")
            return True
        except Exception as e:
            log.warning(f"MediaPipe 초기화 실패: {e}")
            return False

    def reset(self) -> None:
        self._palm_history.clear()
        self._hand_width_history.clear()
        self._last_seen = ("", 0)

    def process(self, frame_rgb: Any, ts_ms: int) -> tuple[str | None, bool]:
        """frame_rgb: HxWx3 uint8 RGB numpy. ts_ms: 단조증가 ms.

        반환: (gesture_name_or_None, wave_detected).
        gesture_name은 cooldown/sustained 통과한 것만.
        """
        if not self._init_ok or self._recognizer is None:
            return None, False
        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=frame_rgb,
            )
            result = self._recognizer.recognize_for_video(mp_image, ts_ms)
        except Exception as e:
            log.debug(f"hand recognize 에러: {e}")
            return None, False

        if not result.gestures or not result.hand_landmarks:
            self.reset()
            return None, False

        # 첫 손만 (num_hands=1)
        gesture_categories = result.gestures[0]
        landmarks = result.hand_landmarks[0]

        # 1) 손바닥 중심 x 시계열 push — wave 감지용
        try:
            mid = landmarks[LM_MIDDLE_MCP]
            wrist = landmarks[LM_WRIST]
            pinky = landmarks[LM_PINKY_MCP]
            idx = landmarks[LM_INDEX_MCP]
            palm_x = float(mid.x)
            hand_width = abs(float(idx.x) - float(pinky.x))
            if hand_width > 0.01:
                self._palm_history.append(palm_x)
                self._hand_width_history.append(hand_width)
        except Exception:
            pass

        wave_detected = self._check_wave()

        # 2) 카테고리 제스처
        gesture_name: str | None = None
        if gesture_categories:
            top = gesture_categories[0]
            cat = top.category_name
            score = float(top.score)
            mapped = GESTURE_MAPPING.get(cat)
            if mapped and score >= self.min_confidence:
                # sustained — 같은 카테고리 N프레임 연속
                if self._last_seen[0] == mapped:
                    self._last_seen = (mapped, self._last_seen[1] + 1)
                else:
                    self._last_seen = (mapped, 1)
                if self._last_seen[1] >= self._sustained_target:
                    # cooldown
                    now = time.time()
                    last = self._last_gesture_at.get(mapped, 0.0)
                    if now - last >= self.gesture_cooldown_sec:
                        self._last_gesture_at[mapped] = now
                        gesture_name = mapped
                        log.info(
                            f"✋ 손 제스처: {mapped} (score={score:.2f})"
                        )
                        self._last_seen = (mapped, 0)  # 다시 sustained 필요

        return gesture_name, wave_detected

    def _check_wave(self) -> bool:
        """손바닥 중심 x oscillation으로 wave 감지."""
        if time.time() - self._last_wave_at < self.wave_cooldown_sec:
            return False
        if len(self._palm_history) < self._palm_history.maxlen:
            return False
        try:
            import numpy as np
        except ImportError:
            return False
        arr = np.fromiter(self._palm_history, dtype=np.float32)
        widths = np.fromiter(self._hand_width_history, dtype=np.float32)
        median_width = float(np.median(widths))
        if median_width < 0.01:
            return False
        amp = float(arr.max() - arr.min())
        amp_ratio = amp / median_width

        median_val = float(np.median(arr))
        signs = np.sign(arr - median_val)
        zc = int(np.sum(np.abs(np.diff(signs)) > 0))

        if amp_ratio >= self.wave_min_amp_ratio and 2 <= zc <= 12:
            log.info(
                f"👋 hand wave 감지! amp_ratio={amp_ratio:.2f} zc={zc}"
            )
            self._last_wave_at = time.time()
            self._palm_history.clear()
            self._hand_width_history.clear()
            return True
        return False

    def close(self) -> None:
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            except Exception:
                pass
            self._recognizer = None
