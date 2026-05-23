"""사용자 표정 인식 — mediapipe FaceMesh 우선, haar cascade fallback.

지원 표정:
- smile (입꼬리 위로)
- surprised (입 크게 벌어짐)
- sad (입꼬리 아래로)
- neutral (그 외)

Backend 선택:
1. mediapipe FaceMesh — 468 landmarks 기반 정밀 분류. 가벼움(CPU only).
   Python 3.12 이하에서만 wheel 제공. Pi 5 Bookworm은 Python 3.11이라 OK.
2. (fallback) OpenCV haar cascade — smile만 잡음. mediapipe 설치 안 됐을 때.

False positive 방지:
- 3 프레임 연속 같은 라벨 → confirmed
- 4초 cooldown per label
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger

log = get_logger("emotion_mirror")


EMOTION_NEUTRAL = "neutral"
EMOTION_SMILE = "smile"
EMOTION_SURPRISED = "surprised"
EMOTION_SAD = "sad"


# === mediapipe FaceMesh landmark 인덱스 ===
# https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
_LM_MOUTH_LEFT_CORNER = 61
_LM_MOUTH_RIGHT_CORNER = 291
_LM_UPPER_LIP_TOP = 13      # 윗입술 안쪽 위 가장자리
_LM_LOWER_LIP_BOTTOM = 14   # 아랫입술 안쪽 아래 가장자리


class EmotionMirror:
    def __init__(
        self,
        confirm_frames: int = 3,
        cooldown_sec: float = 4.0,
    ) -> None:
        self.confirm_frames = confirm_frames
        self.cooldown_sec = cooldown_sec
        self._recent: deque[str] = deque(maxlen=confirm_frames)
        self._last_emit: dict[str, float] = {}
        # backend: "mediapipe" / "haar" / "none"
        self._backend: str | None = None
        self._mp_face_mesh: Any = None
        self._haar: Any = None  # (cv2, face_clf, smile_clf) tuple

    # ── Backend 로딩 ──

    def _ensure_backend(self) -> str:
        """첫 호출 시 backend 결정. 한 번만 시도."""
        if self._backend is not None:
            return self._backend
        # 1) mediapipe 시도
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._backend = "mediapipe"
            log.info("emotion_mirror backend: mediapipe FaceMesh")
            return self._backend
        except Exception as e:
            log.debug(f"mediapipe 사용 불가 — haar로 fallback: {e}")
        # 2) haar 시도
        try:
            import cv2  # type: ignore[import-not-found]
            haar = cv2.data.haarcascades
            face_clf = cv2.CascadeClassifier(haar + "haarcascade_frontalface_default.xml")
            smile_clf = cv2.CascadeClassifier(haar + "haarcascade_smile.xml")
            if face_clf.empty() or smile_clf.empty():
                raise RuntimeError("haar cascade XML 로드 실패")
            self._haar = (cv2, face_clf, smile_clf)
            self._backend = "haar"
            log.info("emotion_mirror backend: haar cascade (mediapipe 없음)")
            return self._backend
        except Exception as e:
            log.warning(f"emotion_mirror 비활성 — backend 없음: {e}")
            self._backend = "none"
            return self._backend

    # ── 공개 API ──

    def process(
        self,
        frame: Any,
        person_bbox: tuple[float, float, float, float] | None,
    ) -> str | None:
        """frame (HxWx3 RGB), bbox 정규화. 새로 confirmed 된 emotion 반환 or None."""
        if frame is None or person_bbox is None:
            self._recent.clear()
            return None

        backend = self._ensure_backend()
        if backend == "none":
            return None

        # bbox 위쪽 절반 crop (얼굴이 보통 위쪽)
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = person_bbox
        upper_y1 = y0 + (y1 - y0) * 0.5
        px0, py0 = int(max(0.0, x0) * w), int(max(0.0, y0) * h)
        px1 = int(min(1.0, x1) * w)
        py1 = int(min(1.0, upper_y1) * h)
        if px1 - px0 < 40 or py1 - py0 < 40:
            return None

        roi = frame[py0:py1, px0:px1]
        if backend == "mediapipe":
            label = self._classify_mediapipe(roi)
        else:
            label = self._classify_haar(roi)

        if label is None:
            # 얼굴 못 잡음 — 시계열 초기화
            return None

        self._recent.append(label)
        return self._maybe_emit(label)

    # ── mediapipe 분류 ──

    def _classify_mediapipe(self, roi_rgb: Any) -> str | None:
        # mediapipe는 RGB 기대. roi는 이미 RGB 가정 (vision_task에서 RGB 전달).
        try:
            result = self._mp_face_mesh.process(roi_rgb)
        except Exception as e:
            log.debug(f"mediapipe process 실패: {e}")
            return None
        if not result.multi_face_landmarks:
            return None
        landmarks = result.multi_face_landmarks[0].landmark

        # 좌표는 ROI 기준 0~1 정규화.
        l_corner = landmarks[_LM_MOUTH_LEFT_CORNER]
        r_corner = landmarks[_LM_MOUTH_RIGHT_CORNER]
        upper = landmarks[_LM_UPPER_LIP_TOP]
        lower = landmarks[_LM_LOWER_LIP_BOTTOM]

        mouth_width = abs(l_corner.x - r_corner.x)
        if mouth_width < 0.02:
            return EMOTION_NEUTRAL
        mouth_height = abs(upper.y - lower.y)
        aspect = mouth_height / mouth_width   # 입 종횡비

        # 입꼬리 평균 y vs 입 중앙 y.
        # y가 클수록 화면 아래 — 입꼬리 y가 작으면 위로 올라감 (smile).
        center_y = (upper.y + lower.y) / 2
        corner_avg_y = (l_corner.y + r_corner.y) / 2
        # mouth_width로 정규화 — 얼굴 크기 무관한 비율.
        corner_offset = (center_y - corner_avg_y) / mouth_width

        # 분류 — 임계값은 경험적. 튜닝 가능.
        if aspect > 0.55:
            # 입이 동그랗게 크게 벌어짐 — 놀람.
            return EMOTION_SURPRISED
        if corner_offset > 0.06:
            return EMOTION_SMILE
        if corner_offset < -0.06:
            return EMOTION_SAD
        return EMOTION_NEUTRAL

    # ── haar fallback ──

    def _classify_haar(self, roi_rgb: Any) -> str | None:
        cv2, face_clf, smile_clf = self._haar
        if roi_rgb.ndim == 3:
            gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi_rgb
        if gray.shape[0] > 240:
            scale = 240 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 240))
        faces = face_clf.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30),
        )
        if len(faces) == 0:
            return None
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[fy:fy + fh, fx:fx + fw]
        lower = face_gray[fh // 2:, :]
        smiles = smile_clf.detectMultiScale(
            lower, scaleFactor=1.7, minNeighbors=20,
            minSize=(int(fw * 0.3), int(fh * 0.15)),
        )
        return EMOTION_SMILE if len(smiles) > 0 else EMOTION_NEUTRAL

    # ── emit 게이팅 ──

    def _maybe_emit(self, label: str) -> str | None:
        if len(self._recent) < self.confirm_frames:
            return None
        if any(r != label for r in self._recent):
            return None
        if label == EMOTION_NEUTRAL:
            return None
        now = time.monotonic()
        last = self._last_emit.get(label, 0.0)
        if now - last < self.cooldown_sec:
            return None
        self._last_emit[label] = now
        log.info(f"😊 emotion: {label}")
        return label
