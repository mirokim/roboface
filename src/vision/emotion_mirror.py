"""사용자 표정 인식 — 사용자가 웃으면 같이 웃고, 놀라면 같이 놀람.

가벼운 첫 버전: OpenCV haar cascade.
- haarcascade_frontalface_default.xml — 얼굴 위치
- haarcascade_smile.xml — 웃음 (정확도 60~70%지만 가벼움)
- 입 크기 (face 높이 대비) → 큰 입 = SURPRISED

frame과 person bbox만 있으면 동작. Pi 5 CPU에서 5fps 충분.

False positive 방지:
- 3 프레임 연속 같은 감지 → confirmed
- 5초 cooldown (같은 표정 반복 트리거 X)
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.utils.logger import get_logger

log = get_logger("emotion_mirror")


# 출력 emotion enum (mirror가 emit하는 단순 상태)
EMOTION_NEUTRAL = "neutral"
EMOTION_SMILE = "smile"
EMOTION_SURPRISED = "surprised"


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
        self._classifiers: Any = None

    def _load_classifiers(self):
        if self._classifiers is not None:
            return self._classifiers
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            log.warning("opencv-python 미설치 — emotion_mirror 비활성")
            self._classifiers = (None, None, None)
            return self._classifiers
        haar = cv2.data.haarcascades
        face_clf = cv2.CascadeClassifier(haar + "haarcascade_frontalface_default.xml")
        smile_clf = cv2.CascadeClassifier(haar + "haarcascade_smile.xml")
        if face_clf.empty() or smile_clf.empty():
            log.warning("haar cascade XML 로드 실패")
            self._classifiers = (None, None, None)
            return self._classifiers
        self._classifiers = (cv2, face_clf, smile_clf)
        log.info("emotion_mirror: haar cascade 로드 완료")
        return self._classifiers

    def process(
        self,
        frame: Any,
        person_bbox: tuple[float, float, float, float] | None,
    ) -> str | None:
        """frame (numpy HxWx3 RGB), bbox 정규화. 새로 confirmed 된 emotion 반환 or None."""
        if frame is None or person_bbox is None:
            self._recent.clear()
            return None

        cv2, face_clf, smile_clf = self._load_classifiers()
        if face_clf is None:
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
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        # 너무 크면 다운샘플 (속도)
        if gray.shape[0] > 240:
            scale = 240 / gray.shape[0]
            new_w = int(gray.shape[1] * scale)
            gray = cv2.resize(gray, (new_w, 240))

        faces = face_clf.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                          minSize=(30, 30))
        if len(faces) == 0:
            self._recent.append(EMOTION_NEUTRAL)
            return None

        # 가장 큰 face 선택
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[fy:fy + fh, fx:fx + fw]

        # smile detector — 얼굴 하단 절반에서만
        lower = face_gray[fh // 2:, :]
        smiles = smile_clf.detectMultiScale(
            lower, scaleFactor=1.7, minNeighbors=20,
            minSize=(int(fw * 0.3), int(fh * 0.15)),
        )

        if len(smiles) > 0:
            label = EMOTION_SMILE
        else:
            label = EMOTION_NEUTRAL

        self._recent.append(label)
        return self._maybe_emit(label)

    def _maybe_emit(self, label: str) -> str | None:
        # confirm_frames 연속 같은 라벨이어야 emit
        if len(self._recent) < self.confirm_frames:
            return None
        if any(r != label for r in self._recent):
            return None
        if label == EMOTION_NEUTRAL:
            return None
        # cooldown
        now = time.monotonic()
        last = self._last_emit.get(label, 0.0)
        if now - last < self.cooldown_sec:
            return None
        self._last_emit[label] = now
        log.info(f"😊 emotion: {label}")
        return label
