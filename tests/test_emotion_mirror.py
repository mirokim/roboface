"""EmotionMirror 단위 테스트 — 합성 frame으로 가벼운 동작 검증.

진짜 표정 인식 정확도는 실제 사용자/카메라 환경에서만 검증 가능.
여기서는 (1) 모듈 동작, (2) bbox 없을 때 reset, (3) confirm/cooldown 로직만 확인.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.vision.emotion_mirror import (
    EMOTION_NEUTRAL, EMOTION_SAD, EMOTION_SMILE, EMOTION_SURPRISED, EmotionMirror,
)


def test_reset_on_no_bbox():
    em = EmotionMirror()
    em._recent.extend(["smile", "smile"])
    result = em.process(np.zeros((100, 100, 3), dtype=np.uint8), None)
    assert result is None
    assert len(em._recent) == 0


def test_neutral_label_does_not_emit():
    em = EmotionMirror(confirm_frames=2)
    em._recent.extend([EMOTION_NEUTRAL, EMOTION_NEUTRAL])
    # _maybe_emit이 neutral 절대 emit 안 함
    assert em._maybe_emit(EMOTION_NEUTRAL) is None


def test_smile_requires_consecutive_confirms():
    em = EmotionMirror(confirm_frames=3, cooldown_sec=0.1)
    em._recent.append(EMOTION_SMILE)
    em._recent.append(EMOTION_NEUTRAL)
    em._recent.append(EMOTION_SMILE)
    # 연속 아니라서 emit 안 됨
    assert em._maybe_emit(EMOTION_SMILE) is None

    em._recent.clear()
    em._recent.extend([EMOTION_SMILE, EMOTION_SMILE, EMOTION_SMILE])
    assert em._maybe_emit(EMOTION_SMILE) == EMOTION_SMILE


def test_smile_cooldown_blocks_immediate_repeat():
    em = EmotionMirror(confirm_frames=2, cooldown_sec=2.0)
    em._recent.extend([EMOTION_SMILE, EMOTION_SMILE])
    assert em._maybe_emit(EMOTION_SMILE) == EMOTION_SMILE
    # 즉시 다시
    assert em._maybe_emit(EMOTION_SMILE) is None


def test_too_small_bbox_returns_none():
    em = EmotionMirror()
    tiny_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox가 너무 작아서 ROI < 40px
    assert em.process(tiny_frame, (0.45, 0.45, 0.5, 0.5)) is None


def test_processes_arbitrary_frame_without_crashing():
    """무작위 잡음 이미지 — 표정은 안 잡히겠지만 크래시 X."""
    em = EmotionMirror()
    rng = np.random.default_rng(seed=42)
    noise = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
    result = em.process(noise, (0.1, 0.05, 0.9, 0.95))
    # 결과는 None 또는 valid emotion
    assert result in (None, EMOTION_SMILE, EMOTION_SURPRISED, EMOTION_SAD)
