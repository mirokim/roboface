"""입 그리기 — 단순한 선 기반 (모든 모양 일관된 두께)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pygame

from src.config import BEHAVIOR, COLOR_BG, COLOR_MOUTH, LINE_THICK
from src.face.expressions import MouthShape
from src.face.eyes import clean_arc


@dataclass
class MouthState:
    shape: MouthShape = MouthShape.NEUTRAL
    talk_amplitude: float = 0.0


def shape_for_amp(amp: float) -> MouthShape:
    """정규화된 입 모양 진폭(0~1) → MouthShape. mouth.py와 tts.py 공유 SSOT."""
    small, mid, large = BEHAVIOR.mouth_amp_thresholds
    if amp < small:
        return MouthShape.NEUTRAL
    if amp < mid:
        return MouthShape.OPEN_SMALL
    if amp < large:
        return MouthShape.OPEN_MID
    return MouthShape.OPEN_LARGE


def update_talking(state: MouthState, audio_rms: float, now: float) -> None:
    """raw RMS(0~0.3대) → 0~1 정규화 후 shape 결정."""
    state.talk_amplitude = min(1.0, audio_rms * BEHAVIOR.mouth_raw_rms_gain)
    state.shape = shape_for_amp(state.talk_amplitude)


def draw_mouth(
    surface: pygame.Surface,
    state: MouthState,
    center: tuple[int, int],
    width: int = 50,
) -> None:
    """입 그리기 — 단순한 호/선."""
    cx, cy = center
    s = state.shape

    if s in (MouthShape.NEUTRAL, MouthShape.SMILE):
        # 미소 호 — SMILE은 더 큰 호
        scale = 0.8 if s == MouthShape.SMILE else 0.55
        w = int(width * scale)
        h = int(width * scale * 0.7)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=False)
    elif s == MouthShape.GRIN:
        w = int(width * 0.95)
        h = int(width * 0.6)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=False)
    elif s == MouthShape.SAD:
        w = int(width * 0.6)
        h = int(width * 0.55)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=True)
    elif s == MouthShape.O:
        size = max(8, int(width * 0.3))
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        # 깔끔한 링: 외곽 채움 → 내부 펀칭
        pygame.draw.ellipse(surface, COLOR_MOUTH, rect)
        inner = rect.inflate(-2 * LINE_THICK, -2 * LINE_THICK)
        if inner.width > 0:
            pygame.draw.ellipse(surface, COLOR_BG, inner)
    elif s == MouthShape.BIG_O:
        size = max(12, int(width * 0.55))
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)
        pygame.draw.ellipse(surface, COLOR_MOUTH, rect)
        inner = rect.inflate(-2 * LINE_THICK, -2 * LINE_THICK)
        if inner.width > 0:
            pygame.draw.ellipse(surface, COLOR_BG, inner)
    elif s == MouthShape.WAVY:
        # 물결 — 4번 꺾이는 지그재그 라인
        w = int(width * 0.55)
        amp = max(2, width // 14)
        x0 = cx - w // 2
        seg = w // 4
        pts = [
            (x0,            cy),
            (x0 + seg,      cy - amp),
            (x0 + 2 * seg,  cy + amp),
            (x0 + 3 * seg,  cy - amp),
            (x0 + 4 * seg,  cy),
        ]
        pygame.draw.lines(surface, COLOR_MOUTH, False, pts, LINE_THICK)
    elif s == MouthShape.FLAT:
        pygame.draw.line(
            surface, COLOR_MOUTH,
            (cx - width // 5, cy), (cx + width // 5, cy),
            LINE_THICK,
        )
    elif s in (MouthShape.OPEN_SMALL, MouthShape.OPEN_MID, MouthShape.OPEN_LARGE):
        amp = max(0.15, state.talk_amplitude)
        h = int(width * 0.15 + width * 0.35 * amp)
        w = int(width * 0.4 + width * 0.2 * amp)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        pygame.draw.ellipse(surface, COLOR_MOUTH, rect)
