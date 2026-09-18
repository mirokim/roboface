"""입 그리기 — 치비 스타일. 선 입 + 속 채운 벌린 입."""

from __future__ import annotations

from dataclasses import dataclass

import pygame
import pygame.gfxdraw

from src.config import BEHAVIOR, COLOR_BG, COLOR_MOUTH, COLOR_MOUTH_INNER, LINE_THICK
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


def _filled_ellipse_outlined(surface, rect: pygame.Rect, inner, outline, thick: int) -> None:
    """속 색 + 외곽선 타원."""
    pygame.draw.ellipse(surface, outline, rect)
    in_rect = rect.inflate(-2 * thick, -2 * thick)
    if in_rect.width > 0 and in_rect.height > 0:
        pygame.draw.ellipse(surface, inner, in_rect)


def _open_smile(surface, center, w: int, h: int) -> None:
    """활짝 벌린 웃는 입 — 아래쪽 반타원 속 채움 + 흰 외곽선 + 윗선."""
    cx, cy = center
    rect = pygame.Rect(cx - w // 2, cy - h, w, 2 * h)
    _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, LINE_THICK)
    # 윗반쪽 마스킹 → 반타원
    pygame.draw.rect(surface, COLOR_BG, pygame.Rect(rect.left - 4, rect.top - 4, rect.width + 8, h + 4))
    pygame.draw.line(surface, COLOR_MOUTH, (rect.left + 1, cy), (rect.right - 2, cy), LINE_THICK)


def draw_mouth(
    surface: pygame.Surface,
    state: MouthState,
    center: tuple[int, int],
    width: int = 50,
) -> None:
    cx, cy = center
    s = state.shape

    if s in (MouthShape.NEUTRAL, MouthShape.SMILE):
        scale = 0.85 if s == MouthShape.SMILE else 0.55
        w = int(width * scale)
        h = int(width * scale * 0.7)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=False)
    elif s == MouthShape.GRIN:
        _open_smile(surface, center, int(width * 1.0), int(width * 0.45))
    elif s == MouthShape.SAD:
        w = int(width * 0.55)
        h = int(width * 0.5)
        rect = pygame.Rect(cx - w // 2, cy - h // 2 + 4, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=True)
    elif s == MouthShape.O:
        size = max(14, int(width * 0.38))
        rect = pygame.Rect(cx - size // 2, cy - size // 2, size, int(size * 1.15))
        _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, LINE_THICK)
    elif s == MouthShape.BIG_O:
        w = max(14, int(width * 0.55))
        h = int(w * 1.25)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, LINE_THICK)
    elif s == MouthShape.WAVY:
        w = int(width * 0.7)
        amp = max(3, width // 10)
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
            (cx - width // 4, cy), (cx + width // 4, cy),
            LINE_THICK,
        )
    elif s in (MouthShape.OPEN_SMALL, MouthShape.OPEN_MID, MouthShape.OPEN_LARGE):
        amp = max(0.15, state.talk_amplitude)
        h = int(width * 0.12 + width * 0.4 * amp)
        w = int(width * 0.45 + width * 0.3 * amp)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, LINE_THICK)
