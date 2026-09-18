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
    """활짝 벌린 웃는 입 — 핑크 반타원 + 얇은 크림 외곽선 (레퍼런스 스타일)."""
    cx, cy = center
    rect = pygame.Rect(cx - w // 2, cy - h, w, 2 * h)
    _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, 2)
    pygame.draw.rect(surface, COLOR_BG,
                     pygame.Rect(rect.left - 4, rect.top - 4, rect.width + 8, h + 4))
    pygame.draw.line(surface, COLOR_MOUTH, (rect.left + 1, cy), (rect.right - 2, cy), 2)


def _pink_oval(surface, center, w: int, h: int) -> None:
    """작은 핑크 타원 입 (o / 말하기)."""
    cx, cy = center
    rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
    _filled_ellipse_outlined(surface, rect, COLOR_MOUTH_INNER, COLOR_MOUTH, 2)


def draw_mouth(
    surface: pygame.Surface,
    state: MouthState,
    center: tuple[int, int],
    width: int = 50,
) -> None:
    cx, cy = center
    s = state.shape
    thin = max(3, LINE_THICK - 1)

    if s == MouthShape.NEUTRAL:
        w = int(width * 0.34)
        h = int(width * 0.26)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=False, thickness=thin)
    elif s == MouthShape.SMILE:
        w = int(width * 0.55)
        h = int(width * 0.36)
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=False, thickness=thin)
    elif s == MouthShape.GRIN:
        _open_smile(surface, center, int(width * 0.62), int(width * 0.30))
    elif s == MouthShape.SAD:
        w = int(width * 0.36)
        h = int(width * 0.30)
        rect = pygame.Rect(cx - w // 2, cy - h // 2 + 4, w, h)
        clean_arc(surface, rect, COLOR_MOUTH, upper=True, thickness=thin)
    elif s == MouthShape.O:
        _pink_oval(surface, center, int(width * 0.20), int(width * 0.28))
    elif s == MouthShape.BIG_O:
        _pink_oval(surface, center, int(width * 0.34), int(width * 0.46))
    elif s == MouthShape.WAVY:
        w = int(width * 0.55)
        amp = max(3, width // 12)
        x0 = cx - w // 2
        seg = w // 4
        pts = [
            (x0,            cy),
            (x0 + seg,      cy - amp),
            (x0 + 2 * seg,  cy + amp),
            (x0 + 3 * seg,  cy - amp),
            (x0 + 4 * seg,  cy),
        ]
        pygame.draw.lines(surface, COLOR_MOUTH, False, pts, thin)
    elif s == MouthShape.FLAT:
        pygame.draw.line(
            surface, COLOR_MOUTH,
            (cx - width // 6, cy), (cx + width // 6, cy),
            thin,
        )
    elif s in (MouthShape.OPEN_SMALL, MouthShape.OPEN_MID, MouthShape.OPEN_LARGE):
        amp = max(0.15, state.talk_amplitude)
        h = int(width * 0.14 + width * 0.34 * amp)
        w = int(width * 0.22 + width * 0.30 * amp)
        _pink_oval(surface, center, w, h)
