"""얼굴 부수 요소 — 땀방울, 떨림, 말풍선.

renderer.draw_face_to_surface 안에서 표정/입과 함께 그려짐.
"""

from __future__ import annotations

import math
from pathlib import Path

import pygame
import pygame.gfxdraw

from src.config import COLOR_BG, COLOR_EYE, DISPLAY_HEIGHT, DISPLAY_WIDTH, LINE_THICK
from src.utils.logger import get_logger

log = get_logger("face_extras")


# ─── 한국어 폰트 ───
# Pi: `sudo apt install -y fonts-noto-cjk` 권장
_FONT_PATH_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/Library/Fonts/AppleSDGothicNeo.ttc",
    "C:/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/NanumGothic.ttf",
]
_FONT_CACHE: dict[int, pygame.font.Font] = {}


def get_font(size: int = 14) -> pygame.font.Font:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    if not pygame.font.get_init():
        pygame.font.init()
    chosen: pygame.font.Font | None = None
    for p in _FONT_PATH_CANDIDATES:
        if Path(p).exists():
            try:
                chosen = pygame.font.Font(p, size)
                break
            except Exception:
                continue
    if chosen is None:
        chosen = pygame.font.Font(None, size)
        log.debug("한국어 폰트 못 찾음 — pygame 기본 폰트 사용")
    _FONT_CACHE[size] = chosen
    return chosen


# ─── 환경 오버레이 (우하단 작은 텍스트) ───

_ENV_FONT_SIZE = 12
_ENV_PADDING_PX = 4


def draw_env_overlay(
    canvas: pygame.Surface,
    temp_c: float | None,
    humidity_pct: float | None,
) -> None:
    """우하단에 "24.5° 52%" 형식의 작은 텍스트 표시.

    None인 값은 생략. 둘 다 None이면 아무것도 안 그림.
    """
    if temp_c is None and humidity_pct is None:
        return
    bits = []
    if temp_c is not None:
        bits.append(f"{temp_c:.1f}°")
    if humidity_pct is not None:
        bits.append(f"{int(round(humidity_pct))}%")
    text = " ".join(bits)
    font = get_font(_ENV_FONT_SIZE)
    # 흐릿한 회색 — 얼굴 방해 X
    surf = font.render(text, True, (140, 140, 150))
    w, h = surf.get_size()
    x = DISPLAY_WIDTH - w - _ENV_PADDING_PX
    y = DISPLAY_HEIGHT - h - _ENV_PADDING_PX
    canvas.blit(surf, (x, y))


# ─── 떨림 (shiver) ───
SHIVER_FREQ_HZ = 7.0
SHIVER_MAX_PX = 4


def shiver_offset(intensity: float, now: float) -> tuple[int, int]:
    """추울 때 얼굴 전체 미세 떨림. (dx, dy) px."""
    if intensity < 0.05:
        return 0, 0
    amp = intensity * SHIVER_MAX_PX
    phase = now * 2 * math.pi * SHIVER_FREQ_HZ
    dx = int(math.sin(phase) * amp)
    dy = int(math.cos(phase * 1.4) * amp * 0.4)
    return dx, dy


# ─── 땀방울 ───

def draw_sweat(
    canvas: pygame.Surface,
    intensity: float,
    right_eye_center: tuple[int, int],
    eye_size: int,
    now: float,
) -> None:
    """오른쪽 눈 옆에 땀방울. intensity 1.0이면 흐름까지 표현."""
    if intensity < 0.1:
        return
    # 위치 — 오른쪽 눈 바깥쪽 (얼굴 측면)
    cx = right_eye_center[0] + int(eye_size * 0.8)
    base_y = right_eye_center[1] - eye_size // 3
    # 흐르는 느낌 — 시간에 따라 y가 내려감
    drip = (now % 2.0) / 2.0   # 0~1 사이클
    cy = base_y + int(drip * eye_size * 0.7)

    # 물방울 모양 — 위쪽 뾰족, 아래쪽 둥근
    drop_w = max(6, int(8 * intensity))
    drop_h = max(10, int(14 * intensity))
    rect = pygame.Rect(cx - drop_w // 2, cy - drop_h // 2, drop_w, drop_h)
    # 채워진 타원 (윤곽선 흰색 톤)
    pygame.gfxdraw.filled_ellipse(
        canvas, rect.centerx, rect.centery,
        drop_w // 2, drop_h // 2, COLOR_EYE,
    )
    pygame.gfxdraw.aaellipse(
        canvas, rect.centerx, rect.centery,
        drop_w // 2, drop_h // 2, COLOR_EYE,
    )
    # 위쪽 꼬리 (작은 삼각형)
    tip_points = [
        (rect.centerx, rect.top - drop_h // 3),
        (rect.centerx - drop_w // 3, rect.top + 1),
        (rect.centerx + drop_w // 3, rect.top + 1),
    ]
    pygame.gfxdraw.filled_polygon(canvas, tip_points, COLOR_EYE)
    pygame.gfxdraw.aapolygon(canvas, tip_points, COLOR_EYE)


# ─── 말풍선 ───

_BUBBLE_MARGIN = 8         # 화면 좌우 여백
_BUBBLE_PADDING = 6        # 텍스트와 박스 내부 여백
_BUBBLE_MAX_LINES = 2
_BUBBLE_LINE_HEIGHT = 16   # 폰트 크기 + 여유
_BUBBLE_RADIUS = 8

# 말풍선은 같은 텍스트가 수십 프레임 반복 — 매 프레임 폰트 렌더링은 CPU 낭비.
# 마지막 그린 (text, font_size) → 완성된 Surface 캐시.
_BUBBLE_CACHE_KEY: tuple[str, int] | None = None
_BUBBLE_CACHE_SURFACE: pygame.Surface | None = None


def _wrap_text(text: str, font: pygame.font.Font, max_width: int) -> list[str]:
    """공백 단위로 줄바꿈 + 한 줄도 너무 길면 글자 단위로 잘라넣음."""
    lines: list[str] = []
    cur = ""

    def line_w(s: str) -> int:
        return font.size(s)[0]

    tokens = text.split(" ")
    for token in tokens:
        candidate = (cur + " " + token).strip() if cur else token
        if line_w(candidate) <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            # token이 단독으로도 너무 길면 글자 단위로 분할
            if line_w(token) > max_width:
                chunk = ""
                for ch in token:
                    if line_w(chunk + ch) > max_width:
                        if chunk:
                            lines.append(chunk)
                        chunk = ch
                    else:
                        chunk += ch
                cur = chunk
            else:
                cur = token
    if cur:
        lines.append(cur)
    return lines


def _build_bubble_surface(text: str, font_size: int) -> pygame.Surface:
    """말풍선 한 장(투명 배경) 만들기 — 폰트 렌더링은 여기서만."""
    font = get_font(font_size)
    max_text_w = DISPLAY_WIDTH - 2 * (_BUBBLE_MARGIN + _BUBBLE_PADDING)
    lines = _wrap_text(text, font, max_text_w)
    if len(lines) > _BUBBLE_MAX_LINES:
        lines = lines[:_BUBBLE_MAX_LINES]
        last = lines[-1]
        while font.size(last + "…")[0] > max_text_w and last:
            last = last[:-1]
        lines[-1] = last + "…"

    line_count = len(lines)
    line_h = font.get_linesize()
    bubble_h = line_count * line_h + 2 * _BUBBLE_PADDING
    bubble_w = DISPLAY_WIDTH - 2 * _BUBBLE_MARGIN
    tail_h = 6

    # 전체 합성용 surface (꼬리 포함 영역) — SRCALPHA로 투명
    surf = pygame.Surface(
        (DISPLAY_WIDTH, 4 + bubble_h + tail_h + 2),
        pygame.SRCALPHA,
    )
    bubble_rect = pygame.Rect(_BUBBLE_MARGIN, 4, bubble_w, bubble_h)
    pygame.draw.rect(surf, COLOR_BG, bubble_rect, border_radius=_BUBBLE_RADIUS)
    pygame.draw.rect(
        surf, COLOR_EYE, bubble_rect, width=2, border_radius=_BUBBLE_RADIUS,
    )
    for i, line in enumerate(lines):
        rendered = font.render(line, True, COLOR_EYE)
        tx = bubble_rect.centerx - rendered.get_width() // 2
        ty = bubble_rect.top + _BUBBLE_PADDING + i * line_h
        surf.blit(rendered, (tx, ty))

    # 꼬리
    tail_top_y = bubble_rect.bottom - 1
    tail_cx = DISPLAY_WIDTH // 2
    tail_w = 8
    pts = [
        (tail_cx - tail_w, tail_top_y),
        (tail_cx + tail_w, tail_top_y),
        (tail_cx, tail_top_y + tail_h),
    ]
    pygame.draw.polygon(surf, COLOR_BG, pts)
    pygame.draw.line(surf, COLOR_EYE, pts[0], pts[2], 2)
    pygame.draw.line(surf, COLOR_EYE, pts[1], pts[2], 2)
    return surf


def draw_speech_bubble(
    canvas: pygame.Surface,
    text: str,
    *,
    font_size: int = 16,
) -> None:
    """화면 상단에 말풍선 — 최대 2줄, 넘으면 …로 자름.

    같은 텍스트는 surface 캐싱해서 매 프레임 폰트 렌더 안 함 (CJK 폰트가 비쌈).
    """
    if not text:
        return
    global _BUBBLE_CACHE_KEY, _BUBBLE_CACHE_SURFACE
    key = (text, font_size)
    if _BUBBLE_CACHE_KEY != key or _BUBBLE_CACHE_SURFACE is None:
        _BUBBLE_CACHE_SURFACE = _build_bubble_surface(text, font_size)
        _BUBBLE_CACHE_KEY = key
    canvas.blit(_BUBBLE_CACHE_SURFACE, (0, 0))
