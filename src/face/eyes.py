"""눈 그리기 — 깔끔한 Stack-chan 정통 스타일.

원칙:
- 모든 "뜬 눈"은 같은 채워진 타원, 모양은 비율/크기로만 차이
- 추가 요소(눈썹/하이라이트/동공) 없음
- 모든 선 두께는 LINE_THICK 일관
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from src.config import BEHAVIOR, COLOR_BG, COLOR_EYE, LINE_THICK
from src.face.expressions import EyeShape


@dataclass
class EyeState:
    """현재 눈 상태."""

    shape: EyeShape = EyeShape.NORMAL
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    blink: float = 0.0          # 0=뜸, 1=완전히 감김
    _next_blink_at: float = 0.0
    _blinking_since: float | None = None


def eye_extent_below(shape: EyeShape, size: int) -> int:
    """현재 눈 모양이 중심에서 얼마나 아래로 뻗는지 (px).

    렌더러가 입 위치 결정할 때 사용 — 표정마다 눈 실제 크기가 달라서.
    """
    if shape in (EyeShape.SQUINT, EyeShape.ANGRY):
        return size // 4               # 꺾인 선, size/4
    if shape == EyeShape.HAPPY:
        return int(size * 0.45)        # 호의 아래 끝
    if shape == EyeShape.SLEEPY:
        return int(size * 0.18)
    if shape == EyeShape.WORRIED:
        return int(4 + size * 0.75 / 2)
    if shape == EyeShape.SURPRISED:
        return int(size * 1.15 / 2)
    if shape == EyeShape.LOVE:
        return size // 2 + size // 6   # 하트 하단 삼각형
    if shape in (EyeShape.STAR, EyeShape.DIZZY):
        return size // 2
    if shape == EyeShape.CLOSED:
        return 0
    if shape in (EyeShape.WINK_LEFT, EyeShape.WINK_RIGHT):
        return size // 2               # 한 쪽이 떴으니 보통 크기 유지
    return size // 2                   # NORMAL 등 기본 타원


def schedule_next_blink(state: EyeState, now: float) -> None:
    interval = random.uniform(
        BEHAVIOR.blink_min_interval_sec,
        BEHAVIOR.blink_max_interval_sec,
    )
    state._next_blink_at = now + interval


def update_blink(state: EyeState, now: float) -> None:
    if state._blinking_since is not None:
        elapsed_ms = (now - state._blinking_since) * 1000
        duration = BEHAVIOR.blink_duration_ms
        if elapsed_ms >= duration:
            state.blink = 0.0
            state._blinking_since = None
            schedule_next_blink(state, now)
        else:
            t = elapsed_ms / duration
            state.blink = math.sin(t * math.pi)
        return
    if state._next_blink_at == 0.0:
        schedule_next_blink(state, now)
        return
    if now >= state._next_blink_at:
        state._blinking_since = now


def trigger_blink(state: EyeState, now: float) -> None:
    state._blinking_since = now


# ─── 깔끔한 두꺼운 호 그리기 ───
# pygame.draw.arc()는 두께 4+에서 픽셀 거칠어짐.
# 대신 외곽 채움 → 내부 펀칭 → 반쪽 마스킹 기법 사용.

def clean_arc(
    surface: pygame.Surface,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    *,
    upper: bool = False,        # True=상단 호(찡그림), False=하단 호(미소)
    thickness: int = LINE_THICK,
) -> None:
    """깔끔한 두꺼운 호. pygame.draw.arc 보다 부드럽고 일정한 두께."""
    if rect.width < 4 or rect.height < 4:
        return
    pygame.draw.ellipse(surface, color, rect)
    inner = rect.inflate(-2 * thickness, -2 * thickness)
    if inner.width > 0 and inner.height > 0:
        pygame.draw.ellipse(surface, COLOR_BG, inner)
    # 반쪽 마스킹
    cy = rect.centery
    if upper:
        mask = pygame.Rect(
            rect.left - 4, cy, rect.width + 8, rect.bottom - cy + 4,
        )
    else:
        mask = pygame.Rect(
            rect.left - 4, rect.top - 4, rect.width + 8, cy - rect.top + 4,
        )
    pygame.draw.rect(surface, COLOR_BG, mask)


# ─── 그리기 ───

def draw_eyes(
    surface: pygame.Surface,
    state: EyeState,
    left_center: tuple[int, int],
    right_center: tuple[int, int],
    size: int = 50,
) -> None:
    """양쪽 눈 그리기."""
    open_ratio = 1.0 - state.blink

    if state.shape == EyeShape.CLOSED or open_ratio < 0.05:
        _line(surface, left_center, size)
        _line(surface, right_center, size)
        return

    # 윙크: 한쪽만 감음
    if state.shape == EyeShape.WINK_LEFT:
        _line(surface, left_center, size)
        _ellipse(surface, right_center, size, open_ratio, scale=1.0)
        return
    if state.shape == EyeShape.WINK_RIGHT:
        _ellipse(surface, left_center, size, open_ratio, scale=1.0)
        _line(surface, right_center, size)
        return

    # 하트
    if state.shape == EyeShape.LOVE:
        _heart(surface, left_center, size)
        _heart(surface, right_center, size)
        return

    # 별
    if state.shape == EyeShape.STAR:
        _star(surface, left_center, size)
        _star(surface, right_center, size)
        return

    # 어지러움 (소용돌이)
    if state.shape == EyeShape.DIZZY:
        _dizzy(surface, left_center, size)
        _dizzy(surface, right_center, size)
        return

    # 라인 표정 (^^, >_<, ><)
    if state.shape == EyeShape.HAPPY:
        _arc_up(surface, left_center, size)
        _arc_up(surface, right_center, size)
        return
    if state.shape == EyeShape.SQUINT:
        _angle(surface, left_center, size, point_right=True)
        _angle(surface, right_center, size, point_right=False)
        return
    if state.shape == EyeShape.ANGRY:
        _angle(surface, left_center, size, point_right=False)  # 바깥쪽으로 (찡그림)
        _angle(surface, right_center, size, point_right=True)
        return

    # 채워진 타원 — 크기/비율만 다름. 시선 오프셋 전달.
    gx, gy = int(state.gaze_x * 6), int(state.gaze_y * 4)
    if state.shape == EyeShape.SURPRISED:
        _ellipse(surface, left_center, size, open_ratio, scale=1.15, offset_x=gx, offset_y=gy)
        _ellipse(surface, right_center, size, open_ratio, scale=1.15, offset_x=gx, offset_y=gy)
    elif state.shape == EyeShape.SLEEPY:
        _ellipse(surface, left_center, size, open_ratio * 0.35, scale=1.0, offset_x=gx)
        _ellipse(surface, right_center, size, open_ratio * 0.35, scale=1.0, offset_x=gx)
    elif state.shape == EyeShape.WORRIED:
        _ellipse(surface, left_center, size, open_ratio, scale=0.75, offset_x=gx, offset_y=4 + gy)
        _ellipse(surface, right_center, size, open_ratio, scale=0.75, offset_x=gx, offset_y=4 + gy)
    else:  # NORMAL 등
        _ellipse(surface, left_center, size, open_ratio, scale=1.0, offset_x=gx, offset_y=gy)
        _ellipse(surface, right_center, size, open_ratio, scale=1.0, offset_x=gx, offset_y=gy)


# ─── 헬퍼 (각 모양마다 함수 하나, 단순하게) ───

def _ellipse(
    surface: pygame.Surface,
    center: tuple[int, int],
    base_size: int,
    open_ratio: float,
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
) -> None:
    """채워진 타원. 깜빡임은 height 압축으로. gaze는 offset으로."""
    w = int(base_size * scale)
    h = max(2, int(base_size * scale * open_ratio))
    rect = pygame.Rect(0, 0, w, h).move(
        center[0] - w // 2 + offset_x, center[1] - h // 2 + offset_y,
    )
    pygame.draw.ellipse(surface, COLOR_EYE, rect)


def _line(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """감긴 눈 — 가로 선."""
    cx, cy = center
    half = size // 2
    pygame.draw.line(
        surface, COLOR_EYE,
        (cx - half + 4, cy), (cx + half - 4, cy),
        LINE_THICK,
    )


def _arc_up(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """위로 휘어진 호 — ⌒ (행복)."""
    cx, cy = center
    w = int(size * 0.85)
    h = int(size * 0.6)
    rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
    clean_arc(surface, rect, COLOR_EYE, upper=False)


def _angle(
    surface: pygame.Surface, center: tuple[int, int], size: int, point_right: bool,
) -> None:
    """꺾인 선 — `>` 또는 `<`. point_right=True면 `>`, False면 `<`.

    선 두께가 채워진 도형보다 시각적으로 약해 보여서, 일반 LINE_THICK의 약 2배.
    """
    cx, cy = center
    half = size // 2
    thick = LINE_THICK * 2
    if point_right:
        pts = [(cx - half + 4, cy - half // 2),
               (cx + half - 4, cy),
               (cx - half + 4, cy + half // 2)]
    else:
        pts = [(cx + half - 4, cy - half // 2),
               (cx - half + 4, cy),
               (cx + half - 4, cy + half // 2)]
    pygame.draw.lines(surface, COLOR_EYE, False, pts, thick)


def _heart(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """단순 채워진 하트."""
    s = size // 2
    cx, cy = center
    pygame.draw.circle(surface, COLOR_EYE, (cx - s // 2, cy - s // 4), s // 2)
    pygame.draw.circle(surface, COLOR_EYE, (cx + s // 2, cy - s // 4), s // 2)
    pygame.draw.polygon(surface, COLOR_EYE, [
        (cx - s, cy - s // 8),
        (cx + s, cy - s // 8),
        (cx, cy + s),
    ])


def _star(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """5각 별."""
    cx, cy = center
    outer_r = size // 2
    inner_r = outer_r * 0.4
    points: list[tuple[float, float]] = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = outer_r if i % 2 == 0 else inner_r
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    pygame.draw.polygon(surface, COLOR_EYE, points)


def _dizzy(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """소용돌이 — 동심원 두 개 + X 표시. 깔끔한 링 기법."""
    cx, cy = center
    r_outer = size // 2
    # 외곽 채움 → 내부 펀칭으로 깔끔한 링
    pygame.draw.circle(surface, COLOR_EYE, (cx, cy), r_outer)
    pygame.draw.circle(surface, COLOR_BG, (cx, cy), r_outer - LINE_THICK)
    # X 표시
    half = r_outer // 2
    pygame.draw.line(
        surface, COLOR_EYE,
        (cx - half, cy - half), (cx + half, cy + half),
        LINE_THICK,
    )
    pygame.draw.line(
        surface, COLOR_EYE,
        (cx + half, cy - half), (cx - half, cy + half),
        LINE_THICK,
    )
