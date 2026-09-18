"""눈 그리기 — 치비 애니메이션 스타일 (Simi & Chapchap 느낌).

구조:
- 공막(흰 세로 타원) 안에 큰 홍채 + 작은 동공 + 하이라이트 2개
- 표정은 "모양 교체"가 아니라 **눈꺼풀 파라미터**(상/하 높이, 기울기, 동공/눈 크기,
  눈썹)로 표현 → 매 프레임 lerp로 부드럽게 전환 (EyeParams)
- ^^ / >< / ♥ / ★ / @ 같은 특수 모양은 별도 드로잉 유지 (전환 시 깜빡임이 가려줌)
- 깜빡임은 윗눈꺼풀이 내려오는 것으로 통합 (blink → lid_top override)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import pygame
import pygame.gfxdraw

from src.config import (
    BEHAVIOR, COLOR_BG, COLOR_BLUSH, COLOR_BLUSH_LINE, COLOR_BROW, COLOR_EYE,
    COLOR_HEART, COLOR_HIGHLIGHT, COLOR_IRIS, COLOR_PUPIL, LINE_THICK,
)
from src.face.expressions import EyeShape


def _aa_filled_circle(surface: pygame.Surface, x: int, y: int, r: int, color) -> None:
    """안티앨리어싱된 채워진 원 — gfxdraw 두 함수 조합."""
    if r <= 0:
        return
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def _aa_filled_polygon(surface: pygame.Surface, pts, color) -> None:
    """안티앨리어싱된 채워진 폴리곤."""
    pygame.gfxdraw.filled_polygon(surface, pts, color)
    pygame.gfxdraw.aapolygon(surface, pts, color)


def _aa_filled_ellipse(surface: pygame.Surface, cx: int, cy: int, rx: int, ry: int, color) -> None:
    if rx <= 0 or ry <= 0:
        return
    pygame.gfxdraw.filled_ellipse(surface, cx, cy, rx, ry, color)
    pygame.gfxdraw.aaellipse(surface, cx, cy, rx, ry, color)


# ─── 눈꺼풀/동공 파라미터 (표정 SSOT — 타원 계열 눈에만 적용) ───

@dataclass
class EyeParams:
    """타원 눈 한 세트의 연속 파라미터. 모두 lerp 가능."""

    lid_top: float = 0.0      # 윗눈꺼풀이 덮는 비율 0~1
    lid_bot: float = 0.0      # 아랫눈꺼풀이 덮는 비율 0~1
    lid_tilt: float = 0.0     # 윗눈꺼풀 기울기(deg). +면 안쪽이 내려감(화남), -면 바깥쪽(걱정)
    eye_scale: float = 1.0    # 공막 크기 배율
    pupil_scale: float = 1.0  # 동공 크기 배율 (놀람=작게)
    iris_scale: float = 1.0   # 홍채 크기 배율
    brow: float = 0.0         # 눈썹 표시 강도 0~1
    brow_tilt: float = 0.0    # 눈썹 기울기(deg). +면 안쪽이 내려감(화남)
    brow_lift: float = 0.0    # 눈썹 높이 오프셋(px, +면 위로)
    blush: float = 0.0        # 볼터치 강도 0~1

    def lerp_toward(self, target: "EyeParams", t: float) -> None:
        for k in self.__dataclass_fields__:
            cur = getattr(self, k)
            setattr(self, k, cur + (getattr(target, k) - cur) * t)


_PARAMS: dict[EyeShape, EyeParams] = {
    EyeShape.NORMAL:    EyeParams(lid_top=0.14, brow=1.0, brow_lift=0),
    EyeShape.SURPRISED: EyeParams(eye_scale=1.08, pupil_scale=0.55, iris_scale=0.9,
                                  brow=1.0, brow_lift=8),
    EyeShape.SLEEPY:    EyeParams(lid_top=0.50, lid_bot=0.10, pupil_scale=0.9,
                                  brow=1.0, brow_lift=-6),
    EyeShape.WORRIED:   EyeParams(lid_top=0.24, lid_tilt=-16, eye_scale=0.96,
                                  pupil_scale=0.9, brow=1.0, brow_tilt=-24, brow_lift=2),
    EyeShape.ANGRY:     EyeParams(lid_top=0.36, lid_tilt=24, pupil_scale=0.85,
                                  brow=1.0, brow_tilt=26, brow_lift=-4, blush=0.8),
    # 특수 모양은 타원 파라미터를 안 쓰지만 blush/brow 보조 요소는 참조.
    EyeShape.HAPPY:     EyeParams(blush=1.0, brow=1.0, brow_lift=4),
    EyeShape.SQUINT:    EyeParams(blush=1.0, brow=1.0, brow_lift=6),
    EyeShape.LOVE:      EyeParams(blush=1.0, brow=1.0, brow_lift=6),
    EyeShape.STAR:      EyeParams(blush=0.7, brow=1.0, brow_lift=6),
    EyeShape.DIZZY:     EyeParams(brow=1.0, brow_tilt=-10),
    EyeShape.WINK_LEFT: EyeParams(blush=0.8, brow=1.0, lid_top=0.14),
    EyeShape.WINK_RIGHT: EyeParams(blush=0.8, brow=1.0, lid_top=0.14),
    EyeShape.CLOSED:    EyeParams(lid_top=1.0, brow=1.0),
}

# 파라미터 전환 속도 (0~1, 프레임당). 30fps 기준 ~0.2s에 90% 도달.
PARAM_BLEND = 0.22

# 타원 눈(눈꺼풀 시스템으로 그리는 모양)
_ELLIPSE_SHAPES = {
    EyeShape.NORMAL, EyeShape.SURPRISED, EyeShape.SLEEPY,
    EyeShape.WORRIED, EyeShape.ANGRY, EyeShape.CLOSED,
    EyeShape.WINK_LEFT, EyeShape.WINK_RIGHT,
}


@dataclass
class EyeState:
    """현재 눈 상태."""

    shape: EyeShape = EyeShape.NORMAL
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    blink: float = 0.0          # 0=뜸, 1=완전히 감김
    _next_blink_at: float = 0.0
    _blinking_since: float | None = None

    # 연속 파라미터 (shape에서 도출된 target으로 매 프레임 lerp)
    params: EyeParams = field(default_factory=EyeParams)

    # === Micro-saccades — 사람 눈은 절대 고정되지 않음 ===
    saccade_x: float = 0.0
    saccade_y: float = 0.0
    _saccade_target_x: float = 0.0
    _saccade_target_y: float = 0.0
    _saccade_next_at: float = 0.0


def eye_extent_below(shape: EyeShape, size: int) -> int:
    """현재 눈 모양이 중심에서 얼마나 아래로 뻗는지 (px). 렌더러 레이아웃용."""
    if shape == EyeShape.HAPPY:
        return int(size * 0.30)
    if shape == EyeShape.SQUINT:
        return int(size * 0.25)
    if shape == EyeShape.CLOSED:
        return int(size * 0.1)
    return size // 2


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


def update_params(state: EyeState) -> None:
    """매 렌더 프레임 — shape 목표 파라미터로 부드럽게 수렴."""
    target = _PARAMS.get(state.shape) or _PARAMS[EyeShape.NORMAL]
    state.params.lerp_toward(target, PARAM_BLEND)


# ─── Micro-saccades ───
SACCADE_AMP_X = 0.04
SACCADE_AMP_Y = 0.025
SACCADE_INTERVAL_MIN = 0.10
SACCADE_INTERVAL_MAX = 0.35
SACCADE_BLEND = 0.35


def update_saccade(state: EyeState, now: float) -> None:
    """매 렌더 프레임 호출 — saccade_x/y 갱신."""
    if now >= state._saccade_next_at:
        state._saccade_target_x = random.uniform(-SACCADE_AMP_X, SACCADE_AMP_X)
        state._saccade_target_y = random.uniform(-SACCADE_AMP_Y, SACCADE_AMP_Y)
        state._saccade_next_at = now + random.uniform(
            SACCADE_INTERVAL_MIN, SACCADE_INTERVAL_MAX,
        )
    state.saccade_x += (state._saccade_target_x - state.saccade_x) * SACCADE_BLEND
    state.saccade_y += (state._saccade_target_y - state.saccade_y) * SACCADE_BLEND


# ─── 깔끔한 두꺼운 호 그리기 (mouth.py도 공유) ───

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
    """양쪽 눈 그리기. size = 눈 높이(px)."""
    p = state.params
    shape = state.shape

    # 볼터치 — 눈 아래 바깥쪽. 눈꺼풀 폴리곤이 덮지 않도록 눈 먼저? 아니,
    # 눈꺼풀은 BG색 폴리곤이라 blush를 덮을 수 있어 blush는 마지막에 그림.
    gx = (state.gaze_x + state.saccade_x)
    gy = (state.gaze_y + state.saccade_y)

    if shape in _ELLIPSE_SHAPES:
        left_closed = shape == EyeShape.WINK_LEFT
        right_closed = shape == EyeShape.WINK_RIGHT
        _draw_ellipse_eye(surface, left_center, size, p, gx, gy,
                          inner_sign=+1, force_closed=left_closed, blink=state.blink)
        _draw_ellipse_eye(surface, right_center, size, p, gx, gy,
                          inner_sign=-1, force_closed=right_closed, blink=state.blink)
    else:
        # 특수 모양 — 깜빡임 중엔 감긴 선으로
        closed = state.blink > 0.6
        for center, inner_sign in ((left_center, +1), (right_center, -1)):
            if closed:
                _closed_line(surface, center, size, p, inner_sign)
                continue
            if shape == EyeShape.HAPPY:
                _arc_up(surface, center, size)
            elif shape == EyeShape.SQUINT:
                _angle(surface, center, size, point_right=(inner_sign > 0))
            elif shape == EyeShape.LOVE:
                _heart(surface, center, size)
            elif shape == EyeShape.STAR:
                _star(surface, center, size)
            elif shape == EyeShape.DIZZY:
                _dizzy(surface, center, size)
            else:
                _arc_up(surface, center, size)

    # 눈썹
    if p.brow > 0.05:
        _brow(surface, left_center, size, p, inner_sign=+1)
        _brow(surface, right_center, size, p, inner_sign=-1)

    # 볼터치
    if p.blush > 0.05:
        _blush(surface, left_center, size, p.blush, inner_sign=+1)
        _blush(surface, right_center, size, p.blush, inner_sign=-1)


def _draw_ellipse_eye(
    surface: pygame.Surface,
    center: tuple[int, int],
    size: int,
    p: EyeParams,
    gx: float,
    gy: float,
    *,
    inner_sign: int,
    force_closed: bool,
    blink: float,
) -> None:
    """공막 + 홍채 + 동공 + 하이라이트 + 눈꺼풀."""
    cx, cy = center
    ry = int(size * 0.5 * p.eye_scale)
    rx = ry                                   # 동그란 눈
    if rx < 2 or ry < 2:
        return

    lid_top = max(p.lid_top, blink)
    if force_closed:
        lid_top = 1.0
    lid_bot = p.lid_bot
    if lid_top + lid_bot >= 0.97:
        _closed_line(surface, center, size, p, inner_sign)
        return

    # 공막
    _aa_filled_ellipse(surface, cx, cy, rx, ry, COLOR_EYE)

    # 홍채/동공 — 시선 오프셋. 공막 안에 머물도록 clamp.
    iris_r = int(size * 0.34 * p.iris_scale)
    pupil_r = int(iris_r * 0.56 * p.pupil_scale)
    max_dx = max(0, rx - iris_r + 3)
    max_dy = max(0, ry - iris_r + 3)
    # 기본적으로 살짝 안쪽·아래를 봄 (레퍼런스 느낌) + 시선
    ox = int(max(-max_dx, min(max_dx, gx * rx * 0.55 + inner_sign * rx * 0.06)))
    oy = int(max(-max_dy, min(max_dy, gy * ry * 0.45 + ry * 0.08)))
    ix, iy = cx + ox, cy + oy
    _aa_filled_circle(surface, ix, iy, iris_r, COLOR_IRIS)
    # 동공은 홍채 안에서 시선 쪽으로 조금 더 치우침
    px = ix + int(ox * 0.35)
    py = iy + int(oy * 0.35)
    _aa_filled_circle(surface, px, py, pupil_r, COLOR_PUPIL)
    # 하이라이트 — 좌상단 큰 점 + 우하단 작은 점
    h1 = max(2, int(iris_r * 0.30))
    h2 = max(1, int(iris_r * 0.14))
    _aa_filled_circle(surface, px - int(pupil_r * 0.45), py - int(pupil_r * 0.5), h1, COLOR_HIGHLIGHT)
    _aa_filled_circle(surface, px + int(pupil_r * 0.55), py + int(pupil_r * 0.55), h2, COLOR_HIGHLIGHT)

    # 눈꺼풀 — BG색 폴리곤으로 덮음 (배경이 단색이라 성립)
    top, bottom = cy - ry, cy + ry
    h = 2 * ry
    pad = 6
    if lid_top > 0.01:
        # 안쪽 끝 y / 바깥쪽 끝 y — tilt(deg)로 차이
        tilt_px = math.tan(math.radians(p.lid_tilt)) * rx
        base = top + lid_top * h
        y_inner = base + tilt_px
        y_outer = base - tilt_px
        if inner_sign > 0:   # 왼쪽 눈: 안쪽 = 오른쪽
            pts = [(cx - rx - pad, top - pad), (cx + rx + pad, top - pad),
                   (cx + rx + pad, y_inner), (cx - rx - pad, y_outer)]
        else:
            pts = [(cx - rx - pad, top - pad), (cx + rx + pad, top - pad),
                   (cx + rx + pad, y_outer), (cx - rx - pad, y_inner)]
        pygame.draw.polygon(surface, COLOR_BG, [(int(x), int(y)) for x, y in pts])
        # 눈꺼풀 경계선 — 공막 안쪽 구간만 크림색 두꺼운 선
        if lid_top > 0.06:
            _lid_line(surface, cx, cy, rx, ry, pts[3], pts[2])
    if lid_bot > 0.01:
        y = bottom - lid_bot * h
        pygame.draw.rect(surface, COLOR_BG,
                         pygame.Rect(cx - rx - pad, int(y), 2 * rx + 2 * pad, ry + pad * 2))


def _lid_line(surface, cx, cy, rx, ry, a, b) -> None:
    """눈꺼풀 경계선 — 공막 원 안에 들어오는 구간만 그림."""
    (x1, y1), (x2, y2) = a, b
    pts = []
    for i in range(0, 41):
        t = i / 40
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
            pts.append((int(x), int(y)))
    if len(pts) >= 2:
        th = LINE_THICK + 1
        pygame.draw.line(surface, COLOR_EYE, pts[0], pts[-1], th)
        _aa_filled_circle(surface, pts[0][0], pts[0][1], th // 2, COLOR_EYE)
        _aa_filled_circle(surface, pts[-1][0], pts[-1][1], th // 2, COLOR_EYE)


def _closed_line(
    surface: pygame.Surface, center: tuple[int, int], size: int,
    p: EyeParams, inner_sign: int,
) -> None:
    """감긴 눈 — 기울기 없으면 ∩ 호, 있으면 기울어진 선."""
    cx, cy = center
    if abs(p.lid_tilt) < 3:
        w = int(size * 0.80)
        h = int(size * 0.55)
        rect = pygame.Rect(cx - w // 2, cy - h // 2 + 6, w, h)
        clean_arc(surface, rect, COLOR_EYE, upper=True, thickness=LINE_THICK + 1)
        return
    half = int(size * 0.36)
    tilt_px = int(math.tan(math.radians(p.lid_tilt)) * half * 0.6)
    y_in = cy + tilt_px
    y_out = cy - tilt_px
    if inner_sign > 0:
        a, b = (cx - half, y_out), (cx + half, y_in)
    else:
        a, b = (cx - half, y_in), (cx + half, y_out)
    pygame.draw.line(surface, COLOR_EYE, a, b, LINE_THICK)
    _aa_filled_circle(surface, a[0], a[1], LINE_THICK // 2, COLOR_EYE)
    _aa_filled_circle(surface, b[0], b[1], LINE_THICK // 2, COLOR_EYE)


def _brow(
    surface: pygame.Surface, center: tuple[int, int], size: int,
    p: EyeParams, inner_sign: int,
) -> None:
    """눈썹 — 얇은 ⌒ 호. tilt로 회전, lift로 높이."""
    cx, cy = center
    ry = int(size * 0.5 * p.eye_scale)
    w = int(size * 0.62)
    flat = min(1.0, abs(p.brow_tilt) / 30.0)      # 기울수록 납작한 호
    h = int(size * (0.34 - 0.16 * flat))
    y0 = cy - ry - 8 - int(p.brow_lift)
    tmp = pygame.Surface((w + 12, h + 12), pygame.SRCALPHA)
    rect = pygame.Rect(6, 6, w, h)
    pygame.draw.ellipse(tmp, COLOR_BROW, rect)
    inner = rect.inflate(-6, -6)
    pygame.draw.ellipse(tmp, (0, 0, 0, 0), inner)
    pygame.draw.rect(tmp, (0, 0, 0, 0),
                     pygame.Rect(0, rect.centery, tmp.get_width(), tmp.get_height()))
    # inner_sign>0(왼눈): 안쪽이 오른쪽. tilt +면 안쪽 내려감.
    angle = -p.brow_tilt * inner_sign * 1.3
    if abs(angle) > 0.5:
        tmp = pygame.transform.rotate(tmp, angle)
    tmp.set_alpha(int(255 * max(0.0, min(1.0, p.brow))))
    r = tmp.get_rect(center=(cx, y0 - h // 4))
    surface.blit(tmp, r)


def _blush(
    surface: pygame.Surface, center: tuple[int, int], size: int,
    intensity: float, inner_sign: int,
) -> None:
    cx, cy = center
    rx = int(size * 0.22)
    ry = int(size * 0.10)
    x = cx - inner_sign * int(size * 0.12)
    y = cy + int(size * 0.55)
    color = tuple(int(b + (c - b) * intensity)
                  for c, b in zip(COLOR_BLUSH, COLOR_BG))
    _aa_filled_ellipse(surface, x, y, rx, ry, color)
    # 사선 3개
    if intensity > 0.5:
        lc = tuple(int(b + (c - b) * intensity) for c, b in zip(COLOR_BLUSH_LINE, COLOR_BG))
        for k in (-1, 0, 1):
            x0 = x + k * int(rx * 0.35)
            pygame.draw.line(surface, lc, (x0 + 3, y - ry + 2), (x0 - 3, y + ry - 2), 2)


# ─── 특수 모양 헬퍼 ───

def _arc_up(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """위로 휘어진 호 — ⌒ (행복)."""
    cx, cy = center
    w = int(size * 0.95)
    h = int(size * 0.80)
    rect = pygame.Rect(cx - w // 2, cy - h // 2 + 10, w, h)
    clean_arc(surface, rect, COLOR_EYE, upper=True, thickness=LINE_THICK + 3)


def _angle(
    surface: pygame.Surface, center: tuple[int, int], size: int, point_right: bool,
) -> None:
    """꺾인 선 — `>` 또는 `<` (신남/찡그린 웃음)."""
    cx, cy = center
    half = int(size * 0.40)
    thick = LINE_THICK * 2 + 1
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
    """흰 원 안에 핑크 하트."""
    cx, cy = center
    R = int(size * 0.5)
    _aa_filled_circle(surface, cx, cy, R, COLOR_EYE)
    s = int(size * 0.30)
    lobe_r = int(s * 0.55)
    lobe_dx = int(s * 0.5)
    top_y = cy - s // 3
    triangle_pts = [
        (cx - lobe_dx - lobe_r + 2, top_y + 2),
        (cx + lobe_dx + lobe_r - 2, top_y + 2),
        (cx, cy + s),
    ]
    _aa_filled_polygon(surface, triangle_pts, COLOR_HEART)
    _aa_filled_circle(surface, cx - lobe_dx, top_y, lobe_r, COLOR_HEART)
    _aa_filled_circle(surface, cx + lobe_dx, top_y, lobe_r, COLOR_HEART)
    _aa_filled_circle(surface, cx - lobe_dx - lobe_r // 3, top_y - lobe_r // 3,
                      max(2, lobe_r // 3), COLOR_HIGHLIGHT)


def _star(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """5각 별 (AA)."""
    cx, cy = center
    outer_r = int(size * 0.42)
    inner_r = outer_r * 0.42
    points: list[tuple[int, int]] = []
    for i in range(10):
        angle = -math.pi / 2 + i * math.pi / 5
        r = outer_r if i % 2 == 0 else inner_r
        points.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    _aa_filled_polygon(surface, points, (255, 220, 90))
    _aa_filled_circle(surface, cx - outer_r // 4, cy - outer_r // 4,
                      max(2, outer_r // 6), COLOR_HIGHLIGHT)


def _dizzy(surface: pygame.Surface, center: tuple[int, int], size: int) -> None:
    """소용돌이 — 나선."""
    cx, cy = center
    r_max = int(size * 0.40)
    pts = []
    turns = 2.5
    steps = 60
    for i in range(steps + 1):
        t = i / steps
        ang = t * turns * 2 * math.pi
        r = r_max * t
        pts.append((int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))))
    pygame.draw.lines(surface, COLOR_EYE, False, pts, LINE_THICK)
