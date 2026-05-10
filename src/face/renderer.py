"""얼굴 렌더러 — Pygame 창(simulator) 또는 LCD(robot) 모두 동일 인터페이스.

도형 그리기 방식 (eyes.py + mouth.py).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pygame

from src.config import (
    COLOR_BG, COLOR_INDICATOR_REC,
    DISPLAY_HEIGHT, DISPLAY_WIDTH, FPS,
)
from src.face import eyes, mouth
from src.face.expressions import Expression, NEUTRAL


@dataclass
class FaceState:
    """렌더러가 그릴 현재 얼굴 상태."""

    expression: Expression = NEUTRAL
    eye_state: eyes.EyeState = field(default_factory=eyes.EyeState)
    mouth_state: mouth.MouthState = field(default_factory=mouth.MouthState)
    recording: bool = False
    brightness: float = 1.0   # 0.0~1.0

    def apply_expression(self, expr: Expression) -> None:
        # 표정이 실제로 바뀌면 깜빡임으로 자연스럽게 전환
        changed = self.expression.name != expr.name
        self.expression = expr
        self.eye_state.shape = expr.eye
        self.mouth_state.shape = expr.mouth
        if changed:
            eyes.trigger_blink(self.eye_state, time.time())


class PygameRenderer:
    """Pygame 창에 얼굴 그리는 렌더러 (PC simulator)."""

    def __init__(self, scale: int = 3, title: str = "Roboface Simulator"):
        pygame.init()
        self.scale = scale
        self.size = (DISPLAY_WIDTH * scale, DISPLAY_HEIGHT * scale)
        self.window = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))

    def render(self, face: FaceState) -> None:
        now = time.time()
        eyes.update_blink(face.eye_state, now)

        canvas = self.canvas
        bg = tuple(int(c * face.brightness) for c in COLOR_BG)
        canvas.fill(bg)

        # === 얼굴 위치 동적 계산 ===
        eye_size = 56
        eye_offset = int(DISPLAY_WIDTH * 0.20)

        # 표정마다 눈/입 차지 영역이 다름 → 전체 블록을 화면 중앙에 정렬
        eye_extent_above = eye_size // 2   # 대부분 모양에서 위쪽 반지름은 ~half
        eye_extent_below = eyes.eye_extent_below(face.expression.eye, eye_size)
        gap = max(eye_extent_below, eye_size // 3) + 10  # 눈 중심 → 입 중심
        mouth_estimated_below = 12  # 입 중심에서 아래로 차지하는 평균

        face_block_height = eye_extent_above + gap + mouth_estimated_below
        face_top = (DISPLAY_HEIGHT - face_block_height) // 2
        eye_y = face_top + eye_extent_above
        mouth_y = eye_y + gap

        left_eye = (DISPLAY_WIDTH // 2 - eye_offset, eye_y)
        right_eye = (DISPLAY_WIDTH // 2 + eye_offset, eye_y)

        eyes.draw_eyes(canvas, face.eye_state, left_eye, right_eye, size=eye_size)

        mouth_center = (DISPLAY_WIDTH // 2, mouth_y)
        mouth.draw_mouth(canvas, face.mouth_state, mouth_center, width=50)

        # 녹음 인디케이터
        if face.recording:
            t = (now * 2) % 1.0
            alpha = int(80 + 175 * abs(0.5 - t) * 2)
            color = (*COLOR_INDICATOR_REC[:3], alpha)
            indicator = pygame.Surface((12, 12), pygame.SRCALPHA)
            pygame.draw.circle(indicator, color, (6, 6), 6)
            canvas.blit(indicator, (DISPLAY_WIDTH - 18, 6))

        if face.brightness < 0.05:
            canvas.fill((0, 0, 0))

        scaled = pygame.transform.scale(canvas, self.size)
        self.window.blit(scaled, (0, 0))
        pygame.display.flip()
        self.clock.tick(FPS)

    def poll_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self) -> None:
        pygame.quit()
