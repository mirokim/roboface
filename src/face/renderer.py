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
from src.face import extras, eyes, mouth
from src.face.expressions import Expression, NEUTRAL


@dataclass
class FaceState:
    """렌더러가 그릴 현재 얼굴 상태."""

    expression: Expression = NEUTRAL
    eye_state: eyes.EyeState = field(default_factory=eyes.EyeState)
    mouth_state: mouth.MouthState = field(default_factory=mouth.MouthState)
    recording: bool = False
    brightness: float = 1.0   # 0.0~1.0

    # 환경 반응 (thermal_state task가 갱신)
    sweat_intensity: float = 0.0     # 0~1, 얼굴 옆에 땀방울 표시
    shiver_intensity: float = 0.0    # 0~1, 얼굴 좌우 미세 떨림

    # 말풍선 — speech_text가 있고 now < speech_until 이면 상단에 표시
    speech_text: str | None = None
    speech_until: float = 0.0

    def apply_expression(self, expr: Expression) -> None:
        # 표정이 실제로 바뀌면 깜빡임으로 자연스럽게 전환
        changed = self.expression.name != expr.name
        self.expression = expr
        self.eye_state.shape = expr.eye
        self.mouth_state.shape = expr.mouth
        if changed:
            eyes.trigger_blink(self.eye_state, time.time())

    def show_speech(self, text: str, duration_sec: float) -> None:
        """발화 시작 시 호출 — 말풍선 표시 시작."""
        self.speech_text = text
        self.speech_until = time.time() + duration_sec + 1.0  # 음성 끝나고 1초 더

    def clear_speech(self) -> None:
        self.speech_text = None
        self.speech_until = 0.0


def draw_face_to_surface(canvas: pygame.Surface, face: FaceState) -> None:
    """얼굴 한 프레임을 surface에 그림. PygameRenderer와 LCDRenderer 공용.

    canvas는 (DISPLAY_WIDTH, DISPLAY_HEIGHT) 크기.
    """
    now = time.time()
    eyes.update_blink(face.eye_state, now)
    eyes.update_saccade(face.eye_state, now)

    bg = tuple(int(c * face.brightness) for c in COLOR_BG)
    canvas.fill(bg)

    # 추울 때 떨림 오프셋 — 얼굴 전체에 적용
    sh_dx, sh_dy = extras.shiver_offset(face.shiver_intensity, now)

    # === 얼굴 위치 동적 계산 ===
    eye_size = 56
    eye_offset = int(DISPLAY_WIDTH * 0.20)

    eye_extent_above = eye_size // 2
    eye_extent_below = eyes.eye_extent_below(face.expression.eye, eye_size)
    gap = max(eye_extent_below, eye_size // 3) + 10
    mouth_estimated_below = 12

    face_block_height = eye_extent_above + gap + mouth_estimated_below
    face_top = (DISPLAY_HEIGHT - face_block_height) // 2
    eye_y = face_top + eye_extent_above
    mouth_y = eye_y + gap

    # 말풍선 떠 있을 때 얼굴 아래로 살짝 내림 (겹침 방지)
    speech_active = face.speech_text and now < face.speech_until
    if speech_active:
        speech_drop = 14
        eye_y += speech_drop
        mouth_y += speech_drop

    left_eye = (DISPLAY_WIDTH // 2 - eye_offset + sh_dx, eye_y + sh_dy)
    right_eye = (DISPLAY_WIDTH // 2 + eye_offset + sh_dx, eye_y + sh_dy)

    eyes.draw_eyes(canvas, face.eye_state, left_eye, right_eye, size=eye_size)

    mouth_center = (DISPLAY_WIDTH // 2 + sh_dx, mouth_y + sh_dy)
    mouth.draw_mouth(canvas, face.mouth_state, mouth_center, width=50)

    # 땀방울
    if face.sweat_intensity > 0.05:
        extras.draw_sweat(canvas, face.sweat_intensity,
                          right_eye, eye_size, now)

    # 말풍선 (얼굴 그리고 나서 상단에 덮어 그림)
    if speech_active:
        extras.draw_speech_bubble(canvas, face.speech_text)
    elif face.speech_text and now >= face.speech_until:
        # 시간 지남 — 자동 클리어
        face.speech_text = None
        face.speech_until = 0.0

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
        draw_face_to_surface(self.canvas, face)
        # smoothscale로 확대 시 안티앨리어싱 적용 (jaggies 완화)
        scaled = pygame.transform.smoothscale(self.canvas, self.size)
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
