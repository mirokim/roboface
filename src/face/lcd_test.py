"""LCD 단순 동작 확인 — 빨강/녹색/파랑 + 얼굴 1프레임.

실행:
    python -m src.face.lcd_test

연결 검증용. 화면에 색이 뜨면 SPI/핀맵/라이브러리 OK.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
from PIL import Image

from src.config import DISPLAY_HEIGHT, DISPLAY_WIDTH
from src.face.expressions import HAPPY, NEUTRAL, SURPRISED
from src.face.renderer import FaceState, draw_face_to_surface


def _open_lcd():
    import board
    import digitalio
    from adafruit_rgb_display import ili9341

    cs_pin = digitalio.DigitalInOut(board.CE0)
    dc_pin = digitalio.DigitalInOut(board.D25)
    rst_pin = digitalio.DigitalInOut(board.D27)
    bl_pin = digitalio.DigitalInOut(board.D18)
    bl_pin.direction = digitalio.Direction.OUTPUT
    bl_pin.value = True

    spi = board.SPI()
    disp = ili9341.ILI9341(
        spi, cs=cs_pin, dc=dc_pin, rst=rst_pin,
        baudrate=24_000_000, rotation=90,
        width=240, height=320,
    )
    return disp


def main() -> None:
    print("LCD 초기화 중...")
    disp = _open_lcd()
    print(f"LCD OK — {disp.width}×{disp.height}")

    # 1. RGB 컬러 테스트
    for color_name, rgb in [("RED", (255, 0, 0)), ("GREEN", (0, 255, 0)),
                              ("BLUE", (0, 0, 255)), ("WHITE", (255, 255, 255)),
                              ("BLACK", (0, 0, 0))]:
        print(f"  → {color_name}")
        img = Image.new("RGB", (disp.width, disp.height), rgb)
        disp.image(img)
        time.sleep(0.8)

    # 2. 얼굴 한 프레임 (NEUTRAL)
    print("얼굴 그리기 (NEUTRAL)")
    pygame.init()
    pygame.display.set_mode((1, 1))
    canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))
    face = FaceState(expression=NEUTRAL)
    draw_face_to_surface(canvas, face)
    raw = pygame.image.tostring(canvas, "RGB")
    pil = Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
    disp.image(pil)
    time.sleep(2)

    # 3. HAPPY
    print("얼굴 그리기 (HAPPY)")
    face.apply_expression(HAPPY)
    draw_face_to_surface(canvas, face)
    raw = pygame.image.tostring(canvas, "RGB")
    pil = Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
    disp.image(pil)
    time.sleep(2)

    # 4. SURPRISED
    print("얼굴 그리기 (SURPRISED)")
    face.apply_expression(SURPRISED)
    draw_face_to_surface(canvas, face)
    raw = pygame.image.tostring(canvas, "RGB")
    pil = Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
    disp.image(pil)
    time.sleep(2)

    print("테스트 완료. 화면 클리어 (검정)")
    img = Image.new("RGB", (disp.width, disp.height), (0, 0, 0))
    disp.image(img)


if __name__ == "__main__":
    main()
