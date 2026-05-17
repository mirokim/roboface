"""LCD 렌더러 — Waveshare 2.4" ILI9341 (SPI).

Pi 5에서 헤드리스로 동작. pygame Surface를 메모리에 그리고
PIL Image로 변환 후 SPI로 LCD 전송.

핀맵 (config.py 기준):
  VCC  → 3.3V
  GND  → GND
  DIN  → GPIO 10 (MOSI / SPI0)
  CLK  → GPIO 11 (SCLK / SPI0)
  CS   → GPIO  8 (CE0)
  DC   → GPIO 25
  RST  → GPIO 27
  BL   → GPIO 18
"""

from __future__ import annotations

import asyncio
import os

# Pi 헤드리스에서 pygame이 SDL 디스플레이 없이 동작하도록
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402

from src.config import DISPLAY_HEIGHT, DISPLAY_WIDTH  # noqa: E402
from src.face.renderer import FaceState, draw_face_to_surface  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

log = get_logger("lcd")


class LCDRenderer:
    """ILI9341 2.4" LCD 렌더러.

    pygame Surface에 그린 후 SPI로 LCD 전송.
    """

    def __init__(self, rotation: int = 90, baudrate: int = 32_000_000):
        # Pi GPIO/SPI 의존성은 robot 모드에서만 import
        import board
        import digitalio
        from adafruit_rgb_display import ili9341
        from PIL import Image  # noqa: F401  (간접 사용 확인)

        # 핀 매핑 — Pi 5는 CE0(GPIO8)이 spidev 드라이버에 점유되어
        # digitalio로 못 잡음. CS는 free GPIO (GPIO 5)로 우회.
        cs_pin = digitalio.DigitalInOut(board.D5)
        dc_pin = digitalio.DigitalInOut(board.D25)
        rst_pin = digitalio.DigitalInOut(board.D27)
        spi = board.SPI()

        # ILI9341은 240×320 네이티브, rotation=90으로 320×240(가로)
        # 우리 DISPLAY_WIDTH/HEIGHT가 320×240이라 일치
        self.disp = ili9341.ILI9341(
            spi,
            cs=cs_pin,
            dc=dc_pin,
            rst=rst_pin,
            baudrate=baudrate,
            rotation=rotation,
            width=240,
            height=320,
        )

        # 백라이트
        self.bl = digitalio.DigitalInOut(board.D18)
        self.bl.direction = digitalio.Direction.OUTPUT
        self.bl.value = True

        # pygame 초기화 (off-screen)
        pygame.init()
        pygame.display.set_mode((1, 1))   # SDL이 surface 만들려면 video init 필요
        self.canvas = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT))

        # PIL Image — pygame surface → bytes → PIL → LCD
        from PIL import Image
        self._Image = Image

        log.info(f"LCD 초기화 완료 (rotation={rotation}, "
                 f"baud={baudrate / 1e6:.0f}MHz, {DISPLAY_WIDTH}×{DISPLAY_HEIGHT})")

    def render(self, face: FaceState) -> None:
        """동기 렌더 (호환용) — SPI 전송이 이벤트 루프 블로킹."""
        draw_face_to_surface(self.canvas, face)
        raw = pygame.image.tostring(self.canvas, "RGB")
        pil = self._Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
        self.disp.image(pil)

    async def render_async(self, face: FaceState) -> None:
        """비동기 렌더 — SPI 전송을 executor로 빼서 이벤트 루프 안 멈추게.

        pygame 캔버스 그리기는 main thread (pygame이 thread-safe 아님),
        SPI 전송만 백그라운드 thread.
        """
        draw_face_to_surface(self.canvas, face)
        raw = pygame.image.tostring(self.canvas, "RGB")
        pil = self._Image.frombytes("RGB", (DISPLAY_WIDTH, DISPLAY_HEIGHT), raw)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.disp.image, pil)

    def set_backlight(self, on: bool) -> None:
        self.bl.value = on

    def close(self) -> None:
        try:
            self.set_backlight(False)
        except Exception:
            pass
        try:
            self.disp.fill(0)   # 화면 끔
        except Exception:
            pass
        pygame.quit()
