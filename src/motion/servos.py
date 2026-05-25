"""서보 제어 — Pan/Tilt 머리 움직임.

simulator 모드: MockServoController (콘솔 로그만)
robot 모드: PCA9685ServoController (실제 하드웨어)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.config import (
    PAN_CENTER_DEG, PAN_MAX_DEG, PAN_MIN_DEG,
    PCA9685_ADDRESS,
    SERVO_PAN_CHANNEL, SERVO_RANGE_DEG,
    SERVO_PULSE_MAX_US, SERVO_PULSE_MIN_US,
    SERVO_TILT_CHANNEL,
    TILT_CENTER_DEG, TILT_MAX_DEG, TILT_MIN_DEG,
    is_robot,
)
from src.utils.logger import get_logger

log = get_logger("servos")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class HeadPosition:
    pan: float = PAN_CENTER_DEG    # 좌우
    tilt: float = TILT_CENTER_DEG  # 상하


class ServoController(ABC):
    """서보 컨트롤러 추상 인터페이스."""

    # 비상 cap — 한 호출당 변화량 한계. 정상 동작은 영향 X:
    # home 45°, look_around 30°, dance amp ±15°, head_tracker step ±8°.
    # 비정상(버그/agent 오작동/발산)으로 큰 점프 명령이 와도 ±50°만 진행.
    _MAX_DELTA_PER_CALL_DEG = 50.0

    def __init__(self) -> None:
        self.position = HeadPosition()

    def _safe_clamp(self, pan: float, tilt: float) -> tuple[float, float]:
        """비상 변화 cap. subclass set_angles 안에서 호출. log.warning으로 알림."""
        d_pan = pan - self.position.pan
        d_tilt = tilt - self.position.tilt
        cap = self._MAX_DELTA_PER_CALL_DEG
        if abs(d_pan) > cap:
            log.warning(f"servo pan delta {d_pan:+.1f}° → clamp ±{cap:.0f}")
            pan = self.position.pan + (cap if d_pan > 0 else -cap)
        if abs(d_tilt) > cap:
            log.warning(f"servo tilt delta {d_tilt:+.1f}° → clamp ±{cap:.0f}")
            tilt = self.position.tilt + (cap if d_tilt > 0 else -cap)
        return pan, tilt

    @abstractmethod
    def set_angles(self, pan: float, tilt: float) -> None: ...

    def home(self) -> None:
        """중앙으로."""
        self.set_angles(PAN_CENTER_DEG, TILT_CENTER_DEG)

    def look(self, dx: float, dy: float) -> None:
        """상대 이동 (-1.0 ~ 1.0)."""
        pan_range = (PAN_MAX_DEG - PAN_MIN_DEG) / 2
        tilt_range = (TILT_MAX_DEG - TILT_MIN_DEG) / 2
        new_pan = PAN_CENTER_DEG + dx * pan_range
        new_tilt = TILT_CENTER_DEG + dy * tilt_range
        self.set_angles(new_pan, new_tilt)

    def smooth_to(self, pan: float, tilt: float, duration_sec: float = 0.5) -> None:
        """부드럽게 보간 이동 (블로킹). asyncio 환경에선 smooth_to_async 사용."""
        steps = max(2, int(duration_sec * 30))
        start_pan = self.position.pan
        start_tilt = self.position.tilt
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            self.set_angles(
                start_pan + (pan - start_pan) * t,
                start_tilt + (tilt - start_tilt) * t,
            )
            time.sleep(duration_sec / steps)

    async def smooth_to_async(
        self, pan: float, tilt: float, duration_sec: float = 0.5,
    ) -> None:
        """async 버전 — 이벤트 루프 블로킹 X."""
        import asyncio
        steps = max(2, int(duration_sec * 30))
        start_pan = self.position.pan
        start_tilt = self.position.tilt
        period = duration_sec / steps
        for i in range(1, steps + 1):
            t = i / steps
            t = t * t * (3 - 2 * t)
            self.set_angles(
                start_pan + (pan - start_pan) * t,
                start_tilt + (tilt - start_tilt) * t,
            )
            await asyncio.sleep(period)


class MockServoController(ServoController):
    """시뮬레이터용 — 로그만 출력."""

    def set_angles(self, pan: float, tilt: float) -> None:
        pan = _clamp(pan, PAN_MIN_DEG, PAN_MAX_DEG)
        tilt = _clamp(tilt, TILT_MIN_DEG, TILT_MAX_DEG)
        pan, tilt = self._safe_clamp(pan, tilt)
        if abs(pan - self.position.pan) > 0.5 or abs(tilt - self.position.tilt) > 0.5:
            log.debug(f"[MockServo] pan={pan:.1f}° tilt={tilt:.1f}°")
        self.position.pan = pan
        self.position.tilt = tilt


class PCA9685ServoController(ServoController):
    """실제 PCA9685 + SG90 180° 서보 제어 + stall 보호."""

    # stall 보호 — 같은 각도로 머문 지 STALL_TIMEOUT_SEC 지나면 신호 끔.
    # 서보가 위치 유지 위해 미세 전류 흘리는데, 못 가는 위치면 큰 전류 지속 → 발열.
    STALL_TIMEOUT_SEC = 0.8

    def __init__(self) -> None:
        super().__init__()
        # robot 모드에서만 import
        from adafruit_servokit import ServoKit  # type: ignore[import-not-found]

        self.kit = ServoKit(channels=16, address=PCA9685_ADDRESS)
        for ch in (SERVO_PAN_CHANNEL, SERVO_TILT_CHANNEL):
            self.kit.servo[ch].set_pulse_width_range(
                SERVO_PULSE_MIN_US, SERVO_PULSE_MAX_US,
            )
            self.kit.servo[ch].actuation_range = SERVO_RANGE_DEG
        self._last_change_at = time.monotonic()
        self._signal_active = True
        self.home()
        log.info("PCA9685 서보 초기화 완료")

    def set_angles(self, pan: float, tilt: float) -> None:
        pan = _clamp(pan, PAN_MIN_DEG, PAN_MAX_DEG)
        tilt = _clamp(tilt, TILT_MIN_DEG, TILT_MAX_DEG)
        pan, tilt = self._safe_clamp(pan, tilt)
        # 명령 보낼 때만 신호 재활성 + 시간 갱신
        moved = (
            abs(pan - self.position.pan) > 0.5
            or abs(tilt - self.position.tilt) > 0.5
            or not self._signal_active
        )
        self.kit.servo[SERVO_PAN_CHANNEL].angle = pan
        self.kit.servo[SERVO_TILT_CHANNEL].angle = tilt
        self.position.pan = pan
        self.position.tilt = tilt
        if moved:
            self._last_change_at = time.monotonic()
            self._signal_active = True
        # 멈춰있은지 STALL_TIMEOUT 지났으면 신호 풀어줘 발열 방지
        elif (time.monotonic() - self._last_change_at) > self.STALL_TIMEOUT_SEC:
            try:
                self.kit.servo[SERVO_PAN_CHANNEL].angle = None
                self.kit.servo[SERVO_TILT_CHANNEL].angle = None
                self._signal_active = False
            except Exception:
                pass


def create_controller() -> ServoController:
    """모드에 맞는 컨트롤러 생성."""
    if is_robot():
        try:
            return PCA9685ServoController()
        except Exception as e:
            log.warning(f"PCA9685 초기화 실패, Mock으로 폴백: {e}")
            return MockServoController()
    return MockServoController()
