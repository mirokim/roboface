"""Vision task — IMX500 카메라 스트림에서 person presence 추출하여 센서 이벤트로 변환.

mmWave와 병행 동작. 둘 다 PRESENCE_NEW/LEFT 발생시키니
work_tracker는 자연스럽게 통합 받음.

카메라 부품/모델 누락 시 graceful skip.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from src.sensors.base import SensorEvent
from src.utils.logger import get_logger
from src.vision.person_detector import PersonDetector

log = get_logger("vision_task")


async def run_vision(emit_event: Callable[[SensorEvent], None]) -> None:
    """카메라 스트림 → person detector → sensor events.

    `emit_event` 는 sensor manager의 events queue에 push하는 콜백.
    """
    try:
        from src.vision.camera import IMX500Camera
    except Exception as e:
        log.warning(f"vision 모듈 import 실패 (헤드리스 모드 유지): {e}")
        return

    try:
        cam = IMX500Camera()
    except Exception as e:
        log.warning(f"AI Camera 초기화 실패 — vision task 건너뜀: {e}")
        return

    detector = PersonDetector()
    log.info("vision task 시작 (IMX500 person detection)")

    try:
        async for detections in cam.stream():
            events = detector.process(detections)
            for ev in events:
                emit_event(ev)
    except asyncio.CancelledError:
        log.info("vision task 취소 요청")
        raise
    except Exception as e:
        log.warning(f"vision task 에러: {e}")
    finally:
        try:
            cam.close()
        except Exception:
            pass
