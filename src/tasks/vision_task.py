"""Vision task — IMX500 카메라 스트림 → person presence + perception 업데이트."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from src.brain.perception import PerceptionState
from src.sensors.base import SensorEvent
from src.utils.logger import get_logger
from src.vision.person_detector import PersonDetector

log = get_logger("vision_task")


async def run_vision(
    emit_event: Callable[[SensorEvent], None],
    perception: PerceptionState | None = None,
) -> None:
    """카메라 스트림 → person detector → sensor events + perception 업데이트."""
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

            # PerceptionState 업데이트 — 사람 위치/거리 매 프레임 추적
            if perception is not None:
                person_dets = [
                    d for d in detections
                    if d.class_name == "person" and d.confidence >= 0.5
                ]
                if person_dets:
                    biggest = max(
                        person_dets,
                        key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                    )
                    perception.update_person(
                        bbox=biggest.bbox,
                        distance_cm=detector._last_distance or -1.0,
                    )
                elif (
                    perception.person_present
                    and time.time() - perception.last_person_seen_at > detector.away_timeout_sec
                ):
                    perception.clear_person()
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
