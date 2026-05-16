"""AI Camera (Sony IMX500) 래퍼 — picamera2 + 온칩 NPU.

객체 감지 결과를 stream으로 제공. Pi CPU 부담 거의 없음.

사용 예:
    cam = IMX500Camera()
    for detections in cam.stream():
        for d in detections:
            print(d.class_name, d.confidence, d.bbox)

기본 모델: MobileNet SSD (COCO 80 classes, "person" 포함).
모델 파일은 imx500-all apt 패키지로 설치됨 (`/usr/share/imx500-models/`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger("camera")


DEFAULT_MODEL_PATH = (
    "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
)

# COCO class names (MobileNet SSD 기본 클래스). 모델이 label 파일을 제공하면 그게 우선.
COCO_CLASSES = [
    "background",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    """단일 객체 감지 결과."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) 정규화 0~1
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        return (f"Detection({self.class_name} {self.confidence:.2f} "
                f"@ {self.bbox})")


class IMX500Camera:
    """IMX500 AI Camera 추론 스트림."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = 0.5,
        target_fps: float = 5.0,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"IMX500 model not found: {model_path}\n"
                "  → sudo apt install -y imx500-all"
            )

        # robot 모드에서만 의존성 import
        from picamera2 import Picamera2
        from picamera2.devices import IMX500

        self.imx500 = IMX500(model_path)
        self.cam = Picamera2(self.imx500.camera_num)
        config = self.cam.create_preview_configuration(
            controls={"FrameRate": target_fps},
            buffer_count=4,
        )
        # 펌웨어 로드 (시간 걸림)
        try:
            self.imx500.show_network_fw_progress_bar()
        except Exception:
            pass
        self.cam.start(config, show_preview=False)

        # 클래스 라벨 로드
        intrinsics = getattr(self.imx500, "network_intrinsics", None)
        labels = getattr(intrinsics, "labels", None) if intrinsics else None
        self.labels: list[str] = labels if labels else COCO_CLASSES

        self.threshold = confidence_threshold
        self.target_fps = target_fps

        log.info(f"IMX500 카메라 초기화 완료 ({model_path}, "
                 f"threshold={confidence_threshold}, target_fps={target_fps})")

    def _get_detections(self) -> list[Detection]:
        """현재 프레임에서 감지된 객체 리스트. 비동기 X — 동기 호출용."""
        try:
            metadata = self.cam.capture_metadata()
        except Exception as e:
            log.debug(f"메타데이터 캡처 실패: {e}")
            return []

        outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if outputs is None or len(outputs) < 3:
            return []

        try:
            boxes = outputs[0][0]
            scores = outputs[1][0]
            classes = outputs[2][0]
        except (IndexError, TypeError) as e:
            log.debug(f"출력 파싱 실패: {e}")
            return []

        detections: list[Detection] = []
        for box, score, cls in zip(boxes, scores, classes):
            try:
                conf = float(score)
            except (TypeError, ValueError):
                continue
            if conf < self.threshold:
                continue
            cls_id = int(cls)
            name = self.labels[cls_id] if 0 <= cls_id < len(self.labels) else f"class_{cls_id}"
            try:
                bbox = tuple(float(v) for v in box[:4])
            except (TypeError, IndexError):
                bbox = (0.0, 0.0, 0.0, 0.0)
            detections.append(Detection(
                class_id=cls_id,
                class_name=name,
                confidence=conf,
                bbox=bbox,
            ))
        return detections

    async def stream(self) -> AsyncIterator[list[Detection]]:
        """비동기 감지 스트림. target_fps 간격으로 yield."""
        period = 1.0 / max(0.5, self.target_fps)
        loop = asyncio.get_event_loop()
        while True:
            # 카메라 캡처는 blocking — executor에 던짐
            try:
                detections = await loop.run_in_executor(None, self._get_detections)
            except Exception as e:
                log.warning(f"detection 에러: {e}")
                detections = []
            yield detections
            await asyncio.sleep(period)

    def get_main_frame(self):
        """현재 main stream의 raw RGB 프레임 (HxWx3 numpy uint8) 또는 None.

        wave_detector 같은 사후 분석용. detection metadata와는 별도 stream.
        """
        try:
            return self.cam.capture_array("main")
        except Exception as e:
            log.warning(f"capture_array 실패: {e}")
            return None

    def close(self) -> None:
        try:
            self.cam.stop()
        except Exception:
            pass
        log.info("IMX500 카메라 종료")
