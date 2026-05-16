"""AI Camera (Sony IMX500) 래퍼 — picamera2 + 온칩 NPU.

두 모드 지원:
- mode="detect" (기본): SSD MobileNet으로 객체 감지. detections로 person bbox 추출.
- mode="pose": HigherHRNet으로 자세 추정. detections 안의 keypoints에 17개 COCO
  관절 (x, y, confidence) 포함. bbox는 keypoints의 minmax로 자동 계산.

사용 예:
    cam = IMX500Camera(mode="pose")
    async for detections in cam.stream():
        for d in detections:
            if d.keypoints is not None:
                left_wrist = d.keypoints[9]   # COCO idx 9 = 왼손목
                right_wrist = d.keypoints[10] # idx 10 = 오른손목

COCO 17 keypoint 순서:
0=nose, 1=l_eye, 2=r_eye, 3=l_ear, 4=r_ear,
5=l_shoulder, 6=r_shoulder, 7=l_elbow, 8=r_elbow,
9=l_wrist, 10=r_wrist, 11=l_hip, 12=r_hip,
13=l_knee, 14=r_knee, 15=l_ankle, 16=r_ankle

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

POSE_MODEL_PATH = (
    "/usr/share/imx500-models/imx500_network_higherhrnet_coco.rpk"
)

# COCO 17 keypoint 인덱스
KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

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
    """단일 객체 감지 결과 (객체 감지 or 자세 추정)."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) 정규화 0~1
    timestamp: float = field(default_factory=time.time)
    # 자세 추정 모드일 때 17개 COCO keypoint: shape (17, 3) — (x, y, conf) 정규화 0~1
    keypoints: Any = None

    def __repr__(self) -> str:
        suffix = " +kp" if self.keypoints is not None else ""
        return (f"Detection({self.class_name} {self.confidence:.2f} "
                f"@ {self.bbox}{suffix})")


class IMX500Camera:
    """IMX500 AI Camera 추론 스트림.

    mode: "detect" (객체 감지, SSD MobileNet) 또는 "pose" (자세 추정, HigherHRNet).
    """

    def __init__(
        self,
        model_path: str | None = None,
        confidence_threshold: float | None = None,
        target_fps: float = 5.0,
        mode: str = "detect",
    ):
        # pose 모드는 보통 0.3 정도가 적절 (공식 데모 기본값)
        if confidence_threshold is None:
            confidence_threshold = 0.3 if mode == "pose" else 0.5
        if mode not in ("detect", "pose"):
            raise ValueError(f"mode는 'detect' 또는 'pose' (got '{mode}')")
        if model_path is None:
            model_path = POSE_MODEL_PATH if mode == "pose" else DEFAULT_MODEL_PATH
        if not Path(model_path).exists():
            hint = "imx500-all" if mode == "detect" else "imx500-models"
            raise FileNotFoundError(
                f"IMX500 model not found: {model_path}\n"
                f"  → sudo apt install -y {hint}"
            )

        # robot 모드에서만 의존성 import
        from picamera2 import Picamera2
        from picamera2.devices import IMX500

        self.mode = mode
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

        # 클래스 라벨 — pose 모드는 항상 person
        intrinsics = getattr(self.imx500, "network_intrinsics", None)
        labels = getattr(intrinsics, "labels", None) if intrinsics else None
        self.labels: list[str] = labels if labels else COCO_CLASSES

        # pose post-processor lazy import (picamera2 모듈명: postprocess_highernet)
        self._pose_postprocess = None
        if mode == "pose":
            try:
                from picamera2.devices.imx500.postprocess_highernet import (
                    postprocess_higherhrnet,
                )
                self._pose_postprocess = postprocess_higherhrnet
            except ImportError as e:
                raise RuntimeError(
                    f"pose 모드는 picamera2 postprocess_highernet 필요: {e}"
                ) from e

        self.threshold = confidence_threshold
        self.target_fps = target_fps

        log.info(f"IMX500 카메라 초기화 완료 ({model_path}, mode={mode}, "
                 f"threshold={confidence_threshold}, target_fps={target_fps})")

    def _get_detections(self) -> list[Detection]:
        """현재 프레임에서 감지된 객체 리스트. 비동기 X — 동기 호출용."""
        try:
            metadata = self.cam.capture_metadata()
        except Exception as e:
            log.debug(f"메타데이터 캡처 실패: {e}")
            return []

        if self.mode == "pose":
            return self._get_pose_detections(metadata)
        return self._get_object_detections(metadata)

    def _get_object_detections(self, metadata: dict) -> list[Detection]:
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

    # HigherHRNet 후처리 — 640x480 입력 (H, W) 기준
    POSE_IMG_H = 480
    POSE_IMG_W = 640

    _pose_diag_last = 0.0

    def _get_pose_detections(self, metadata: dict) -> list[Detection]:
        """HigherHRNet 출력 → 17 keypoint per detected person."""
        import numpy as np

        outputs = self.imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            now = time.time()
            if now - self._pose_diag_last > 5.0:
                self._pose_diag_last = now
                log.debug("pose: get_outputs 결과 None")
            return []

        try:
            keypoints, scores, _boxes = self._pose_postprocess(
                outputs=outputs,
                img_size=(self.POSE_IMG_H, self.POSE_IMG_W),
                img_w_pad=(0, 0),
                img_h_pad=(0, 0),
                detection_threshold=self.threshold,
                network_postprocess=True,
            )
        except Exception as e:
            log.debug(f"pose postprocess 실패: {e}")
            return []

        # 5초마다 한 번 현재 상태 진단
        now = time.time()
        if now - self._pose_diag_last > 5.0:
            self._pose_diag_last = now
            sc_count = 0 if scores is None else len(scores)
            log.info(f"pose 진단: detections={sc_count}, threshold={self.threshold}")

        if scores is None or len(scores) == 0:
            return []

        try:
            # keypoints는 img_size 좌표계 (640 W × 480 H) — 정규화 0~1로 변환
            kp_arr = np.reshape(
                np.stack(keypoints, axis=0), (len(scores), 17, 3),
            ).astype(np.float32)
            kp_arr[:, :, 0] /= float(self.POSE_IMG_W)   # x ÷ 640
            kp_arr[:, :, 1] /= float(self.POSE_IMG_H)   # y ÷ 480
        except Exception as e:
            log.debug(f"pose reshape 실패: {e}")
            return []

        detections: list[Detection] = []
        for i, score in enumerate(scores):
            conf = float(score)
            if conf < self.threshold:
                continue
            kps = kp_arr[i]  # (17, 3)
            # bbox: 신뢰도 있는 keypoint들의 minmax (없으면 전체)
            valid = kps[kps[:, 2] >= 0.2]
            if len(valid) >= 3:
                x0 = float(valid[:, 0].min())
                y0 = float(valid[:, 1].min())
                x1 = float(valid[:, 0].max())
                y1 = float(valid[:, 1].max())
            else:
                x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
            detections.append(Detection(
                class_id=0,
                class_name="person",
                confidence=conf,
                bbox=(x0, y0, x1, y1),
                keypoints=kps,
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
