"""디버그 스냅샷 — 현재 카메라 프레임에 bbox + keypoints 그려서 저장.

robot_cli snapshot 명령으로 트리거. vision_task가 다음 프레임에서 캡처해서
DEBUG_SNAPSHOT_PATH(/tmp/roboface_debug.jpg)에 저장. 항상 같은 경로라
scp로 한 번 가져오면 매번 새 사진.

표시 정보:
- person bbox (녹색 사각형) + score
- 17개 keypoint (점) + conf 0.10 이상은 노란색, 이하 회색
- 어깨/손목/코 라벨
- 모서리에 진단 텍스트 (face orientation, lock 상태, wave 누적 등)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger("debug_snapshot")

DEBUG_SNAPSHOT_PATH = Path("/tmp/roboface_debug.jpg")

# 외부에서 set하면 vision_task가 다음 프레임에서 클리어하면서 캡처.
# 단순 bool 플래그 (asyncio 환경, GIL 보호).
_request_capture: dict = {"flag": False, "info": ""}


def request_snapshot(info: str = "") -> None:
    """다음 프레임에서 디버그 스냅샷 저장 요청."""
    _request_capture["flag"] = True
    _request_capture["info"] = info


def pending() -> bool:
    return _request_capture["flag"]


def clear_pending() -> str:
    info = _request_capture["info"]
    _request_capture["flag"] = False
    _request_capture["info"] = ""
    return info


# COCO 17 keypoint 이름 (어떤 점인지 보이게)
_KP_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]
# 표시할 핵심 keypoint (전부 그리면 어지러움)
_LABEL_KEYS = {0, 5, 6, 9, 10}


def save_debug_snapshot(
    frame: Any,
    detections: list[Any],
    diag_lines: list[str],
    path: Path | None = None,
) -> bool:
    """frame(RGB numpy) + Detection 리스트 + 진단 텍스트 → 어노테이트된 JPG.

    detections: src.vision.camera.Detection 리스트. keypoints 정규화 0~1.
    diag_lines: 좌상단에 출력할 텍스트 라인들.
    """
    if path is None:
        path = DEBUG_SNAPSHOT_PATH
    if frame is None:
        log.warning("snapshot: frame None")
        return False
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        log.warning(f"cv2/numpy 미설치: {e}")
        return False

    try:
        img = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR).copy()
    except Exception as e:
        log.warning(f"이미지 변환 실패: {e}")
        return False

    h, w = img.shape[:2]

    # 각 detection 그리기
    for d in detections:
        if d.class_name != "person":
            continue
        b = d.bbox
        x0, y0, x1, y1 = (
            int(b[0] * w), int(b[1] * h), int(b[2] * w), int(b[3] * h),
        )
        # bbox
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(
            img, f"score={d.confidence:.2f}",
            (x0, max(15, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )
        # keypoints
        if d.keypoints is None:
            continue
        for i, kp in enumerate(d.keypoints):
            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
            px, py = int(x * w), int(y * h)
            if 0 <= px < w and 0 <= py < h:
                color = (0, 255, 255) if conf >= 0.10 else (128, 128, 128)
                cv2.circle(img, (px, py), 5, color, -1)
                if i in _LABEL_KEYS:
                    cv2.putText(
                        img, f"{_KP_NAMES[i]}:{conf:.2f}",
                        (px + 6, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
                    )

    # 진단 텍스트 좌상단
    y = 20
    for line in diag_lines:
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )
        y += 18

    # 시각
    ts = time.strftime("%H:%M:%S")
    cv2.putText(
        img, ts, (w - 90, 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
    )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(
            str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 80],
        )
    except Exception as e:
        log.warning(f"snapshot 저장 실패: {e}")
        return False
    if ok:
        log.info(f"📸 debug snapshot: {path}")
    return ok
