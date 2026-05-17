"""포토 메모리 — 주기 캡처 + 메타 저장 + 자동 정리.

vision_task에서 호출. 사진 자체는 DATA_DIR/face_snapshots/<날짜>/<시각>.jpg.
DB는 메타데이터(시각/감정/거리/이름)만.

privacy: 기본 7일 후 자동 삭제.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.brain import memory
from src.config import DATA_DIR
from src.utils.logger import get_logger

log = get_logger("photo_memory")

SNAPSHOT_DIR = DATA_DIR / "face_snapshots"
KEEP_DAYS = 7.0


def _ensure_dir(date_str: str) -> Path:
    d = SNAPSHOT_DIR / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_snapshot(
    frame: Any,
    bbox: tuple[float, float, float, float] | None,
    emotion: str | None = None,
    distance_cm: float | None = None,
    user_name: str | None = None,
) -> bool:
    """frame (numpy HxWx3 RGB) 받아 JPG로 저장 + DB log. 성공 시 True."""
    if frame is None:
        return False
    try:
        import cv2
        import numpy as np
    except ImportError as e:
        log.warning(f"cv2/numpy 미설치 — 스냅샷 skip: {e}")
        return False
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    fname = now.strftime("%H-%M-%S") + ".jpg"
    d = _ensure_dir(date_str)
    path = d / fname

    # bbox 영역만 crop (있으면) — 사진 용량 줄이고 face만 저장
    h, w = frame.shape[:2]
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        # 정규화 0~1 가정. 살짝 여유 둠.
        margin = 0.05
        x0 = max(0.0, x0 - margin)
        y0 = max(0.0, y0 - margin)
        x1 = min(1.0, x1 + margin)
        y1 = min(1.0, y1 + margin)
        crop = frame[
            int(y0 * h):int(y1 * h),
            int(x0 * w):int(x1 * w),
        ]
        if crop.size == 0:
            crop = frame
    else:
        crop = frame

    # RGB → BGR (cv2가 BGR로 저장)
    if crop.ndim == 3 and crop.shape[2] >= 3:
        crop_bgr = cv2.cvtColor(crop[..., :3], cv2.COLOR_RGB2BGR)
    else:
        crop_bgr = crop
    try:
        ok = cv2.imwrite(str(path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    except Exception as e:
        log.warning(f"imwrite 실패: {e}")
        return False
    if not ok:
        return False
    memory.save_snapshot(
        photo_path=str(path),
        emotion=emotion,
        distance_cm=distance_cm,
        user_name=user_name,
    )
    log.info(f"📸 snapshot saved: {path.name} (emotion={emotion or 'unknown'})")
    return True


def purge_old(keep_days: float = KEEP_DAYS) -> int:
    """오래된 사진 파일 + DB 행 삭제. 삭제된 파일 수 반환."""
    paths = memory.purge_old_snapshots(keep_days=keep_days)
    n = 0
    for p in paths:
        try:
            os.remove(p)
            n += 1
        except FileNotFoundError:
            pass
        except Exception as e:
            log.debug(f"파일 삭제 실패 {p}: {e}")
    # 빈 날짜 폴더도 정리
    if SNAPSHOT_DIR.exists():
        for d in SNAPSHOT_DIR.iterdir():
            if d.is_dir() and not any(d.iterdir()):
                try:
                    d.rmdir()
                except Exception:
                    pass
    if n > 0:
        log.info(f"📸 {n}개 오래된 스냅샷 삭제")
    return n
