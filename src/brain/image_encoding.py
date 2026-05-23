"""이미지(numpy RGB) → JPEG base64 인코딩 — Claude vision API용.

opencv-python을 우선 사용 (이미 의존성에 있음). 없으면 PIL fallback.
"""

from __future__ import annotations

import base64
from typing import Any

from src.utils.logger import get_logger

log = get_logger("image_encoding")


def encode_jpeg_b64(
    frame: Any,
    quality: int = 70,
    max_side_px: int = 480,
) -> str | None:
    """numpy RGB array → JPEG base64 문자열 (헤더 제외).

    - max_side_px보다 큰 쪽이 있으면 종횡비 유지하며 축소
    - quality는 0~100 (JPEG 품질)
    - 실패 시 None
    """
    if frame is None:
        return None
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return _encode_with_pil(frame, quality, max_side_px)

    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        log.debug(f"unexpected frame shape: {arr.shape}")
        return None

    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest > max_side_px:
        scale = max_side_px / longest
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # cv2.imencode은 BGR 기대 — RGB를 BGR로 변환.
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        log.debug("cv2.imencode 실패")
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _encode_with_pil(
    frame: Any, quality: int, max_side_px: int,
) -> str | None:
    """PIL fallback — opencv 없을 때."""
    try:
        from io import BytesIO
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as e:
        log.warning(f"이미지 인코딩 불가 (cv2/PIL 모두 없음): {e}")
        return None
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return None
    img = Image.fromarray(arr.astype("uint8"), mode="RGB")
    h, w = arr.shape[:2]
    longest = max(h, w)
    if longest > max_side_px:
        scale = max_side_px / longest
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")
