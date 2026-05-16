"""얼굴 기억 — 등록된 사람 face embedding과 코사인 유사도로 매칭.

가벼운 첫 버전: 32×32 grayscale flatten + L2 normalize.
정확도는 빛/각도에 약하지만 의존성 zero (numpy + opencv만). 추후 정확한
face_recognition / dlib 기반으로 교체 가능.

DB: SQLite (faces 테이블)
"""

from __future__ import annotations

import io
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger

log = get_logger("face_memory")

EMBED_SIZE = 32  # 32×32 = 1024-d
MATCH_THRESHOLD = 0.94  # cosine similarity
MIN_FACE_PX = 40


@dataclass
class FaceMatch:
    name: str
    confidence: float


def _ensure_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            embedding BLOB NOT NULL,
            created_at REAL NOT NULL,
            last_seen_at REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _embedding_to_blob(emb: Any) -> bytes:
    import numpy as np
    buf = io.BytesIO()
    np.save(buf, emb.astype("float32"))
    return buf.getvalue()


def _blob_to_embedding(blob: bytes) -> Any:
    import numpy as np
    return np.load(io.BytesIO(blob))


def compute_face_embedding(face_crop: Any) -> Any | None:
    """face_crop: BGR/RGB numpy HxWx3 or HxW grayscale. → 1024-d L2-normalized vector."""
    if face_crop is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    h, w = face_crop.shape[:2]
    if h < MIN_FACE_PX or w < MIN_FACE_PX:
        return None
    if face_crop.ndim == 3:
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = face_crop
    gray = cv2.equalizeHist(gray)  # 조명 변화에 약간 robust
    resized = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    vec = resized.astype("float32").flatten()
    # 평균 0, L2 normalize
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def detect_face_crop(
    frame: Any,
    person_bbox: tuple[float, float, float, float] | None,
) -> Any | None:
    """frame + bbox에서 얼굴 위치 찾고 grayscale crop 반환."""
    if frame is None or person_bbox is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = person_bbox
    upper_y1 = y0 + (y1 - y0) * 0.5
    px0, py0 = int(max(0.0, x0) * w), int(max(0.0, y0) * h)
    px1 = int(min(1.0, x1) * w)
    py1 = int(min(1.0, upper_y1) * h)
    if px1 - px0 < MIN_FACE_PX or py1 - py0 < MIN_FACE_PX:
        return None

    roi = frame[py0:py1, px0:px1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY) if roi.ndim == 3 else roi

    classifier = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if classifier.empty():
        return None
    faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                        minSize=(40, 40))
    if len(faces) == 0:
        return None
    # 가장 큰 face
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    # 약간 padding
    pad = int(fw * 0.1)
    fx0 = max(0, fx - pad)
    fy0 = max(0, fy - pad)
    fx1 = min(gray.shape[1], fx + fw + pad)
    fy1 = min(gray.shape[0], fy + fh + pad)
    return gray[fy0:fy1, fx0:fx1]


class FaceMemory:
    """등록된 사람들의 face embedding DB + 코사인 유사도 검색."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _ensure_db(self.db_path)
        self._cache: list[tuple[str, Any]] = []
        self._load_cache()

    def _load_cache(self) -> None:
        cur = self.conn.execute("SELECT name, embedding FROM faces")
        self._cache = []
        for name, blob in cur:
            try:
                emb = _blob_to_embedding(blob)
                self._cache.append((name, emb))
            except Exception as e:
                log.warning(f"face DB row '{name}' 로드 실패: {e}")
        log.info(f"face_memory: {len(self._cache)}명 로드")

    def register(self, name: str, face_crop: Any) -> bool:
        """face_crop으로 embedding 계산 + DB 저장. 같은 이름이면 update."""
        emb = compute_face_embedding(face_crop)
        if emb is None:
            log.warning("face embedding 계산 실패 — 등록 안 됨")
            return False
        blob = _embedding_to_blob(emb)
        now = time.time()
        self.conn.execute(
            "INSERT INTO faces (name, embedding, created_at, last_seen_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET embedding=excluded.embedding, "
            "last_seen_at=excluded.last_seen_at",
            (name, blob, now, now),
        )
        self.conn.commit()
        self._load_cache()
        log.info(f"face_memory: '{name}' 등록")
        return True

    def recognize(self, face_crop: Any) -> FaceMatch | None:
        """face_crop과 가장 가까운 사람 — threshold 못 넘으면 None."""
        if not self._cache:
            return None
        emb = compute_face_embedding(face_crop)
        if emb is None:
            return None
        try:
            import numpy as np
        except ImportError:
            return None
        best_name = ""
        best_sim = -1.0
        for name, ref in self._cache:
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_sim < MATCH_THRESHOLD:
            return None
        # last_seen 업데이트 (가벼움)
        try:
            self.conn.execute(
                "UPDATE faces SET last_seen_at=? WHERE name=?",
                (time.time(), best_name),
            )
            self.conn.commit()
        except Exception:
            pass
        return FaceMatch(name=best_name, confidence=best_sim)

    def list_names(self) -> list[str]:
        return [n for n, _ in self._cache]

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
