"""얼굴 기억 — 사람별 face embedding DB + 코사인 유사도 매칭 + 썸네일 라이브러리.

백엔드 두 가지 (자동 선택):
- **sface** (기본): OpenCV YuNet(검출/정렬) + SFace(128-d 임베딩). 정확도 높음.
  모델 ONNX 두 개를 DATA_DIR/models/에 두면 활성 (없으면 최초 1회 다운로드 시도).
- **pixel** (fallback/테스트): Haar 검출 + 32×32 grayscale flatten. 의존성 zero.

DB: SQLite (faces 테이블). 백엔드가 바뀌면 임베딩 차원이 달라 호환 안 되므로
DB 파일을 백엔드별로 분리한다 (faces.db=pixel, faces_sface.db=sface).
썸네일: DATA_DIR/faces/<name>.jpg — 웹 UI 라이브러리에서 이름 붙이기용.
"""

from __future__ import annotations

import io
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import DATA_DIR
from src.utils.logger import get_logger

log = get_logger("face_memory")

# ─── pixel 백엔드 ───
EMBED_SIZE = 32  # 32×32 = 1024-d
MATCH_THRESHOLD = 0.88          # pixel 백엔드 코사인 임계 (테스트/호환용 공개 상수)
MIN_FACE_PX = 40

# ─── sface 백엔드 ───
MODELS_DIR = DATA_DIR / "models"
YUNET_FILE = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_FILE = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
# opencv_zoo는 Git LFS — raw.githubusercontent는 포인터 텍스트만 줌. media 호스트 사용.
_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"
_MODEL_URLS = {
    YUNET_FILE: f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    SFACE_FILE: f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
}
SFACE_MATCH_THRESHOLD = 0.363   # OpenCV 권장 코사인 임계
SFACE_CROP = (112, 112)         # alignCrop 출력 크기
YUNET_SCORE = 0.7

# 썸네일 라이브러리
THUMBS_DIR = DATA_DIR / "faces"

# auto_track 카운트 throttle — 같은 사람 N초 안엔 1번만 +1 (매 frame 카운트 방지)
_AUTO_COUNT_THROTTLE_SEC = 10.0
# "주인" 승급 최소 seen_count — 잠깐 지나간 사람은 주인 안 됨
DEFAULT_OWNER_MIN_SEEN = 20
# auto cluster name prefix — explicit register와 구분
AUTO_NAME_PREFIX = "auto_"

_NAME_RE = re.compile(r"^[\w가-힣 .-]{1,32}$")


@dataclass
class FaceMatch:
    name: str
    confidence: float


# ─── 모델 로딩 (lazy, 프로세스당 1회) ───

_sface_state: dict[str, Any] = {"tried": False, "detector": None, "recognizer": None}


def ensure_models(download: bool = True) -> bool:
    """ONNX 모델 두 개 준비. 없으면 다운로드 시도(best-effort). 성공 여부 반환."""
    if YUNET_FILE.exists() and SFACE_FILE.exists():
        return True
    if not download:
        return False
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for path, url in _MODEL_URLS.items():
        if path.exists():
            continue
        try:
            log.info(f"face model 다운로드: {path.name}")
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            if tmp.stat().st_size < 100_000:      # LFS 포인터/에러 페이지 방어
                tmp.unlink(missing_ok=True)
                raise RuntimeError("파일이 너무 작음 (LFS 포인터?)")
            tmp.replace(path)
        except Exception as e:
            log.warning(f"face model 다운로드 실패 ({path.name}): {e}")
            return False
    return YUNET_FILE.exists() and SFACE_FILE.exists()


def _sface() -> tuple[Any, Any] | None:
    """(YuNet detector, SFace recognizer) 또는 None (미가용)."""
    st = _sface_state
    if st["detector"] is not None:
        return st["detector"], st["recognizer"]
    if st["tried"]:
        return None
    st["tried"] = True
    try:
        import cv2
        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            return None
        if not ensure_models(download=False):   # 다운로드는 vision_task 시작 시 명시적으로
            return None
        det = cv2.FaceDetectorYN.create(str(YUNET_FILE), "", (320, 320),
                                        YUNET_SCORE, 0.3, 5000)
        rec = cv2.FaceRecognizerSF.create(str(SFACE_FILE), "")
        st["detector"], st["recognizer"] = det, rec
        log.info("face_memory: sface 백엔드 활성 (YuNet + SFace)")
        return det, rec
    except Exception as e:
        log.warning(f"sface 백엔드 초기화 실패 → pixel fallback: {e}")
        return None


def backend_name() -> str:
    return "sface" if _sface() is not None else "pixel"


def match_threshold() -> float:
    return SFACE_MATCH_THRESHOLD if _sface() is not None else MATCH_THRESHOLD


def default_db_path() -> Path:
    return DATA_DIR / ("faces_sface.db" if _sface() is not None else "faces.db")


# ─── DB ───

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
    # migration: auto-tracking 도입 시 seen_count 컬럼 추가. idempotent.
    try:
        conn.execute("ALTER TABLE faces ADD COLUMN seen_count INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass   # 이미 있음
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


# ─── 임베딩 / 검출 ───

def _is_aligned_crop(face_crop: Any) -> bool:
    return (
        getattr(face_crop, "ndim", 0) == 3
        and tuple(face_crop.shape[:2]) == SFACE_CROP
    )


def compute_face_embedding(face_crop: Any) -> Any | None:
    """face_crop → L2-normalized 벡터.

    - sface: detect_face_crop이 만든 112×112×3 정렬 crop → 128-d
    - pixel: 아무 crop(gray/RGB) → 1024-d
    """
    if face_crop is None:
        return None
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    sf = _sface()
    if sf is not None and _is_aligned_crop(face_crop):
        _, rec = sf
        try:
            feat = rec.feature(face_crop).flatten().astype("float32")
        except Exception as e:
            log.debug(f"sface feature 실패: {e}")
            return None
        norm = float(np.linalg.norm(feat))
        return feat / norm if norm > 1e-6 else None

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
    vec -= float(vec.mean())
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return None
    return vec / norm


def detect_face_crop(
    frame: Any,
    person_bbox: tuple[float, float, float, float] | None,
) -> Any | None:
    """frame(RGB) + 사람 bbox(정규화)에서 얼굴 찾아 crop 반환.

    sface: YuNet 검출 → alignCrop 112×112 BGR (임베딩 입력 규격)
    pixel: Haar 검출 → grayscale crop
    """
    if frame is None or person_bbox is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = person_bbox
    upper_y1 = y0 + (y1 - y0) * 0.65   # 상체 위쪽 — 얼굴은 여기 안에
    px0, py0 = int(max(0.0, x0) * w), int(max(0.0, y0) * h)
    px1 = int(min(1.0, x1) * w)
    py1 = int(min(1.0, upper_y1) * h)
    if px1 - px0 < MIN_FACE_PX or py1 - py0 < MIN_FACE_PX:
        return None
    roi = frame[py0:py1, px0:px1]

    sf = _sface()
    if sf is not None and frame.ndim == 3:
        # 전체 프레임에서 검출 → 사람 bbox 안(여유 포함)의 가장 큰 얼굴 선택.
        # ROI만 잘라 넣으면 alignCrop이 ROI 밖을 검정으로 채워 임베딩이 망가짐.
        det, rec = sf
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        try:
            det.setInputSize((w, h))
            _, faces = det.detect(bgr)
        except Exception as e:
            log.debug(f"yunet detect 실패: {e}")
            return None
        if faces is None or len(faces) == 0:
            return None
        mx0, my0 = max(0.0, x0 - 0.1) * w, max(0.0, y0 - 0.1) * h
        mx1, my1 = min(1.0, x1 + 0.1) * w, min(1.0, upper_y1 + 0.1) * h
        cands = [
            f for f in faces
            if mx0 <= f[0] + f[2] / 2 <= mx1 and my0 <= f[1] + f[3] / 2 <= my1
        ]
        if not cands:
            return None
        best = max(cands, key=lambda f: f[2] * f[3])
        if best[2] < MIN_FACE_PX or best[3] < MIN_FACE_PX:
            return None
        try:
            return rec.alignCrop(bgr, best)
        except Exception as e:
            log.debug(f"sface alignCrop 실패: {e}")
            return None

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
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    pad = int(fw * 0.1)
    fx0 = max(0, fx - pad)
    fy0 = max(0, fy - pad)
    fx1 = min(gray.shape[1], fx + fw + pad)
    fy1 = min(gray.shape[0], fy + fh + pad)
    return gray[fy0:fy1, fx0:fx1]


def _save_thumb(name: str, face_crop: Any) -> None:
    """클러스터 썸네일 저장 (best-effort)."""
    try:
        import cv2
        THUMBS_DIR.mkdir(parents=True, exist_ok=True)
        img = face_crop
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif not _is_aligned_crop(img):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(THUMBS_DIR / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    except Exception as e:
        log.debug(f"thumb 저장 실패 ({name}): {e}")


class FaceMemory:
    """등록된 사람들의 face embedding DB + 코사인 유사도 검색."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = _ensure_db(self.db_path)
        self._cache: list[tuple[str, Any]] = []
        self._load_cache()

    def _load_cache(self) -> None:
        cur = self.conn.execute(
            "SELECT name, embedding, last_seen_at FROM faces"
        )
        self._cache = []
        # auto_track 카운트 throttle용 — 마지막 카운트 시각 (DB last_seen_at은
        # recognize도 갱신해서 throttle 기준으로 직접 못 씀)
        self._last_counted_at: dict[str, float] = {}
        for name, blob, last_seen in cur:
            try:
                emb = _blob_to_embedding(blob)
                self._cache.append((name, emb))
                self._last_counted_at[name] = float(last_seen)
            except Exception as e:
                log.warning(f"face DB row '{name}' 로드 실패: {e}")
        log.info(f"face_memory: {len(self._cache)}명 로드 (backend={backend_name()})")

    def _best(self, emb: Any) -> tuple[str, float]:
        import numpy as np
        best_name, best_sim = "", -1.0
        for name, ref in self._cache:
            if ref.shape != emb.shape:
                continue   # 다른 백엔드 임베딩 — 비교 불가
            sim = float(np.dot(emb, ref))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        return best_name, best_sim

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
        _save_thumb(name, face_crop)
        log.info(f"face_memory: '{name}' 등록")
        return True

    def recognize(self, face_crop: Any) -> FaceMatch | None:
        """face_crop과 가장 가까운 사람 — threshold 못 넘으면 None."""
        if not self._cache:
            return None
        emb = compute_face_embedding(face_crop)
        if emb is None:
            return None
        best_name, best_sim = self._best(emb)
        if best_sim < match_threshold():
            return None
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

    def list_people(self) -> list[dict[str, Any]]:
        """라이브러리 뷰용 — name/seen_count/first/last/thumb 경로."""
        cur = self.conn.execute(
            "SELECT name, seen_count, created_at, last_seen_at FROM faces "
            "ORDER BY seen_count DESC, id ASC"
        )
        out = []
        for name, seen, created, last in cur:
            thumb = THUMBS_DIR / f"{name}.jpg"
            out.append({
                "name": name, "seen_count": int(seen or 0),
                "created_at": float(created), "last_seen_at": float(last),
                "is_auto": name.startswith(AUTO_NAME_PREFIX),
                "thumb": str(thumb) if thumb.exists() else None,
            })
        return out

    def rename(self, old: str, new: str) -> bool:
        """auto 클러스터에 이름 붙이기 (웹 UI). 이름 규칙 검증."""
        new = new.strip()
        if not _NAME_RE.match(new) or new.startswith(AUTO_NAME_PREFIX):
            return False
        try:
            cur = self.conn.execute(
                "UPDATE faces SET name=? WHERE name=?", (new, old),
            )
            self.conn.commit()
            if cur.rowcount == 0:
                return False
        except sqlite3.IntegrityError:
            return False
        old_t, new_t = THUMBS_DIR / f"{old}.jpg", THUMBS_DIR / f"{new}.jpg"
        if old_t.exists():
            try:
                old_t.replace(new_t)
            except Exception:
                pass
        self._load_cache()
        log.info(f"face_memory: '{old}' → '{new}' 이름 변경")
        return True

    def delete(self, name: str) -> bool:
        cur = self.conn.execute("DELETE FROM faces WHERE name=?", (name,))
        self.conn.commit()
        t = THUMBS_DIR / f"{name}.jpg"
        if t.exists():
            try:
                t.unlink()
            except Exception:
                pass
        self._load_cache()
        return cur.rowcount > 0

    def prune(self, max_age_sec: float = 86400.0, min_seen: int = 2) -> int:
        """노이즈 정리 — seen_count < min_seen 인 auto 클러스터가 max_age 지나면 삭제.

        오검출/지나가던 사람 한 번 찍힌 것들이 DB를 채우는 걸 막는다.
        """
        cutoff = time.time() - max_age_sec
        cur = self.conn.execute(
            "SELECT name FROM faces WHERE name LIKE ? AND seen_count < ? "
            "AND last_seen_at < ?",
            (f"{AUTO_NAME_PREFIX}%", min_seen, cutoff),
        )
        names = [r[0] for r in cur]
        for n in names:
            self.delete(n)
        if names:
            log.info(f"face_memory: 노이즈 cluster {len(names)}개 정리")
        return len(names)

    # ─── 자동 학습 (등장 빈도 누적) ───

    def auto_track(self, face_crop: Any) -> tuple[str, int] | None:
        """face_crop을 자동 cluster에 누적 학습.

        흐름:
          1) 기존 cluster와 매칭 시도 (match_threshold)
          2) 매칭되면 seen_count += 1 (단, _AUTO_COUNT_THROTTLE_SEC 안엔 skip)
          3) 매칭 안 되면 새 "auto_N" cluster 생성 (seen_count=1) + 썸네일

        반환: (cluster_name, current_seen_count) 또는 None (embedding 실패).
        """
        emb = compute_face_embedding(face_crop)
        if emb is None:
            return None
        best_name, best_sim = self._best(emb)
        now = time.time()
        if best_sim >= match_threshold():
            last_counted = self._last_counted_at.get(best_name, 0.0)
            if now - last_counted >= _AUTO_COUNT_THROTTLE_SEC:
                try:
                    self.conn.execute(
                        "UPDATE faces SET seen_count = seen_count + 1, "
                        "last_seen_at = ? WHERE name = ?",
                        (now, best_name),
                    )
                    self.conn.commit()
                    self._last_counted_at[best_name] = now
                except Exception as e:
                    log.warning(f"seen_count 업데이트 실패: {e}")
            cur = self.conn.execute(
                "SELECT seen_count FROM faces WHERE name = ?", (best_name,),
            )
            row = cur.fetchone()
            count = int(row[0]) if row else 1
            return (best_name, count)

        cur = self.conn.execute(
            f"SELECT name FROM faces WHERE name LIKE '{AUTO_NAME_PREFIX}%'"
        )
        existing_nums = []
        for (name,) in cur:
            try:
                existing_nums.append(int(name[len(AUTO_NAME_PREFIX):]))
            except ValueError:
                continue
        next_num = (max(existing_nums) + 1) if existing_nums else 1
        new_name = f"{AUTO_NAME_PREFIX}{next_num:03d}"
        blob = _embedding_to_blob(emb)
        try:
            self.conn.execute(
                "INSERT INTO faces (name, embedding, created_at, last_seen_at, "
                "seen_count) VALUES (?, ?, ?, ?, 1)",
                (new_name, blob, now, now),
            )
            self.conn.commit()
            self._load_cache()
            _save_thumb(new_name, face_crop)
            log.info(f"face_memory: 새 자동 cluster '{new_name}' 등록")
        except Exception as e:
            log.warning(f"새 auto cluster 생성 실패: {e}")
            return None
        return (new_name, 1)

    def get_owner(
        self, min_seen: int = DEFAULT_OWNER_MIN_SEEN,
    ) -> tuple[str, int] | None:
        """가장 자주 등장한 cluster — (name, seen_count). min_seen 미만이면 None."""
        cur = self.conn.execute(
            "SELECT name, seen_count FROM faces "
            "WHERE seen_count >= ? ORDER BY seen_count DESC, id ASC LIMIT 1",
            (min_seen,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return (str(row[0]), int(row[1]))

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass
