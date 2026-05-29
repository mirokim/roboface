"""웨이크 워드 감지 — Picovoice Porcupine.

기본 키워드: "jarvis" (Porcupine 무료 빌트인).
한국어 커스텀 키워드는 PORCUPINE_KEYWORD_PATH 환경변수에 .ppn 경로 지정.

graceful fallback: pvporcupine 미설치 / access key 없음 시
WakeWordError 발생 — voice_assistant가 폴백 처리 (push-to-talk 모드 등).
"""

from __future__ import annotations

import asyncio
import os
import struct
from typing import Any

from src.utils.logger import get_logger

log = get_logger("wake_word")


class WakeWordError(RuntimeError):
    pass


class PorcupineWakeWord:
    """Porcupine 엔진 wrapper.

    Porcupine은 16kHz/mono/int16 PCM 프레임(frame_length=512 샘플)을 요구.
    우리 mic.py는 30ms = 480 샘플로 캡처하므로,
    내부 버퍼링으로 frame_length 만큼 채워서 처리한다.
    """

    def __init__(
        self,
        access_key: str | None = None,
        keyword: str = "jarvis",
        keyword_path: str | None = None,
        sensitivity: float = 0.6,
    ) -> None:
        access_key = access_key or os.getenv("PORCUPINE_ACCESS_KEY", "")
        if not access_key:
            raise WakeWordError(
                "PORCUPINE_ACCESS_KEY 환경변수 필요 "
                "(https://console.picovoice.ai 에서 무료 발급)"
            )

        try:
            import pvporcupine  # type: ignore[import-not-found]
        except ImportError as e:
            raise WakeWordError(
                f"pvporcupine 미설치: {e}. pip install pvporcupine"
            ) from e

        try:
            if keyword_path:
                self._engine = pvporcupine.create(
                    access_key=access_key,
                    keyword_paths=[keyword_path],
                    sensitivities=[sensitivity],
                )
                self.keyword = os.path.basename(keyword_path)
            else:
                self._engine = pvporcupine.create(
                    access_key=access_key,
                    keywords=[keyword],
                    sensitivities=[sensitivity],
                )
                self.keyword = keyword
        except Exception as e:
            raise WakeWordError(f"Porcupine 초기화 실패: {e}") from e

        self.frame_length: int = self._engine.frame_length  # 512
        self.sample_rate: int = self._engine.sample_rate    # 16000
        self._buffer = bytearray()
        log.info(
            f"Porcupine 준비: keyword={self.keyword}, "
            f"frame_length={self.frame_length}, sr={self.sample_rate}"
        )

    def process_pcm(self, pcm_bytes: bytes) -> bool:
        """30ms 프레임 누적 → 512 샘플 청크마다 추론. 감지되면 True."""
        self._buffer.extend(pcm_bytes)
        chunk_bytes = self.frame_length * 2  # int16
        detected = False
        while len(self._buffer) >= chunk_bytes:
            chunk = bytes(self._buffer[:chunk_bytes])
            del self._buffer[:chunk_bytes]
            pcm = struct.unpack_from("<" + "h" * self.frame_length, chunk)
            result = self._engine.process(pcm)
            if result >= 0:
                detected = True
        return detected

    def close(self) -> None:
        try:
            self._engine.delete()
        except Exception:
            pass


class OpenWakeWord:
    """openWakeWord 엔진 wrapper — Porcupine과 동일 인터페이스 (process_pcm/close).

    MIT 라이센스, 키 불필요, 로컬 실행. Pre-trained 모델:
    "alexa", "hey_jarvis", "hey_mycroft", "hey_rhasspy".
    16kHz mono int16 입력. 80ms 청크 단위 추론 (1280 샘플).
    """

    # 모델명 → openwakeword 0.4.0 번들 onnx 파일 prefix 매핑
    _MODEL_ALIASES = {
        "jarvis": "hey_jarvis",
        "hey jarvis": "hey_jarvis",
        "hey_jarvis": "hey_jarvis",
        "alexa": "alexa",
        "mycroft": "hey_mycroft",
        "hey mycroft": "hey_mycroft",
        "hey_mycroft": "hey_mycroft",
        "marvin": "hey_marvin",
        "hey marvin": "hey_marvin",
        "hey_marvin": "hey_marvin",
    }

    CHUNK_SAMPLES = 1280   # 80ms @ 16kHz — openwakeword 기본
    SAMPLE_RATE = 16000

    def __init__(
        self,
        keyword: str = "hey_jarvis",
        threshold: float = 0.5,
    ) -> None:
        try:
            from openwakeword.model import Model  # type: ignore[import-not-found]
        except ImportError as e:
            raise WakeWordError(
                f"openwakeword 미설치: {e}. pip install openwakeword"
            ) from e

        model_prefix = self._MODEL_ALIASES.get(keyword.lower(), keyword)
        # 번들된 onnx 파일 자동 찾기 — 0.4.0은 wakeword_model_paths 인자가
        # 파일 경로 리스트 (이름이 아님). 미지정 시 모든 사전학습 모델 로드.
        model_path = self._find_bundled_model(model_prefix)
        try:
            # 0.4.0은 inference_framework kwarg X — onnx 자동 사용 (onnxruntime
            # 설치돼 있으면). model_paths 미지정 시 모든 사전학습 모델 로드.
            if model_path is not None:
                self._model = Model(wakeword_model_paths=[model_path])
            else:
                self._model = Model()
        except Exception as e:
            raise WakeWordError(
                f"openWakeWord 초기화 실패 ({model_prefix}): {e}"
            ) from e

        # 실제 로드된 모델 키 (predict() 결과 dict의 키와 일치)
        self._target_keys = list(self._model.models.keys())
        self.keyword = model_prefix
        self.threshold = threshold
        self.frame_length = self.CHUNK_SAMPLES
        self.sample_rate = self.SAMPLE_RATE
        self._buffer = bytearray()
        log.info(
            f"openWakeWord 준비: model={model_prefix} keys={self._target_keys}, "
            f"threshold={threshold}, sr={self.sample_rate}"
        )

    @staticmethod
    def _find_bundled_model(prefix: str) -> str | None:
        """openwakeword/resources/models/ 안의 {prefix}_v*.onnx 찾기."""
        import openwakeword
        from pathlib import Path
        base = Path(openwakeword.__file__).parent / "resources" / "models"
        if not base.is_dir():
            return None
        matches = sorted(base.glob(f"{prefix}_v*.onnx"))
        return str(matches[0]) if matches else None

    def process_pcm(self, pcm_bytes: bytes) -> bool:
        """Porcupine 인터페이스와 동일. 30ms 프레임 누적 → 1280 샘플마다 추론."""
        import numpy as np

        self._buffer.extend(pcm_bytes)
        chunk_bytes = self.frame_length * 2  # int16
        detected = False
        while len(self._buffer) >= chunk_bytes:
            chunk = bytes(self._buffer[:chunk_bytes])
            del self._buffer[:chunk_bytes]
            audio = np.frombuffer(chunk, dtype=np.int16)
            scores = self._model.predict(audio)
            # scores: {model_key: float}. _target_keys만 보면 (다른 모델 무시)
            if self._target_keys:
                hit = any(
                    scores.get(k, 0.0) >= self.threshold
                    for k in self._target_keys
                )
            else:
                hit = any(s >= self.threshold for s in scores.values())
            if hit:
                detected = True
                # 한 번 감지되면 잔여 버퍼/내부 state 리셋 (연속 fire 방지)
                try:
                    self._model.reset()
                except Exception:
                    pass
                self._buffer.clear()
                break
        return detected

    def close(self) -> None:
        # openWakeWord는 별도 cleanup 불필요
        pass


def create_wake_word(
    keyword: str = "jarvis",
    keyword_path: str | None = None,
    sensitivity: float = 0.6,
) -> "PorcupineWakeWord | OpenWakeWord":
    """Wake word 엔진 자동 선택.

    1. PORCUPINE_ACCESS_KEY 있으면 Porcupine (한국어 .ppn 지원 + 정확)
    2. 없으면 openWakeWord (오픈소스, 영어 사전학습)
    3. 둘 다 실패하면 WakeWordError
    """
    if os.getenv("PORCUPINE_ACCESS_KEY", ""):
        try:
            return PorcupineWakeWord(
                keyword=keyword, keyword_path=keyword_path,
                sensitivity=sensitivity,
            )
        except WakeWordError as e:
            log.warning(f"Porcupine 실패 — openWakeWord로 fallback: {e}")
    return OpenWakeWord(keyword=keyword, threshold=sensitivity)


async def wait_for_wake(
    mic: Any,
    wake: "PorcupineWakeWord | OpenWakeWord",
    stop_event=None,
) -> bool:
    """마이크에서 프레임을 받아 wake word 감지될 때까지 대기.

    stop_event(asyncio.Event)가 set되면 중단 후 False 반환.
    """
    while stop_event is None or not stop_event.is_set():
        frame = await mic.frame(timeout=1.0)
        if frame is None:
            await asyncio.sleep(0.01)
            continue
        if wake.process_pcm(frame):
            log.info(f"🔔 wake word 감지: {wake.keyword}")
            return True
    return False
