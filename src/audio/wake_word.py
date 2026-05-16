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


async def wait_for_wake(mic: Any, wake: PorcupineWakeWord, stop_event=None) -> bool:
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
