"""STT — OpenAI Whisper API.

WAV 바이트 → 한국어 텍스트.
"""

from __future__ import annotations

import asyncio
import io
import os

from src.utils.logger import get_logger

log = get_logger("stt")


class STTError(RuntimeError):
    pass


class OpenAIWhisperSTT:
    """OpenAI gpt-4o-mini-transcribe / whisper-1 wrapper."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
        language: str = "ko",
    ) -> None:
        api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise STTError("OPENAI_API_KEY 환경변수 필요")
        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except ImportError as e:
            raise STTError(f"openai 미설치: {e}. pip install openai") from e
        self._client = OpenAI(api_key=api_key)
        self.model = model
        self.language = language

    async def transcribe(self, wav_bytes: bytes) -> str:
        """WAV 바이트 → 텍스트."""
        if not wav_bytes:
            return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, wav_bytes)

    def _transcribe_sync(self, wav_bytes: bytes) -> str:
        buf = io.BytesIO(wav_bytes)
        buf.name = "utterance.wav"  # OpenAI SDK uses .name for content type
        try:
            resp = self._client.audio.transcriptions.create(
                model=self.model,
                file=buf,
                language=self.language,
                response_format="text",
            )
            text = resp if isinstance(resp, str) else getattr(resp, "text", "")
            text = text.strip()
            log.info(f'STT: "{text}"')
            return text
        except Exception as e:
            log.warning(f"STT 실패: {e}")
            return ""
