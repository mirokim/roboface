"""STT — OpenAI Whisper API + 로컬 faster-whisper.

WAV 바이트 → 한국어 텍스트. 두 백엔드 동일 인터페이스 (transcribe).
"""

from __future__ import annotations

import asyncio
import io
import os

from src.utils.logger import get_logger

log = get_logger("stt")


class STTError(RuntimeError):
    pass


class LocalFasterWhisperSTT:
    """faster-whisper 로컬 — API 비용 0, 100% on-device.

    Pi5에서 'base' 모델(74MB) int8 양자화 시 거의 실시간. 첫 사용 시 모델
    자동 다운로드 (~150MB CTranslate2 양자화 weights).
    """

    def __init__(
        self,
        model_size: str = "base",
        language: str = "ko",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError as e:
            raise STTError(
                f"faster-whisper 미설치: {e}. pip install faster-whisper"
            ) from e
        try:
            self._model = WhisperModel(
                model_size, device=device, compute_type=compute_type,
            )
        except Exception as e:
            raise STTError(f"faster-whisper 모델 로드 실패: {e}") from e
        self.language = language
        self.beam_size = beam_size
        self.model_size = model_size
        log.info(
            f"LocalFasterWhisperSTT 준비: model={model_size} "
            f"device={device} compute={compute_type} lang={language}"
        )

    async def transcribe(self, wav_bytes: bytes) -> str:
        if not wav_bytes:
            return ""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, wav_bytes)

    # 한국어 Whisper가 무음/노이즈에 자주 끼워넣는 hallucination 패턴.
    # 유튜브 학습 데이터 잔재 — 정확한/부분 일치 시 빈 텍스트로 폐기.
    _HALLUCINATION_PATTERNS = (
        "구독", "좋아요", "시청해", "시청 해",
        "감사합니다", "고맙습니다", "고마워요", "고마워", "감사해",
        "수고하셨습니다", "수고하셨어요", "수고하세요",
        "안녕히 계세요", "안녕히 가세요",
        "MBC 뉴스", "KBS 뉴스", "한국어 자막",
        "이 영상", "다음 영상", "다음 시간",
    )

    def _is_hallucination(self, text: str) -> bool:
        t = text.strip()
        if not t:
            return True
        # 짧은 텍스트 + 알려진 패턴 포함 → hallucination
        for pat in self._HALLUCINATION_PATTERNS:
            if pat in t:
                return True
        return False

    def _transcribe_sync(self, wav_bytes: bytes) -> str:
        buf = io.BytesIO(wav_bytes)
        try:
            segments, _info = self._model.transcribe(
                buf,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=False,
                # hallucination 억제:
                # - no_speech_threshold 높임: noise→텍스트 변환 빡세게
                # - condition_on_previous_text False: 이전 텍스트 영향 차단
                # - compression_ratio_threshold 낮춤: 반복/이상 출력 차단
                no_speech_threshold=0.7,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.0,
                initial_prompt=None,
            )
            text = " ".join(s.text for s in segments).strip()
            if self._is_hallucination(text):
                log.info(f'STT(local): hallucination drop — "{text}"')
                return ""
            log.info(f'STT(local): "{text}"')
            return text
        except Exception as e:
            log.warning(f"local STT 실패: {e}")
            return ""


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


def create_stt(backend: str | None = None):
    """STT 백엔드 자동 선택.

    backend:
      - "local"  → LocalFasterWhisperSTT ('base' 기본)
      - "openai" → OpenAIWhisperSTT
      - "auto"   → local 시도 후 실패 시 openai (기본)
      - None     → env STT_BACKEND (없으면 "auto")
    """
    if backend is None:
        backend = os.getenv("STT_BACKEND", "auto").lower()
    model_size = os.getenv("STT_LOCAL_MODEL", "base")

    if backend == "local":
        return LocalFasterWhisperSTT(model_size=model_size)
    if backend == "openai":
        return OpenAIWhisperSTT()
    # auto: local 우선, 실패 시 openai fallback
    try:
        return LocalFasterWhisperSTT(model_size=model_size)
    except STTError as e:
        log.warning(f"local STT 실패 — OpenAI fallback: {e}")
        return OpenAIWhisperSTT()
