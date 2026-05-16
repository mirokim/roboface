"""음성 어시스턴트 메인 루프 — wake → listen → STT → Claude → TTS.

흐름:
  1. 마이크 항상 켜고 wake word 감지 대기
  2. wake 감지 시: 표정 LISTENING, VAD로 발화 녹음
  3. STT (OpenAI Whisper) → 텍스트
  4. Claude로 응답 생성 (대화 히스토리 유지)
  5. TTS (OpenAI) + 립싱크로 출력
  6. 다시 wake 대기

graceful: 필요한 백엔드 미설치 시 task가 조용히 종료 (다른 task 영향 X).
"""

from __future__ import annotations

import asyncio

from src.audio.mic import Microphone, MicCaptureError, VADRecorder
from src.audio.stt import OpenAIWhisperSTT, STTError
from src.audio.tts import OpenAITTS, TTSError, speak as tts_speak
from src.audio.wake_word import PorcupineWakeWord, WakeWordError, wait_for_wake
from src.brain import conversation
from src.brain.state_machine import State, StateContext
from src.config import (
    AUDIO_INPUT_DEVICE,
    PORCUPINE_KEYWORD,
    PORCUPINE_KEYWORD_PATH,
)
from src.face.expressions import FOCUSED, NEUTRAL
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("voice_assistant")

# 대화 히스토리 (최근 N턴만 Claude로 전달)
_HISTORY_MAX_TURNS = 6


class VoiceAssistant:
    def __init__(
        self,
        ctx: StateContext,
        face: FaceState,
        *,
        mic_device: int | str | None = None,
        wake_keyword: str = "jarvis",
        wake_keyword_path: str | None = None,
        max_utterance_sec: float = 10.0,
    ) -> None:
        self.ctx = ctx
        self.face = face
        self.mic_device = mic_device
        self.wake_keyword = wake_keyword
        self.wake_keyword_path = wake_keyword_path
        self.max_utterance_sec = max_utterance_sec
        self.history: list[dict] = []
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def _init_components(self) -> tuple[Microphone | None,
                                              PorcupineWakeWord | None,
                                              OpenAIWhisperSTT | None,
                                              OpenAITTS | None]:
        # 마이크
        try:
            mic = Microphone(device=self.mic_device)
        except MicCaptureError as e:
            log.warning(f"마이크 초기화 실패 — voice_assistant 비활성화: {e}")
            return None, None, None, None

        # Wake word
        try:
            wake = PorcupineWakeWord(
                keyword=self.wake_keyword,
                keyword_path=self.wake_keyword_path,
            )
        except WakeWordError as e:
            log.warning(f"Wake word 비활성: {e}")
            wake = None

        # STT
        try:
            stt = OpenAIWhisperSTT()
        except STTError as e:
            log.warning(f"STT 비활성: {e}")
            stt = None

        # TTS
        try:
            tts = OpenAITTS()
        except TTSError as e:
            log.warning(f"TTS 비활성 (fake animation 사용): {e}")
            tts = None

        return mic, wake, stt, tts

    async def run(self) -> None:
        log.info("=== Voice assistant 시작 ===")
        mic, wake, stt, tts = await self._init_components()
        if mic is None:
            log.info("voice_assistant: 마이크 없어 종료")
            return
        if wake is None:
            log.info("voice_assistant: wake word 백엔드 없어 종료")
            return
        if stt is None:
            log.info("voice_assistant: STT 백엔드 없어 종료")
            return

        recorder = VADRecorder(mic, aggressiveness=2, silence_ms=700,
                               start_timeout_sec=5.0)

        try:
            with mic:
                while not self._stop.is_set():
                    log.info(f"💤 wake 대기 중… (\"{wake.keyword}\")")
                    triggered = await wait_for_wake(mic, wake, self._stop)
                    if not triggered:
                        break
                    await self._handle_conversation(recorder, stt, tts)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.exception(f"voice_assistant 루프 에러: {e}")
        finally:
            try:
                if wake is not None:
                    wake.close()
            except Exception:
                pass
            log.info("=== Voice assistant 종료 ===")

    async def _handle_conversation(
        self,
        recorder: VADRecorder,
        stt: OpenAIWhisperSTT,
        tts: OpenAITTS | None,
    ) -> None:
        # 발화 듣기
        prev_state = self.ctx.state
        self.ctx.transition(State.LISTENING, self.face)
        self.face.apply_expression(FOCUSED)
        log.info("🎤 발화 녹음 중…")
        wav = await recorder.record_utterance(max_sec=self.max_utterance_sec)
        if not wav:
            log.info("발화 감지 실패 — 다시 wake 대기")
            self.ctx.transition(prev_state, self.face)
            return

        # STT
        text = await stt.transcribe(wav)
        if not text:
            log.info("STT 결과 비어있음")
            self.ctx.transition(prev_state, self.face)
            return

        # 대화 기록
        self.history.append({"role": "user", "content": text})
        self.history = self.history[-_HISTORY_MAX_TURNS:]

        # Claude 응답
        loop = asyncio.get_running_loop()
        reply = await loop.run_in_executor(
            None,
            lambda: conversation.respond_to_user(text, self.history),
        )
        reply = (reply or "").strip()
        if not reply:
            log.info("Claude 빈 응답")
            self.ctx.transition(prev_state, self.face)
            return
        log.info(f'🗣️  Claude: "{reply}"')
        self.history.append({"role": "assistant", "content": reply})

        # TTS 발화
        self.ctx.transition(State.TALKING, self.face)
        self.face.apply_expression(NEUTRAL)
        await tts_speak(self.face, reply, tts=tts)

        # 잠시 후 복귀
        await asyncio.sleep(0.3)
        if self.ctx.user_present:
            self.ctx.transition(State.WATCHING, self.face)
        else:
            self.ctx.transition(State.IDLE, self.face)


async def run_voice_assistant(ctx: StateContext, face: FaceState) -> None:
    """main_robot에서 task로 시작하는 진입점."""
    va = VoiceAssistant(
        ctx,
        face,
        mic_device=AUDIO_INPUT_DEVICE,
        wake_keyword=PORCUPINE_KEYWORD,
        wake_keyword_path=PORCUPINE_KEYWORD_PATH,
    )
    await va.run()
