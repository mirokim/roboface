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
from src.brain.state_machine import State, StateContext, motion_busy_scope
from src.config import (
    AUDIO_INPUT_DEVICE,
    PORCUPINE_KEYWORD,
    PORCUPINE_KEYWORD_PATH,
)
from src.face.expressions import FOCUSED, HAPPY, NEUTRAL
from src.face.renderer import FaceState
from src.motion import poses
from src.motion.servos import ServoController
from src.utils.logger import get_logger

log = get_logger("voice_assistant")


class _NoopCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# 이름 등록 명령 패턴
_NAME_PATTERNS = (
    "내 이름은 ",
    "내 이름 ",
    "이 사람 이름은 ",
    "이 사람은 ",
    "이름 ",
)


def _extract_register_name(text: str) -> str | None:
    """등록 명령에서 이름 추출. 매칭 안 되면 None."""
    t = text.strip()
    for pat in _NAME_PATTERNS:
        idx = t.find(pat)
        if idx < 0:
            continue
        rest = t[idx + len(pat):]
        # 뒤의 조사/꼬리 제거 ("미로야", "미로입니다", "미로", "미로요")
        rest = rest.split(" ", 1)[0]
        for tail in ("이야", "야", "입니다", "이에요", "예요", "요", "임"):
            if rest.endswith(tail):
                rest = rest[: -len(tail)]
                break
        rest = rest.strip("., !?")
        if 1 <= len(rest) <= 20:
            return rest
    return None

# 대화 히스토리 (최근 N턴만 Claude로 전달)
_HISTORY_MAX_TURNS = 6


class VoiceAssistant:
    def __init__(
        self,
        ctx: StateContext,
        face: FaceState,
        *,
        servos: ServoController | None = None,
        mic: Microphone | None = None,
        mic_device: int | str | None = None,
        wake_keyword: str = "jarvis",
        wake_keyword_path: str | None = None,
        max_utterance_sec: float = 10.0,
        owns_mic: bool = True,
    ) -> None:
        self.ctx = ctx
        self.face = face
        self.servos = servos
        self.mic = mic                # 외부에서 공유 마이크 받음 (None이면 자체 생성)
        self.owns_mic = owns_mic if mic is None else False
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
        # 마이크 — 외부에서 받지 않았으면 자체 생성
        if self.mic is not None:
            mic = self.mic
        else:
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

        # 마이크 owner면 with 컨텍스트 진입, 아니면 외부에서 이미 시작된 상태
        ctx_mic = mic if self.owns_mic else _NoopCtx()
        try:
            with ctx_mic:
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

        # DB에서 이전 대화 맥락 복원 — 재시작 후에도 유지
        from src.brain import memory
        try:
            db_recent = memory.recent_conversation(minutes=60.0, limit=20)
            persistent = [
                {"role": "assistant" if r["speaker"] == "robot" else "user",
                 "content": r["text"]}
                for r in db_recent
                if r["kind"] in ("voice", "voice_reply")
            ]
            self.history = persistent[-_HISTORY_MAX_TURNS:]
        except Exception as e:
            log.debug(f"history 복원 실패: {e}")
        # 이번 user 발화 로그 (history에는 이미 포함 안 됨)
        try:
            memory.log_user(text, kind="voice")
        except Exception as e:
            log.debug(f"conv log 실패: {e}")

        # 사전 정의 명령 인터셉트 — Claude 거치지 않고 즉시 실행
        if await self._maybe_handle_command(text):
            self.ctx.transition(prev_state, self.face)
            return

        # Claude 응답 — 인식된 사용자 이름이 있으면 컨텍스트 prefix
        loop = asyncio.get_running_loop()
        user_text = text
        if self.ctx.user_name:
            user_text = f"[지금 대화 상대: {self.ctx.user_name}] {text}"
        reply = await loop.run_in_executor(
            None,
            lambda: conversation.respond_to_user(user_text, self.history),
        )
        reply = (reply or "").strip()
        if not reply:
            log.info("Claude 빈 응답")
            self.ctx.transition(prev_state, self.face)
            return
        log.info(f'🗣️  Claude: "{reply}"')
        self.history.append({"role": "assistant", "content": reply})
        try:
            from src.brain import memory
            memory.log_robot(reply, kind="voice_reply")
            try:
                from src.brain import stats as robot_stats
                robot_stats.on_event("voice_chat")
            except Exception:
                pass
        except Exception as e:
            log.debug(f"conv log 실패: {e}")

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


    async def _maybe_handle_command(self, text: str) -> bool:
        """사전 정의 음성 명령 인터셉트. 처리되면 True."""
        t = text.lower().strip().rstrip("!?.,")
        # 댄스 명령
        dance_keywords = ("춤춰", "춤 춰", "춤추자", "춤줘", "댄스", "dance")
        if any(k in t for k in dance_keywords):
            log.info("🎵 댄스 명령 인식")
            self.ctx.transition(State.TALKING, self.face)
            self.face.apply_expression(HAPPY)
            if self.servos is not None:
                async with motion_busy_scope(self.ctx):
                    await poses.dance(self.servos, self.face, bpm=120, beats=8)
            else:
                # 서보 없으면 표정/입만 사이클
                from src.motion.poses import dance  # type: ignore[unused-import]
                log.info("서보 없음 — 댄스 모션 스킵, 표정만")
                await asyncio.sleep(2.0)
            return True

        # 이름 등록: "내 이름은 OO이야" / "내 이름 OO" / "이 사람 OO이야"
        name = _extract_register_name(text)
        if name is not None:
            log.info(f"📝 face 등록 요청: '{name}' (다음 프레임에서 캡처)")
            self.ctx.pending_register_name = name
            return True

        return False


async def run_voice_assistant(
    ctx: StateContext,
    face: FaceState,
    servos: ServoController | None = None,
    mic: Microphone | None = None,
) -> None:
    """main_robot에서 task로 시작하는 진입점.

    mic이 외부에서 주어지면 (audio_monitor와 공유) owner가 아님 — 시작/종료는 외부 책임.
    """
    va = VoiceAssistant(
        ctx,
        face,
        servos=servos,
        mic=mic,
        mic_device=AUDIO_INPUT_DEVICE,
        wake_keyword=PORCUPINE_KEYWORD,
        wake_keyword_path=PORCUPINE_KEYWORD_PATH,
    )
    await va.run()
