"""가짜 TTS — 텍스트 길이 비례로 입 모양 사이클링.

OPENAI_API_KEY가 설정돼 있으면 실제 OpenAI TTS로 자동 전환.
없으면 더미 입 모양만 표시 (헤드리스/시뮬레이터 호환).
"""

from __future__ import annotations

import asyncio
import random

from src.face.expressions import MouthShape
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("fake_tts")

# 한 번 만들고 재사용 (실패하면 None 유지 → fake)
_tts_singleton = None
_tts_tried = False


def _get_tts():
    global _tts_singleton, _tts_tried
    if _tts_tried:
        return _tts_singleton
    _tts_tried = True
    # 명시적 비활성 (스피커 없는 셋업) — OpenAI 인증 시도조차 안 함
    from src.config import TTS_DISABLED
    if TTS_DISABLED:
        log.info("TTS_DISABLED=1 — OpenAI TTS skip, fake animation 사용")
        _tts_singleton = None
        return None
    try:
        from src.audio.tts import OpenAITTS
        _tts_singleton = OpenAITTS()
        log.info("OpenAI TTS 활성")
    except Exception as e:
        log.debug(f"OpenAI TTS 미사용 (fake animation): {e}")
        _tts_singleton = None
    return _tts_singleton


async def speak(face: FaceState, text: str, duration_per_char: float = 0.06) -> None:
    """발화 — OpenAI TTS 사용 가능하면 실제 음성, 아니면 입모양만.

    `proactive_speaker.fire_trigger` 등 기존 호출처는 코드 변경 없이 자동 업그레이드.
    """
    tts = _get_tts()
    if tts is not None:
        from src.audio.tts import speak as tts_speak
        await tts_speak(face, text, tts=tts)
        return
    # 폴백 — 발화 시작 전 텍스트 정리 (이모지/괄호 무대지문 제거).
    # face.show_speech도 같은 strip을 하지만 로그/duration도 정리된 기준으로.
    from src.face.renderer import strip_emoji
    clean = strip_emoji(text) or text
    duration = max(0.5, len(clean) * duration_per_char)
    log.info(f'speaking ({duration:.1f}s): "{clean}"')

    face.show_speech(clean, duration)
    end = asyncio.get_event_loop().time() + duration
    saved_shape = face.mouth_state.shape
    while asyncio.get_event_loop().time() < end:
        # 무작위 입 모양 (약 100ms씩)
        face.mouth_state.shape = random.choice([
            MouthShape.OPEN_SMALL,
            MouthShape.OPEN_MID,
            MouthShape.OPEN_LARGE,
        ])
        face.mouth_state.talk_amplitude = random.uniform(0.2, 0.9)
        await asyncio.sleep(random.uniform(0.08, 0.14))

    # 발화 끝 — 원래 모양 복귀
    face.mouth_state.shape = saved_shape
    face.mouth_state.talk_amplitude = 0.0
