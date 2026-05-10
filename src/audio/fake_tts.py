"""가짜 TTS — 텍스트 길이 비례로 입 모양 사이클링.

실제 TTS 통합 전 데모용. 부품 도착 후 Piper/ElevenLabs로 교체.
"""

from __future__ import annotations

import asyncio
import random

from src.face.expressions import MouthShape
from src.face.renderer import FaceState
from src.utils.logger import get_logger

log = get_logger("fake_tts")


async def speak(face: FaceState, text: str, duration_per_char: float = 0.06) -> None:
    """가짜 발화 — 입을 무작위로 OPEN_SMALL/MID/LARGE 사이클.

    실제 RMS 기반 립싱크와 비슷한 효과.
    """
    duration = max(0.5, len(text) * duration_per_char)
    log.info(f'speaking ({duration:.1f}s): "{text}"')

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
