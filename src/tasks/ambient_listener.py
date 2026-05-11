"""주변 청취 — STT로 사용자 발화 텍스트화 + 후속 처리.

부품 도착 전: mock transcripts (가끔 사전 정의 문장 emit)
부품 도착 후: audio/stt.py가 Whisper로 실제 텍스트 제공.

후속 처리:
- 일정/약속 언급 → schedule_extractor로 전달
- 의미 있는 발화 → journal_writer로 전달
- 직접 명령("조용히", "오늘 일정") → 즉시 응답
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any

from src.utils.logger import get_logger

log = get_logger("ambient")


# Mock transcript pool — 부품 도착 전 시뮬레이션용
_MOCK_TRANSCRIPTS = [
    "내일 오후 3시에 김 부장님과 회의가 있어",
    "다음주 월요일까지 보고서 제출해야 해",
    "오늘은 좀 피곤하네",
    "점심은 뭐 먹지",
    "이번주 금요일에 친구랑 저녁 약속 잡았어",
    "프로젝트 마감이 다음달 15일이야",
    "이거 정말 재미있는데",
    "내일 9시에 치과 예약 있어",
    "주말에 영화 보러 갈 거야",
    "방금 그 회의 어땠어?",
]


class MockSTT:
    """가짜 STT — 가끔 무작위 transcript 생성."""

    def __init__(self, mean_interval_sec: float = 90.0) -> None:
        self.mean_interval = mean_interval_sec

    async def stream(self) -> AsyncIterator[str]:
        while True:
            wait = random.expovariate(1.0 / self.mean_interval)
            await asyncio.sleep(wait)
            text = random.choice(_MOCK_TRANSCRIPTS)
            log.info(f"[mock STT] \"{text}\"")
            yield text


# 콜백 시그니처: 발화 텍스트 받아서 처리
TranscriptHandler = Callable[[str], Coroutine[Any, Any, None]]


class AmbientListener:
    """STT 결과를 받아 등록된 핸들러들에게 전달."""

    def __init__(self, stt: MockSTT | None = None) -> None:
        self.stt = stt or MockSTT()
        self.handlers: list[TranscriptHandler] = []

    def add_handler(self, handler: TranscriptHandler) -> None:
        self.handlers.append(handler)

    async def run(self) -> None:
        async for text in self.stt.stream():
            for h in self.handlers:
                try:
                    await h(text)
                except Exception as e:
                    log.warning(f"handler 에러: {e}")
