"""의미 있는 발화 → ThinkTank Journal 자동 저장.

전략:
- 모든 발화를 다 저장하면 노이즈 큼
- 감정 표현/사건 언급/계획 등 "의미 신호"가 있을 때만
- 간단한 키워드 휴리스틱 + 길이 필터 (정교한 분류는 향후 LLM)
"""

from __future__ import annotations

from src.integrations.thinktank.client import JournalEntry, ThinkTankClient
from src.utils.logger import get_logger

log = get_logger("journal_writer")


_MEANINGFUL_KEYWORDS = [
    # 감정
    "피곤", "지치", "행복", "즐거", "기쁘", "슬프", "우울", "재미", "신나", "스트레스",
    # 사건/계획
    "약속", "회의", "마감", "프로젝트", "오늘", "내일", "주말", "어제",
    # 사람
    "친구", "가족", "부장", "팀장",
]


def _is_meaningful(text: str) -> bool:
    if len(text) < 8:
        return False
    return any(kw in text for kw in _MEANINGFUL_KEYWORDS)


async def handle_transcript(text: str) -> None:
    """ambient_listener handler."""
    if not _is_meaningful(text):
        return

    log.info(f"Journal 후보: \"{text}\"")
    try:
        async with ThinkTankClient() as client:
            ok = await client.healthcheck()
            if not ok:
                log.debug("ThinkTank 다운 — Journal 저장 보류")
                return
            entry = JournalEntry(
                content=text,
                mode="auto",
                themes=["robot_ambient"],
            )
            data = await client.add_journal(entry)
            log.info(f"  ✓ Journal 저장 (id={data.get('id', '?')})")
    except Exception as e:
        log.debug(f"Journal 저장 실패: {e}")
