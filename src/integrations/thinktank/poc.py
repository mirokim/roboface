"""ThinkTank 통합 PoC — 헬스체크 + Journal POST 테스트.

실제 ThinkTank 서버가 도메인에 떠 있어야 함.
환경변수 THINKTANK_BASE_URL / THINKTANK_ROBOT_TOKEN 필요.
"""

from __future__ import annotations

from src.integrations.thinktank.client import (
    JournalEntry, ThinkTankClient,
)
from src.utils.logger import get_logger

log = get_logger("thinktank_poc")


async def run_poc(test_message: str | None = None) -> dict[str, object]:
    """헬스체크 → Journal POST → 결과 dict 반환.

    반환값:
        {"healthcheck": bool, "journal_ok": bool, "error": str | None}
    """
    result: dict[str, object] = {
        "healthcheck": False,
        "journal_ok": False,
        "error": None,
    }
    try:
        async with ThinkTankClient() as client:
            ok = await client.healthcheck()
            result["healthcheck"] = ok
            if not ok:
                result["error"] = "healthcheck failed (서버 다운 또는 토큰 X)"
                return result

            entry = JournalEntry(
                content=test_message or "[Roboface PoC] 테스트 메모입니다.",
                mode="auto",
                themes=["roboface_test"],
            )
            data = await client.add_journal(entry)
            result["journal_ok"] = True
            result["response"] = data
            log.info(f"Journal 추가 성공: {data}")
    except Exception as e:
        result["error"] = str(e)
        log.warning(f"PoC 실패: {e}")
    return result
