"""ThinkTank1 HTTP 클라이언트.

인증: X-Robot-Token 헤더 (ThinkTank 측 requireAuth에 추가 필요).
모든 메소드 비동기 (httpx.AsyncClient).
오프라인이면 로컬 큐에 쌓아두고 복구 시 재전송 — 추후 구현.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from src.config import (
    THINKTANK_BASE_URL, THINKTANK_RETRY,
    THINKTANK_ROBOT_TOKEN, THINKTANK_TIMEOUT_SEC,
)
from src.utils.logger import get_logger

log = get_logger("thinktank")


class ThinkTankError(Exception):
    pass


@dataclass
class JournalEntry:
    content: str
    mode: str = "auto"          # "auto" / "manual" / "voice"
    mood_tags: list[str] | None = None
    themes: list[str] | None = None


@dataclass
class CalendarEvent:
    title: str
    start: str            # ISO datetime
    end: str | None = None
    description: str | None = None
    source: str = "robot"


class ThinkTankClient:
    """ThinkTank1 API 비동기 클라이언트."""

    def __init__(
        self,
        base_url: str = THINKTANK_BASE_URL,
        token: str = THINKTANK_ROBOT_TOKEN,
        timeout: float = THINKTANK_TIMEOUT_SEC,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ThinkTankClient":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers={
                "X-Robot-Token": self.token,
                "Content-Type": "application/json",
            },
        )
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def _request(
        self, method: str, path: str, **kwargs,
    ) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("Use 'async with ThinkTankClient() as c:'")
        last_err: Exception | None = None
        for attempt in range(THINKTANK_RETRY):
            try:
                r = await self._client.request(method, path, **kwargs)
                r.raise_for_status()
                if r.headers.get("content-type", "").startswith("application/json"):
                    return r.json()
                return {"text": r.text}
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_err = e
                log.debug(f"{method} {path} attempt {attempt + 1} 실패: {e}")
        raise ThinkTankError(f"{method} {path} failed: {last_err}")

    # === Health ===

    async def healthcheck(self) -> bool:
        try:
            await self._request("GET", "/api/config")
            return True
        except Exception as e:
            log.warning(f"ThinkTank 헬스체크 실패: {e}")
            return False

    # === Journal ===

    async def add_journal(self, entry: JournalEntry) -> dict[str, Any]:
        """자동 저널 항목 추가."""
        payload = {
            "content": entry.content,
            "mode": entry.mode,
        }
        if entry.mood_tags:
            payload["mood_tags"] = entry.mood_tags
        if entry.themes:
            payload["themes"] = entry.themes
        return await self._request("POST", "/api/journal", json=payload)

    # === Calendar ===

    async def add_calendar_event(self, event: CalendarEvent) -> dict[str, Any]:
        payload = {
            "title": event.title,
            "start": event.start,
            "source": event.source,
        }
        if event.end:
            payload["end"] = event.end
        if event.description:
            payload["description"] = event.description
        return await self._request("POST", "/api/calendar/events", json=payload)

    # === Vault (RAG) ===

    async def search_vault(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        try:
            data = await self._request(
                "POST", "/api/vault/search",
                json={"q": query, "limit": limit},
            )
            return data.get("results", [])
        except ThinkTankError:
            return []

    # === Robot 전용 (ThinkTank 측 추가 필요) ===

    async def get_pending_notifications(self) -> list[dict[str, Any]]:
        try:
            data = await self._request("GET", "/api/robot/notifications/pending")
            return data.get("notifications", [])
        except ThinkTankError:
            return []

    async def post_observation(
        self, observation_type: str, data: dict[str, Any],
    ) -> None:
        """자세/작업시간/존재 관찰 데이터 전송."""
        try:
            await self._request(
                "POST", "/api/robot/observation",
                json={"type": observation_type, "data": data},
            )
        except ThinkTankError as e:
            log.debug(f"observation 전송 실패: {e}")

    async def get_context_bundle(self) -> dict[str, Any]:
        """현재 시점 컨텍스트 (다음 일정, 미확인 알림 등) 한 묶음."""
        try:
            return await self._request("GET", "/api/robot/context-bundle")
        except ThinkTankError:
            return {}
