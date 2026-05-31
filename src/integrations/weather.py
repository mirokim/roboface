"""OpenWeather API 클라이언트 — agent 컨텍스트용 한 줄 날씨 요약.

설계:
- async snapshot() → "맑음 18°C (체감 16°C) · 습도 45%" 같은 짧은 한 줄
- TTL 캐시 (BEHAVIOR.weather_cache_sec, 기본 30min) — 무료 tier 안전 운용
- API 키 없으면 snapshot() → None (agent에서 자동 skip)
- HTTP 에러/타임아웃 시 마지막 성공 캐시 그대로 반환 (stale-while-error)
- urllib만 사용 — 의존성 추가 X (requests 빼고 가벼움)

agent.py가 매 tick 호출하지만 캐시 hit는 비용 0이라 부담 없음.
"""

from __future__ import annotations

import asyncio
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

from src.config import (
    BEHAVIOR,
    OPENWEATHER_API_KEY,
    WEATHER_LAT,
    WEATHER_LOCATION_NAME,
    WEATHER_LON,
)
from src.utils.logger import get_logger

log = get_logger("weather")


_OWM_URL = "https://api.openweathermap.org/data/2.5/weather"


@dataclass(frozen=True)
class WeatherSnapshot:
    """현재 날씨 한 컷. agent prompt 한 줄로 압축할 수 있는 최소 정보."""

    description: str        # "맑음", "흐림", "비" 등 (lang=kr)
    temp_c: float           # 현재 기온
    feels_like_c: float     # 체감 기온
    humidity: int           # 습도 %
    wind_mps: float         # 풍속 m/s
    location_name: str      # 표시용 ("분당")
    fetched_at: float       # epoch sec — staleness 판단용

    def one_liner(self) -> str:
        """agent prompt 주입용 짧은 표현.

        예: "분당 맑음 18°C(체감 16°C) · 습도 45% · 바람 2m/s"
        체감-기온 차이 1°C 이내면 (체감 ...) 생략으로 짧게.
        """
        parts = [f"{self.location_name} {self.description}", f"{self.temp_c:.0f}°C"]
        if abs(self.feels_like_c - self.temp_c) >= 1.0:
            parts.append(f"(체감 {self.feels_like_c:.0f}°C)")
        parts.append(f"습도 {self.humidity}%")
        if self.wind_mps >= 3.0:
            parts.append(f"바람 {self.wind_mps:.0f}m/s")
        return " · ".join(parts)


class WeatherClient:
    """싱글톤성 캐시 클라이언트. 모듈 레벨 get_client()로 공유.

    cache_sec 안에는 같은 snapshot 재사용 — 네트워크 0회. agent가 매 tick
    호출해도 비용 없음.
    """

    def __init__(
        self,
        api_key: str = OPENWEATHER_API_KEY,
        lat: float = WEATHER_LAT,
        lon: float = WEATHER_LON,
        location_name: str = WEATHER_LOCATION_NAME,
        cache_sec: float | None = None,
        timeout_sec: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.location_name = location_name
        self.cache_sec = cache_sec if cache_sec is not None else BEHAVIOR.weather_cache_sec
        self.timeout_sec = (
            timeout_sec if timeout_sec is not None else BEHAVIOR.weather_http_timeout_sec
        )
        self._cached: WeatherSnapshot | None = None
        self._lock = asyncio.Lock()

    def _build_url(self) -> str:
        params = {
            "lat": f"{self.lat:.4f}",
            "lon": f"{self.lon:.4f}",
            "appid": self.api_key,
            "units": "metric",
            "lang": "kr",
        }
        return f"{_OWM_URL}?{urllib.parse.urlencode(params)}"

    def _fetch_sync(self) -> WeatherSnapshot:
        """blocking urllib 호출 — async snapshot()이 executor로 감싸 호출.

        파싱 실패/HTTP 에러는 RuntimeError로 raise — 호출부가 캐시 fallback.
        """
        req = urllib.request.Request(
            self._build_url(),
            headers={"User-Agent": "roboface/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        # OpenWeather 응답 형식: weather[0].description, main.{temp, feels_like, humidity}, wind.speed
        try:
            weather = payload["weather"][0]
            main = payload["main"]
            wind = payload.get("wind", {})
            return WeatherSnapshot(
                description=str(weather.get("description") or weather.get("main") or "?"),
                temp_c=float(main["temp"]),
                feels_like_c=float(main.get("feels_like", main["temp"])),
                humidity=int(main.get("humidity", 0)),
                wind_mps=float(wind.get("speed", 0.0)),
                location_name=self.location_name,
                fetched_at=time.time(),
            )
        except (KeyError, TypeError, ValueError) as e:
            # payload 거대하면 로그 폭주 방지 — 200자로 truncate
            payload_str = str(payload)
            if len(payload_str) > 200:
                payload_str = payload_str[:200] + "...(truncated)"
            raise RuntimeError(
                f"OpenWeather 응답 파싱 실패: {e} (payload={payload_str})"
            ) from e

    async def snapshot(self) -> WeatherSnapshot | None:
        """현재 날씨. 키 없으면 None, 캐시 살아 있으면 캐시, 만료면 fetch.

        네트워크 실패 시 stale 캐시 그대로 반환 (있으면) — agent prompt가 빈
        한 줄 노출하는 것보단 stale이 낫다.
        """
        if not self.api_key:
            return None

        now = time.time()
        cached = self._cached
        if cached is not None and (now - cached.fetched_at) < self.cache_sec:
            return cached

        # 동시 호출 폭주 방지 — lock 안에서 다시 확인 (double-check)
        async with self._lock:
            cached = self._cached
            if cached is not None and (time.time() - cached.fetched_at) < self.cache_sec:
                return cached
            loop = asyncio.get_running_loop()
            try:
                fresh = await loop.run_in_executor(None, self._fetch_sync)
                self._cached = fresh
                log.info(f"weather fetched: {fresh.one_liner()}")
                return fresh
            except Exception as e:
                log.warning(f"weather fetch 실패 (stale fallback): {e}")
                return self._cached   # 있으면 stale, 없으면 None


# 모듈 레벨 싱글톤 — agent.py가 import해서 그대로 사용
_client_singleton: WeatherClient | None = None


def get_client() -> WeatherClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = WeatherClient()
    return _client_singleton
