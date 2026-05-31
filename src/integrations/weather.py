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
_OWM_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"


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


@dataclass(frozen=True)
class ForecastSnapshot:
    """특정 날짜의 일일 forecast 요약 — min/max 기온 + 대표 날씨 + 강수확률."""

    date_label: str         # "내일" 같은 표시 라벨 (호출자가 결정)
    description: str        # 대표 날씨 (낮 시간대 가장 많이 나온 description)
    temp_min: float
    temp_max: float
    pop_max: float          # 그 날 최대 강수확률 (0~1)
    humidity: int           # 평균 습도 %
    location_name: str
    fetched_at: float

    def one_liner(self) -> str:
        """예: "내일 분당 흐림 19~29°C · 비올 확률 60%"

        pop < 30%면 강수확률 생략 — 비/눈 가능성 낮을 땐 한 줄 더 깔끔하게.
        """
        parts = [
            f"{self.date_label} {self.location_name}",
            self.description,
            f"{self.temp_min:.0f}~{self.temp_max:.0f}°C",
        ]
        if self.pop_max >= 0.30:
            parts.append(f"비올 확률 {int(self.pop_max * 100)}%")
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
        forecast_cache_sec: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.location_name = location_name
        self.cache_sec = cache_sec if cache_sec is not None else BEHAVIOR.weather_cache_sec
        # forecast는 변화 더 느림 → 1시간 캐시 (BehaviorConfig override 가능)
        self.forecast_cache_sec = (
            forecast_cache_sec if forecast_cache_sec is not None
            else getattr(BEHAVIOR, "weather_forecast_cache_sec", 3600.0)
        )
        self.timeout_sec = (
            timeout_sec if timeout_sec is not None else BEHAVIOR.weather_http_timeout_sec
        )
        self._cached: WeatherSnapshot | None = None
        # forecast 캐시 — 날짜 라벨별 별도 (예: "내일" → ForecastSnapshot)
        self._cached_forecast: dict[str, ForecastSnapshot] = {}
        self._lock = asyncio.Lock()
        self._forecast_lock = asyncio.Lock()

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

    # ─── forecast (내일 등 미래 날짜) ───

    def _build_forecast_url(self) -> str:
        params = {
            "lat": f"{self.lat:.4f}",
            "lon": f"{self.lon:.4f}",
            "appid": self.api_key,
            "units": "metric",
            "lang": "kr",
        }
        return f"{_OWM_FORECAST_URL}?{urllib.parse.urlencode(params)}"

    def _fetch_forecast_sync(self) -> list[dict]:
        """5일 3시간 forecast의 list 그대로 반환. 파싱은 호출부가."""
        req = urllib.request.Request(
            self._build_forecast_url(),
            headers={"User-Agent": "roboface/1.0"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        items = payload.get("list")
        if not isinstance(items, list):
            payload_str = str(payload)[:200]
            raise RuntimeError(f"OpenWeather forecast.list 없음 (payload={payload_str})")
        return items

    def _summarize_day(
        self, items: list[dict], target_date,
        date_label: str,
    ) -> ForecastSnapshot | None:
        """list에서 target_date(naive date) 항목만 추려 일일 요약 만듦.

        대표 description은 9~21시(낮 활동 시간) 윈도우에서 가장 자주 나온 값.
        해당 윈도우 항목 없으면 전체 day 항목 중 majority.
        """
        from collections import Counter
        from datetime import datetime, timezone

        day_items = []
        day_time_items = []   # 9~21시 윈도우
        for item in items:
            dt_ts = item.get("dt")
            if dt_ts is None:
                continue
            # local timezone — 단순화로 KST(UTC+9) 가정. WEATHER_LAT/LON이 한국 좌표 전제.
            # 좀 더 정확히 하려면 OWM 응답의 city.timezone (sec offset) 사용 가능.
            local = datetime.fromtimestamp(int(dt_ts), tz=timezone.utc).astimezone()
            if local.date() != target_date:
                continue
            day_items.append(item)
            if 9 <= local.hour < 21:
                day_time_items.append(item)

        if not day_items:
            return None

        temps = [float(it["main"]["temp"]) for it in day_items if "main" in it]
        if not temps:
            return None
        temp_min = min(temps)
        temp_max = max(temps)
        pops = [float(it.get("pop", 0.0)) for it in day_items]
        pop_max = max(pops) if pops else 0.0
        humidities = [int(it["main"].get("humidity", 0)) for it in day_items]
        humidity_avg = int(sum(humidities) / len(humidities)) if humidities else 0

        rep_items = day_time_items or day_items
        descriptions = [
            str(it["weather"][0]["description"]) for it in rep_items
            if it.get("weather") and len(it["weather"]) > 0
        ]
        if descriptions:
            description = Counter(descriptions).most_common(1)[0][0]
        else:
            description = "?"

        return ForecastSnapshot(
            date_label=date_label,
            description=description,
            temp_min=temp_min,
            temp_max=temp_max,
            pop_max=pop_max,
            humidity=humidity_avg,
            location_name=self.location_name,
            fetched_at=time.time(),
        )

    async def forecast_for_tomorrow(self) -> ForecastSnapshot | None:
        """내일(local 자정~다음 자정) forecast 요약. 1시간 캐시.

        네트워크 실패 + 캐시 없으면 None. agent/voice_command가 안내 메시지 띄움.
        """
        if not self.api_key:
            return None

        from datetime import datetime, timedelta

        cache_key = "tomorrow"
        now = time.time()
        cached = self._cached_forecast.get(cache_key)
        if cached is not None and (now - cached.fetched_at) < self.forecast_cache_sec:
            return cached

        async with self._forecast_lock:
            cached = self._cached_forecast.get(cache_key)
            if cached is not None and (time.time() - cached.fetched_at) < self.forecast_cache_sec:
                return cached
            loop = asyncio.get_running_loop()
            try:
                items = await loop.run_in_executor(None, self._fetch_forecast_sync)
            except Exception as e:
                log.warning(f"forecast fetch 실패 (stale fallback): {e}")
                return self._cached_forecast.get(cache_key)   # stale or None

            tomorrow = (datetime.now().astimezone() + timedelta(days=1)).date()
            snap = self._summarize_day(items, tomorrow, "내일")
            if snap is None:
                log.warning("forecast: 내일 데이터 없음 (forecast 범위 밖?)")
                return self._cached_forecast.get(cache_key)
            self._cached_forecast[cache_key] = snap
            log.info(f"forecast fetched: {snap.one_liner()}")
            return snap


# 모듈 레벨 싱글톤 — agent.py가 import해서 그대로 사용
_client_singleton: WeatherClient | None = None


def get_client() -> WeatherClient:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = WeatherClient()
    return _client_singleton
