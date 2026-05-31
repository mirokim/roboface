"""WeatherClient — TTL 캐시, 한 줄 포맷, fallback 동작."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest

from src.integrations.weather import WeatherClient, WeatherSnapshot


def _snap(**overrides) -> WeatherSnapshot:
    base = dict(
        description="맑음",
        temp_c=18.0,
        feels_like_c=16.0,
        humidity=45,
        wind_mps=2.0,
        location_name="분당",
        fetched_at=time.time(),
    )
    base.update(overrides)
    return WeatherSnapshot(**base)


def test_one_liner_feels_like_diff_shown():
    s = _snap(temp_c=18.0, feels_like_c=14.0)
    line = s.one_liner()
    assert "체감 14" in line
    assert "맑음" in line and "18" in line and "45" in line


def test_one_liner_feels_like_close_omitted():
    s = _snap(temp_c=18.0, feels_like_c=18.4)
    line = s.one_liner()
    assert "체감" not in line   # 차이 1°C 미만이면 생략


def test_one_liner_wind_threshold():
    weak = _snap(wind_mps=1.5).one_liner()
    strong = _snap(wind_mps=4.2).one_liner()
    assert "바람" not in weak
    assert "바람" in strong and "4m/s" in strong


def test_snapshot_returns_none_without_key():
    c = WeatherClient(api_key="")
    snap = asyncio.run(c.snapshot())
    assert snap is None


def test_snapshot_cache_hit_skips_fetch():
    c = WeatherClient(api_key="dummy", cache_sec=60.0)
    fake = _snap()
    c._cached = fake
    # _fetch_sync을 호출하면 테스트 실패 — 절대 안 불려야 함
    with patch.object(
        WeatherClient, "_fetch_sync",
        side_effect=AssertionError("cache hit인데 fetch 호출됨"),
    ):
        out = asyncio.run(c.snapshot())
    assert out is fake


def test_snapshot_expired_cache_refetches():
    c = WeatherClient(api_key="dummy", cache_sec=10.0)
    stale = _snap(fetched_at=time.time() - 999.0, description="흐림")
    c._cached = stale
    fresh = _snap(description="맑음")
    with patch.object(WeatherClient, "_fetch_sync", return_value=fresh):
        out = asyncio.run(c.snapshot())
    assert out is fresh
    assert c._cached is fresh


def test_snapshot_fetch_error_returns_stale():
    """네트워크 실패 시 마지막 캐시 그대로 반환 (있으면)."""
    c = WeatherClient(api_key="dummy", cache_sec=10.0)
    stale = _snap(fetched_at=time.time() - 999.0)
    c._cached = stale
    with patch.object(
        WeatherClient, "_fetch_sync", side_effect=RuntimeError("network down"),
    ):
        out = asyncio.run(c.snapshot())
    assert out is stale   # stale-while-error


def test_snapshot_fetch_error_no_cache_returns_none():
    c = WeatherClient(api_key="dummy", cache_sec=10.0)
    with patch.object(
        WeatherClient, "_fetch_sync", side_effect=RuntimeError("network down"),
    ):
        out = asyncio.run(c.snapshot())
    assert out is None


def test_fetch_sync_parses_owm_payload():
    """실제 OpenWeather 응답 형태 파싱 — 네트워크 모킹."""
    import json
    from io import BytesIO

    payload = {
        "weather": [{"main": "Clear", "description": "맑음"}],
        "main": {"temp": 17.3, "feels_like": 15.8, "humidity": 42},
        "wind": {"speed": 1.8},
    }

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def read(self):
            return json.dumps(payload).encode("utf-8")

    c = WeatherClient(api_key="dummy", location_name="분당")
    with patch("urllib.request.urlopen", return_value=FakeResp()):
        snap = c._fetch_sync()
    assert snap.description == "맑음"
    assert snap.temp_c == pytest.approx(17.3)
    assert snap.feels_like_c == pytest.approx(15.8)
    assert snap.humidity == 42
    assert snap.wind_mps == pytest.approx(1.8)
    assert snap.location_name == "분당"


def test_fetch_sync_missing_field_raises():
    """OWM 스키마 어긋나면 RuntimeError — snapshot()이 stale fallback 발동."""
    import json

    payload = {"weather": [{"main": "Clear"}]}  # main.temp 누락

    class FakeResp:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def read(self): return json.dumps(payload).encode("utf-8")

    c = WeatherClient(api_key="dummy")
    with patch("urllib.request.urlopen", return_value=FakeResp()):
        with pytest.raises(RuntimeError, match="파싱 실패"):
            c._fetch_sync()


def test_url_includes_coords_and_lang_kr():
    c = WeatherClient(api_key="abc", lat=37.5, lon=127.1)
    url = c._build_url()
    assert "lat=37.5000" in url
    assert "lon=127.1000" in url
    assert "appid=abc" in url
    assert "units=metric" in url
    assert "lang=kr" in url


# ─── forecast (내일) ───

def _fake_forecast_item(dt_unix: int, temp: float, desc: str, pop: float = 0.0,
                        humidity: int = 50):
    return {
        "dt": dt_unix,
        "main": {"temp": temp, "humidity": humidity},
        "weather": [{"main": "X", "description": desc}],
        "pop": pop,
    }


def test_forecast_snapshot_one_liner_with_pop():
    from src.integrations.weather import ForecastSnapshot

    snap = ForecastSnapshot(
        date_label="내일", description="비",
        temp_min=18.0, temp_max=22.0, pop_max=0.7, humidity=80,
        location_name="분당", fetched_at=time.time(),
    )
    line = snap.one_liner()
    assert "내일" in line and "분당" in line
    assert "비" in line and "18~22" in line
    assert "70%" in line   # pop 70%


def test_forecast_snapshot_one_liner_skips_low_pop():
    from src.integrations.weather import ForecastSnapshot

    snap = ForecastSnapshot(
        date_label="내일", description="맑음",
        temp_min=15.0, temp_max=25.0, pop_max=0.10, humidity=50,
        location_name="분당", fetched_at=time.time(),
    )
    line = snap.one_liner()
    assert "비올 확률" not in line   # 10% < 30% → 생략


def test_summarize_day_picks_min_max_and_majority_description():
    """3시간 간격 데이터에서 min/max + 대표 description."""
    from datetime import date as _date
    from datetime import datetime, timezone, timedelta

    c = WeatherClient(api_key="abc", location_name="분당")
    # 내일(local) 데이터 만들기 — KST 가정
    tomorrow_local = (datetime.now().astimezone() + timedelta(days=1)).date()
    # 9~21시 윈도우 중 "흐림" 3회, "맑음" 1회 → 대표 "흐림"
    local_tz = datetime.now().astimezone().tzinfo

    def at(hour: int, temp: float, desc: str, pop: float = 0.0):
        dt_local = datetime.combine(tomorrow_local, datetime.min.time(),
                                    tzinfo=local_tz).replace(hour=hour)
        dt_utc = dt_local.astimezone(timezone.utc)
        return _fake_forecast_item(int(dt_utc.timestamp()), temp, desc, pop)

    items = [
        at(0, 15.0, "맑음"),
        at(9, 20.0, "흐림"),
        at(12, 25.0, "흐림", pop=0.4),
        at(15, 27.0, "흐림", pop=0.6),
        at(18, 22.0, "맑음"),
    ]
    snap = c._summarize_day(items, tomorrow_local, "내일")
    assert snap is not None
    assert snap.temp_min == 15.0
    assert snap.temp_max == 27.0
    assert snap.description == "흐림"   # 9~21시 윈도우 majority
    assert snap.pop_max == pytest.approx(0.6)


def test_summarize_day_returns_none_when_no_data():
    from datetime import date as _date
    c = WeatherClient(api_key="abc")
    snap = c._summarize_day([], _date(2099, 1, 1), "내일")
    assert snap is None


def test_forecast_for_tomorrow_no_key_returns_none():
    import asyncio
    c = WeatherClient(api_key="")
    snap = asyncio.run(c.forecast_for_tomorrow())
    assert snap is None


def test_forecast_for_tomorrow_cache_hit_skips_fetch():
    import asyncio
    from src.integrations.weather import ForecastSnapshot

    c = WeatherClient(api_key="dummy", forecast_cache_sec=3600.0)
    fake = ForecastSnapshot(
        date_label="내일", description="맑음", temp_min=15.0, temp_max=25.0,
        pop_max=0.0, humidity=50, location_name="분당", fetched_at=time.time(),
    )
    c._cached_forecast["tomorrow"] = fake
    with patch.object(
        WeatherClient, "_fetch_forecast_sync",
        side_effect=AssertionError("cache hit인데 fetch 호출"),
    ):
        out = asyncio.run(c.forecast_for_tomorrow())
    assert out is fake


def test_forecast_for_tomorrow_fetch_error_returns_stale():
    import asyncio
    from src.integrations.weather import ForecastSnapshot

    c = WeatherClient(api_key="dummy", forecast_cache_sec=10.0)
    stale = ForecastSnapshot(
        date_label="내일", description="흐림", temp_min=10.0, temp_max=15.0,
        pop_max=0.5, humidity=70, location_name="분당",
        fetched_at=time.time() - 999.0,
    )
    c._cached_forecast["tomorrow"] = stale
    with patch.object(
        WeatherClient, "_fetch_forecast_sync",
        side_effect=RuntimeError("network down"),
    ):
        out = asyncio.run(c.forecast_for_tomorrow())
    assert out is stale
