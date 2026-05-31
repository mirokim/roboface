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
