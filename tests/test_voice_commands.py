"""voice_commands.VoiceCommandHandler — 트리거 매칭 + nmcli 호출."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from src.tasks.voice_commands import (
    PHONE_TETHER_SSID,
    VoiceCommandHandler,
    _normalize,
)


class _FaceStub:
    def __init__(self) -> None:
        self.expression_name = "neutral"
        self.speech_calls: list[tuple[str, float]] = []

    class _Expr:
        def __init__(self, name: str) -> None:
            self.name = name

    def __post_init__(self) -> None:
        pass

    @property
    def expression(self):
        return self._Expr(self.expression_name)

    def apply_expression(self, exp) -> None:
        self.expression_name = exp.name

    def show_speech(self, text: str, duration_sec: float) -> None:
        self.speech_calls.append((text, duration_sec))


def test_normalize_strips_whitespace_and_punct():
    assert _normalize("디버그 모드") == "디버그모드"
    assert _normalize("디버그 모드.") == "디버그모드"
    assert _normalize("디버그 모드!!!") == "디버그모드"
    assert _normalize(" 디버그모드 ") == "디버그모드"


def test_trigger_matches_various_phrasings():
    """'디버그 모드입니다' 등 추가 텍스트와 함께 와도 매칭. consumed True 반환."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        r1 = asyncio.run(handler("디버그 모드"))
        r2 = asyncio.run(handler("이거 디버그모드 시작"))  # cooldown으로 nmcli 발동 X
    # 둘 다 consumed (cooldown 케이스도 명령 인식 자체는 됐으니)
    assert r1 is True
    assert r2 is True
    # 실제 nmcli는 첫번째만
    assert m.call_count == 1


def test_no_trigger_for_unrelated_text():
    """일반 발화는 False 반환 — ambient_listener가 일반 흐름 진행."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        r1 = asyncio.run(handler("디버깅이 어려워"))
        r2 = asyncio.run(handler("오늘 점심 뭐 먹지"))
    assert r1 is False
    assert r2 is False
    assert m.call_count == 0


def test_success_path_sets_happy_and_message():
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ):
        asyncio.run(handler("디버그 모드"))
    assert face.expression_name == "happy"
    # FOCUSED 표시 → HAPPY 전환 도중 둘 다 show_speech 발동
    texts = [c[0] for c in face.speech_calls]
    assert any(PHONE_TETHER_SSID in t for t in texts)
    assert any("연결" in t for t in texts)


def test_fallback_chain_tries_second_wifi_on_phone_failure():
    """폰 실패 시 fallback chain의 첫 wifi로 자동 전환."""
    from src.tasks.voice_commands import WIFI_FALLBACK_CHAIN

    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    call_log: list[str] = []

    async def fake_up(ssid, timeout=15.0):
        call_log.append(ssid)
        if ssid == PHONE_TETHER_SSID:
            return (1, "", "Error: Wi-Fi network could not be found")
        # 첫 fallback 성공
        if ssid == WIFI_FALLBACK_CHAIN[0]:
            return (0, "Connection activated", "")
        return (1, "", "Error: timeout")

    with patch.object(VoiceCommandHandler, "_run_nmcli_up", side_effect=fake_up):
        asyncio.run(handler("디버그 모드"))

    # 폰 먼저 시도, 그 다음 첫 fallback
    assert call_log[0] == PHONE_TETHER_SSID
    assert call_log[1] == WIFI_FALLBACK_CHAIN[0]
    # 첫 fallback 성공 → CONTENT
    assert face.expression_name == "content"
    texts = [c[0] for c in face.speech_calls]
    assert any(WIFI_FALLBACK_CHAIN[0] in t for t in texts)


def test_all_wifi_fail_gives_worried():
    """폰 + 모든 fallback 다 실패 시 WORRIED + 실패 메시지."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(1, "", "Error: not found"),
    ):
        asyncio.run(handler("디버그 모드"))

    assert face.expression_name == "worried"
    texts = [c[0] for c in face.speech_calls]
    assert any("전부 실패" in t or "wifi 연결" in t for t in texts)


# ─── 셧다운 confirm 패턴 ───

def test_shutdown_first_trigger_enters_pending():
    """첫 셧다운 발화는 pending 진입만 — poweroff 호출 X."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_shutdown") as m:
        r = asyncio.run(handler("셧다운"))
    assert r is True
    assert m.call_count == 0
    assert handler._shutdown_pending_until > 0
    assert face.expression_name == "worried"
    texts = [c[0] for c in face.speech_calls]
    assert any("정말" in t or "다시" in t for t in texts)


def test_shutdown_second_trigger_within_window_executes():
    """confirm 윈도우 안에 두번째 발화 → poweroff 호출."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_shutdown") as m:
        asyncio.run(handler("셧다운"))      # 1단계
        asyncio.run(handler("셧다운"))      # 2단계 (즉시 = 윈도우 안)
    assert m.call_count == 1
    assert handler._shutdown_pending_until == 0.0


def test_shutdown_cancel_clears_pending():
    """pending 상태에서 '취소' → 해제, poweroff 호출 X."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_shutdown") as m:
        asyncio.run(handler("셧다운"))
        r = asyncio.run(handler("취소"))
    assert r is True
    assert m.call_count == 0
    assert handler._shutdown_pending_until == 0.0
    assert face.expression_name == "content"


def test_shutdown_timeout_auto_cancels(monkeypatch):
    """confirm 윈도우 지나면 자동 취소 — 다음 발화는 confirm 무관."""
    import src.tasks.voice_commands as vc
    monkeypatch.setattr(vc, "_SHUTDOWN_CONFIRM_SEC", 0.01)
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_shutdown") as m:
        asyncio.run(handler("셧다운"))
        import time as _t
        _t.sleep(0.05)
        # timeout 후 두번째 발화 — 새 pending 진입 (실행 X)
        r = asyncio.run(handler("셧다운"))
    assert r is True
    assert m.call_count == 0   # timeout으로 직전 confirm 사라짐


def test_shutdown_trigger_variations_match():
    """다양한 표현 매칭."""
    face = _FaceStub()
    for phrase in ["셧다운", "shutdown", "전원 꺼", "전원종료"]:
        handler = VoiceCommandHandler(face)
        r = asyncio.run(handler(phrase))
        assert r is True, f"매칭 실패: {phrase}"
        assert handler._shutdown_pending_until > 0


def test_single_short_word_kkeo_does_not_trigger():
    """단독 '꺼'는 너무 false positive 위험 → 매칭 X (전원꺼만)."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    r = asyncio.run(handler("꺼"))
    assert r is False


# ─── 재시작 confirm 패턴 ───

def test_restart_first_trigger_enters_pending():
    """첫 재시작 발화 → pending 진입만, systemctl 호출 X."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_restart") as m:
        r = asyncio.run(handler("재시작"))
    assert r is True
    assert m.call_count == 0
    assert handler._restart_pending_until > 0
    assert face.expression_name == "focused"


def test_restart_second_trigger_within_window_executes():
    """confirm 윈도우 안에 두번째 발화 → systemctl restart 호출."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_restart") as m:
        asyncio.run(handler("재시작"))
        asyncio.run(handler("재시작"))
    assert m.call_count == 1
    assert handler._restart_pending_until == 0.0


def test_restart_cancel_clears_pending():
    """pending 상태에서 '취소' → 해제, systemctl 호출 X."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_restart") as m:
        asyncio.run(handler("재시작"))
        r = asyncio.run(handler("취소"))
    assert r is True
    assert m.call_count == 0
    assert handler._restart_pending_until == 0.0


def test_restart_trigger_variations_match():
    """다양한 표현 매칭."""
    face = _FaceStub()
    for phrase in ["재시작", "다시 시작", "리스타트", "restart", "리셋"]:
        h = VoiceCommandHandler(face)
        r = asyncio.run(h(phrase))
        assert r is True, f"매칭 실패: {phrase}"
        assert h._restart_pending_until > 0


def test_restart_independent_from_shutdown_pending():
    """셧다운 pending이 진행 중이어도 재시작은 별도 pending state."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(VoiceCommandHandler, "_perform_shutdown") as m_s, \
         patch.object(VoiceCommandHandler, "_perform_restart") as m_r:
        asyncio.run(handler("셧다운"))    # shutdown pending
        # 셧다운 pending 동안 "재시작" 발화는 셧다운 confirm 안의 "그 외"로
        # 처리돼 일단 False 반환 (셧다운 pending 유지)
        r = asyncio.run(handler("재시작"))
    # 셧다운 cancel/confirm 키워드 아니므로 pass through — False
    assert r is False
    assert m_s.call_count == 0
    assert m_r.call_count == 0
    # 셧다운 pending 그대로
    assert handler._shutdown_pending_until > 0
    assert handler._restart_pending_until == 0.0


# ─── 날씨 명령 ───

def _weather_snap(line: str):
    """WeatherSnapshot stub — one_liner만 반환."""
    from dataclasses import dataclass

    @dataclass
    class _S:
        _line: str
        def one_liner(self) -> str:
            return self._line
    return _S(line)


def test_weather_trigger_shows_snapshot_line():
    """'날씨 알려줘' → snapshot.one_liner() LCD에 표시 + CONTENT."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    snap = _weather_snap("분당 맑음 18°C · 습도 45%")

    async def fake_snapshot():
        return snap

    with patch(
        "src.integrations.weather.get_client",
        return_value=type("C", (), {"snapshot": staticmethod(fake_snapshot)}),
    ):
        r = asyncio.run(handler("날씨 알려줘"))

    assert r is True
    assert face.expression_name == "content"
    texts = [c[0] for c in face.speech_calls]
    assert any("분당" in t and "18" in t for t in texts)


def test_weather_no_api_key_returns_worried():
    """snapshot()이 None(키 없음) → WORRIED + 안내 메시지."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)

    async def fake_snapshot():
        return None

    with patch(
        "src.integrations.weather.get_client",
        return_value=type("C", (), {"snapshot": staticmethod(fake_snapshot)}),
    ):
        r = asyncio.run(handler("날씨 어때"))

    assert r is True
    assert face.expression_name == "worried"
    texts = [c[0] for c in face.speech_calls]
    assert any("정보 없음" in t or "API 키" in t for t in texts)


def test_weather_trigger_variations():
    """여러 표현 매칭."""
    snap = _weather_snap("분당 맑음 18°C")

    async def fake_snapshot():
        return snap

    for phrase in [
        "날씨 알려줘", "날씨 어때", "오늘 날씨", "지금 날씨",
        "날씨가 어때", "weather", "날씨 알려주세요",
    ]:
        face = _FaceStub()
        handler = VoiceCommandHandler(face)
        with patch(
            "src.integrations.weather.get_client",
            return_value=type("C", (), {"snapshot": staticmethod(fake_snapshot)}),
        ):
            r = asyncio.run(handler(phrase))
        assert r is True, f"매칭 실패: {phrase}"


def test_weather_unrelated_does_not_match():
    """일반 대화('날씨 좋네')는 매칭 X."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    r = asyncio.run(handler("날씨 좋네"))
    assert r is False


def test_weather_cooldown_blocks_rapid_repeat():
    """짧은 cooldown — 같은 요청 반복 시 LCD 깜빡 방지."""
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    snap = _weather_snap("분당 맑음 18°C")

    call_count = {"n": 0}

    async def fake_snapshot():
        call_count["n"] += 1
        return snap

    with patch(
        "src.integrations.weather.get_client",
        return_value=type("C", (), {"snapshot": staticmethod(fake_snapshot)}),
    ):
        asyncio.run(handler("날씨 알려줘"))
        asyncio.run(handler("날씨 알려줘"))   # cooldown
    # 첫번째만 실제 snapshot 호출
    assert call_count["n"] == 1


def test_cooldown_blocks_second_trigger_within_30s():
    face = _FaceStub()
    handler = VoiceCommandHandler(face)
    with patch.object(
        VoiceCommandHandler, "_run_nmcli_up",
        return_value=(0, "OK", ""),
    ) as m:
        asyncio.run(handler("디버그 모드"))
        # 즉시 다시 — cooldown
        asyncio.run(handler("디버그 모드"))
    assert m.call_count == 1
