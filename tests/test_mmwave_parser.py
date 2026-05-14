"""mmWave 패킷 파서 단위 테스트."""

from src.sensors.mmwave import _parse_payload


def _make_payload(
    target_state: int = 1,
    moving_distance: int = 120,
    moving_energy: int = 80,
    static_distance: int = 0,
    static_energy: int = 0,
    detection_distance: int = 120,
) -> bytes:
    """HLK-LD2410 표준 모드 페이로드 빌더."""
    return bytes([
        0x02, 0xAA,
        target_state,
        moving_distance & 0xff, (moving_distance >> 8) & 0xff,
        moving_energy,
        static_distance & 0xff, (static_distance >> 8) & 0xff,
        static_energy,
        detection_distance & 0xff, (detection_distance >> 8) & 0xff,
        0x55, 0x00,
    ])


def test_parse_no_target():
    p = _make_payload(target_state=0, moving_distance=0, detection_distance=0)
    r = _parse_payload(p)
    assert r is not None
    assert r.presence is False


def test_parse_moving_target():
    p = _make_payload(target_state=1, moving_distance=150, detection_distance=150)
    r = _parse_payload(p)
    assert r is not None
    assert r.presence is True
    assert r.static_presence is False
    assert r.distance_cm == 150


def test_parse_static_only():
    p = _make_payload(
        target_state=2, moving_distance=0, static_distance=80, detection_distance=80,
    )
    r = _parse_payload(p)
    assert r is not None
    assert r.presence is True
    assert r.static_presence is True
    assert r.distance_cm == 80


def test_parse_invalid_too_short():
    assert _parse_payload(b"\x02\xaa\x01") is None


def test_parse_invalid_header():
    p = bytearray(_make_payload())
    p[0] = 0xFF
    assert _parse_payload(bytes(p)) is None
