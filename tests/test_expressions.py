"""표정/얼굴 데이터 단위 테스트."""

from src.face import expressions as ex


def test_all_expressions_have_distinct_names():
    names = [e.name for e in ex.EXPRESSIONS_BY_NAME.values()]
    assert len(names) == len(set(names)), "표정 이름이 중복됨"


def test_get_returns_neutral_for_unknown():
    assert ex.get("nope") is ex.NEUTRAL


def test_get_returns_correct_for_known():
    assert ex.get("happy") is ex.HAPPY
    assert ex.get("love") is ex.LOVE


def test_expression_count():
    # 기본 12 + 추가 8
    assert len(ex.EXPRESSIONS_BY_NAME) >= 20
