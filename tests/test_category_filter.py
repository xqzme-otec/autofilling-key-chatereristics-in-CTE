from ml.data.category_filter import detect_category
from ml.data.domain import Category


def test_detect_category_from_okpd2_prefixes() -> None:
    assert detect_category(["14.12.30"], [], [], []) == Category.CLOTHING
    assert detect_category(["31.01.12"], [], [], []) == Category.FURNITURE
    assert detect_category(["25.73.30"], [], [], []) == Category.TOOLS


def test_unknown_category_is_not_included_by_default() -> None:
    assert detect_category(["10.10"], [], [], []) is None

