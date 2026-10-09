from pathlib import Path

from ml.data.domain import Category, CharacteristicType
from ml.data.xml_parser import iter_ktru_positions


SAMPLE_PATH = Path("data/samples/ktru_sample.xml")


def _positions():
    return list(iter_ktru_positions(SAMPLE_PATH, include_uncategorized=True))


def test_reads_single_ktru_position_code_and_name() -> None:
    position = _positions()[0]
    assert position.code == "14.12.30.100-00000001"
    assert position.name == "Костюм рабочий"


def test_reads_okpd2_and_category_linkage() -> None:
    positions = _positions()
    assert positions[0].okpd2 == ["14.12.30"]
    assert positions[0].category == Category.CLOTHING
    assert positions[1].category == Category.FURNITURE
    assert positions[2].category == Category.TOOLS


def test_reads_qualitative_characteristics() -> None:
    position = _positions()[0]
    characteristic = position.characteristics[0]
    assert characteristic.canonical_name == "Цвет"
    assert characteristic.type == CharacteristicType.QUALITATIVE


def test_reads_quantitative_characteristics() -> None:
    position = _positions()[1]
    characteristic = position.characteristics[0]
    assert characteristic.type == CharacteristicType.QUANTITATIVE
    assert characteristic.allowed_values[0].lower_bound == "800"
    assert characteristic.allowed_values[0].upper_bound == "1200"


def test_reads_is_required() -> None:
    position = _positions()[0]
    assert position.characteristics[0].is_required is True
    assert position.characteristics[1].is_required is False


def test_reads_choice_type() -> None:
    assert _positions()[0].characteristics[0].choice_type == "single"


def test_reads_multiple_allowed_values() -> None:
    values = [value.value for value in _positions()[0].characteristics[0].allowed_values]
    assert values == ["синий", "черный"]


def test_reads_okei_from_value() -> None:
    unit = _positions()[1].characteristics[0].allowed_values[0].unit
    assert unit is not None
    assert unit.code == "006"
    assert unit.symbol == "м"


def test_missing_optional_element_does_not_crash_parser() -> None:
    material = _positions()[1].characteristics[1]
    assert material.allowed_values == []
    assert material.actual is None


def test_parser_excludes_positions_outside_three_categories() -> None:
    positions = list(iter_ktru_positions(SAMPLE_PATH))
    assert [position.code for position in positions] == [
        "14.12.30.100-00000001",
        "31.01.12.110-00000002",
        "25.73.30.299-00000003",
    ]


def test_unicode_russian_text_is_preserved() -> None:
    assert _positions()[2].name == "Отвертка"


def test_actual_positions_are_not_lost() -> None:
    actual_positions = [position for position in _positions() if position.actual is True]
    assert len(actual_positions) == 4

