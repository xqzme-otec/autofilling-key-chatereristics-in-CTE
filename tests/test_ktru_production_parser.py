from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from ml.data.ktru_filter import classify_position
from ml.data.ktru_models import Category, CharacteristicType, MathNotation
from ml.data.ktru_parser import find_ktru_xml_names, iter_positions_from_xml, iter_positions_from_zip
from ml.data.prepare_ktru import deduplicate_positions, prepare_dataset


XML_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<export xmlns="http://zakupki.gov.ru/oos/export/1" xmlns:oos="http://zakupki.gov.ru/oos/types/1">
  <nsiKTRUs>
    <oos:position>
      <oos:data>
        <oos:code>14.12.30.100-00001</oos:code>
        <oos:versionCode>14.12.30.100-000011</oos:versionCode>
        <oos:version>1</oos:version>
        <oos:name>Костюм рабочий</oos:name>
        <oos:OKPD2><oos:code>14.12.30</oos:code><oos:name>Спецодежда</oos:name></oos:OKPD2>
        <oos:status>ACTIVE</oos:status>
        <oos:OKEIs><oos:OKEI><oos:code>796</oos:code><oos:name>Штука</oos:name></oos:OKEI></oos:OKEIs>
        <oos:characteristics>
          <oos:characteristic>
            <oos:code>c1</oos:code><oos:name>Цвет</oos:name><oos:type>1</oos:type>
            <oos:kind>1</oos:kind><oos:actual>true</oos:actual><oos:isRequired>true</oos:isRequired><oos:choiceType>1</oos:choiceType>
            <oos:values>
              <oos:value><oos:valueCode>v1</oos:valueCode><oos:qualityDescription>синий</oos:qualityDescription></oos:value>
            </oos:values>
          </oos:characteristic>
          <oos:characteristic>
            <oos:code>c2</oos:code><oos:name>Длина</oos:name><oos:type>2</oos:type>
            <oos:isRequired>false</oos:isRequired><oos:choiceType>1</oos:choiceType>
            <oos:values>
              <oos:value>
                <oos:valueCode>v2</oos:valueCode>
                <oos:OKEI><oos:code>003</oos:code><oos:name>Миллиметр</oos:name></oos:OKEI>
                <oos:rangeSet><oos:valueRange><oos:minMathNotation>greaterOrEqual</oos:minMathNotation><oos:min>10.5</oos:min><oos:maxMathNotation>lessOrEqual</oos:maxMathNotation><oos:max>20</oos:max></oos:valueRange></oos:rangeSet>
              </oos:value>
            </oos:values>
          </oos:characteristic>
          <oos:characteristic>
            <oos:code>c3</oos:code><oos:name>Неизвестная</oos:name><oos:type>9</oos:type>
            <oos:values><oos:value><oos:rangeSet><oos:valueRange><oos:minMathNotation>weird</oos:minMathNotation><oos:min>x</oos:min></oos:valueRange></oos:rangeSet></oos:value></oos:values>
          </oos:characteristic>
        </oos:characteristics>
      </oos:data>
    </oos:position>
  </nsiKTRUs>
</export>
"""


def _write_zip(path: Path, xml_name: str = "nsiKTRUNew_all_actual_20261003020003_001.xml") -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr(xml_name, XML_FIXTURE)


def test_parse_one_position_code_name_and_okpd2() -> None:
    positions = list(iter_positions_from_xml(_bytes(XML_FIXTURE), "fixture.zip", "fixture.xml"))
    assert len(positions) == 1
    assert positions[0].code == "14.12.30.100-00001"
    assert positions[0].name == "Костюм рабочий"
    assert positions[0].okpd2_code == "14.12.30"


def test_parse_qualitative_and_quantitative_characteristics() -> None:
    position = list(iter_positions_from_xml(_bytes(XML_FIXTURE), "fixture.zip", "fixture.xml"))[0]
    assert position.characteristics[0].type == CharacteristicType.QUALITATIVE
    assert position.characteristics[0].values[0].quality_description == "синий"
    assert position.characteristics[1].type == CharacteristicType.QUANTITATIVE


def test_parse_okei_ranges_required_and_choice_type() -> None:
    position = list(iter_positions_from_xml(_bytes(XML_FIXTURE), "fixture.zip", "fixture.xml"))[0]
    characteristic = position.characteristics[1]
    value = characteristic.values[0]
    value_range = value.ranges[0]
    assert value.okei_code == "003"
    assert value.okei_name == "Миллиметр"
    assert str(value_range.min_value) == "10.5"
    assert value_range.min_notation == MathNotation.GREATER_OR_EQUAL
    assert value_range.max_notation == MathNotation.LESS_OR_EQUAL
    assert characteristic.is_required is False
    assert characteristic.choice_type == "1"


def test_missing_optional_unknown_type_and_unknown_math_notation() -> None:
    position = list(iter_positions_from_xml(_bytes(XML_FIXTURE), "fixture.zip", "fixture.xml"))[0]
    characteristic = position.characteristics[2]
    assert characteristic.actual is None
    assert characteristic.type == CharacteristicType.UNKNOWN
    assert characteristic.type_raw == "9"
    assert characteristic.values[0].ranges[0].min_notation == MathNotation.UNKNOWN


def test_xml_inside_zip(tmp_path: Path) -> None:
    zip_path = tmp_path / "001_nsiKTRUNew_all.zip"
    _write_zip(zip_path)
    positions = list(iter_positions_from_zip(zip_path))
    assert len(positions) == 1
    assert positions[0].source_zip == zip_path.name


def test_zip_traversal_member_is_ignored(tmp_path: Path) -> None:
    zip_path = tmp_path / "bad.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("../nsiKTRUNew_all_actual_20261003020003_001.xml", XML_FIXTURE)
    with ZipFile(zip_path) as archive:
        assert find_ktru_xml_names(archive) == []


def test_malformed_archive_is_recorded(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    (input_dir / "bad.zip").write_text("not a zip", encoding="utf-8")
    result = prepare_dataset(input_dir, tmp_path / "out.parquet")
    assert result.stats.failed_archives


def test_category_filtering_and_prepared_output(tmp_path: Path) -> None:
    input_dir = tmp_path / "raw"
    input_dir.mkdir()
    _write_zip(input_dir / "001_nsiKTRUNew_all.zip")
    result = prepare_dataset(input_dir, tmp_path / "out.parquet")
    assert result.stats.output_records == 1
    assert result.examples["clothing"].category == Category.CLOTHING
    assert classify_position(result.examples["clothing"]) == Category.CLOTHING


def test_duplicate_detection_selects_highest_version() -> None:
    first = list(iter_positions_from_xml(_bytes(XML_FIXTURE), "a.zip", "a.xml"))[0]
    second = list(iter_positions_from_xml(_bytes(XML_FIXTURE.replace("<oos:version>1</oos:version>", "<oos:version>2</oos:version>")), "b.zip", "b.xml"))[0]
    deduped, ambiguities = deduplicate_positions([first, second])
    assert ambiguities == 0
    assert len(deduped) == 1
    assert deduped[0].version == "2"


def _bytes(text: str):
    import io

    return io.BytesIO(text.encode("utf-8"))

