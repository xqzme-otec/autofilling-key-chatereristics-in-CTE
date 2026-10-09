from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, List, Optional
from zipfile import BadZipFile, ZipFile

from ml.data.ktru_models import (
    CharacteristicDefinition,
    CharacteristicType,
    CharacteristicValue,
    ClassifierInfo,
    ClassifierValue,
    KtruPosition,
    MathNotation,
    OkeiUnit,
    ParentPositionInfo,
    ProductInfo,
    RubricatorInfo,
    ValueRange,
    parse_decimal,
)

LOGGER = logging.getLogger(__name__)
OOS_NS = "http://zakupki.gov.ru/oos/types/1"
XML_NAME_PREFIX = "nsiktrunew_all_actual_"


class KtruArchiveError(Exception):
    pass


def iter_ktru_zip_paths(input_dir: Path, limit_zips: Optional[int] = None) -> Iterator[Path]:
    zip_paths = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".zip")
    if limit_zips is not None:
        zip_paths = zip_paths[:limit_zips]
    yield from zip_paths


def find_ktru_xml_names(zip_file: ZipFile) -> List[str]:
    return [
        name
        for name in zip_file.namelist()
        if name.lower().endswith(".xml") and Path(name).name.lower().startswith(XML_NAME_PREFIX)
        and _safe_zip_member(name)
    ]


def iter_positions_from_zip(zip_path: Path) -> Iterator[KtruPosition]:
    try:
        with ZipFile(zip_path) as archive:
            xml_names = find_ktru_xml_names(archive)
            if not xml_names:
                LOGGER.warning("Archive without nsiKTRUNew actual XML: %s", zip_path)
                return
            for xml_name in sorted(xml_names):
                with archive.open(xml_name) as stream:
                    yield from iter_positions_from_xml(stream, source_zip=zip_path.name, source_xml=xml_name)
    except BadZipFile as exc:
        raise KtruArchiveError(f"Malformed ZIP archive {zip_path}: {exc}") from exc


def iter_positions_from_xml(stream: BinaryIO, source_zip: str, source_xml: str) -> Iterator[KtruPosition]:
    context = ET.iterparse(stream, events=("end",))
    for _, elem in context:
        if _local(elem.tag) == "position":
            position = _parse_position(elem, source_zip=source_zip, source_xml=source_xml)
            if position is not None:
                yield position
            elem.clear()


def _parse_position(elem: ET.Element, source_zip: str, source_xml: str) -> Optional[KtruPosition]:
    data = _child(elem, "data")
    if data is None:
        LOGGER.warning("Malformed position without data in %s/%s", source_zip, source_xml)
        return None
    code = _text(data, "code")
    if not code:
        LOGGER.warning("Malformed position without code in %s/%s", source_zip, source_xml)
        return None
    okpd2 = _child(data, "OKPD2")
    return KtruPosition(
        code=code,
        name=_text(data, "name"),
        version_code=_text(data, "versionCode"),
        version=_text(data, "version"),
        status=_text(data, "status"),
        okpd2_code=_text(okpd2, "code") if okpd2 is not None else None,
        okpd2_name=_text(okpd2, "name") if okpd2 is not None else None,
        inclusion_date=_text(data, "inclusionDate"),
        publish_date=_text(data, "publishDate"),
        update_date=_text(data, "updateDate"),
        application_date_start=_text(data, "applicationDateStart"),
        application_date_end=_text(data, "applicationDateEnd"),
        nsi_description=_text(data, "nsiDescription"),
        is_template=_parse_bool(_text(data, "isTemplate")),
        no_new_features=_parse_bool(_text(data, "noNewFeatures")),
        characteristics=_parse_characteristics(data),
        products=_parse_products(data),
        rubricators=_parse_rubricators(data),
        parent_position_info=_parse_parent(data),
        position_okeis=_parse_position_okeis(data),
        classifiers=_parse_classifiers(data),
        source_zip=source_zip,
        source_xml=source_xml,
    )


def _parse_characteristics(data: ET.Element) -> List[CharacteristicDefinition]:
    container = _child(data, "characteristics")
    if container is None:
        return []
    result: List[CharacteristicDefinition] = []
    for char in _children(container, "characteristic"):
        type_raw = _text(char, "type")
        result.append(
            CharacteristicDefinition(
                code=_text(char, "code"),
                name=_text(char, "name"),
                type=_parse_characteristic_type(type_raw),
                type_raw=type_raw,
                kind=_text(char, "kind"),
                actual=_parse_bool(_text(char, "actual")),
                is_required=_parse_bool(_text(char, "isRequired")),
                choice_type=_text(char, "choiceType"),
                values=_parse_characteristic_values(char),
            )
        )
    return result


def _parse_characteristic_values(char: ET.Element) -> List[CharacteristicValue]:
    container = _child(char, "values")
    if container is None:
        return []
    values: List[CharacteristicValue] = []
    for value in _children(container, "value"):
        okei = _child(value, "OKEI")
        values.append(
            CharacteristicValue(
                value_code=_text(value, "valueCode"),
                quality_description=_text(value, "qualityDescription"),
                okei_code=_text(okei, "code") if okei is not None else None,
                okei_name=_text(okei, "name") if okei is not None else None,
                value_format=_text(value, "valueFormat"),
                ranges=_parse_ranges(value),
                concrete_values=_parse_concrete_values(value),
                raw_children=[_local(child.tag) for child in list(value)],
            )
        )
    return values


def _parse_ranges(value: ET.Element) -> List[ValueRange]:
    range_set = _child(value, "rangeSet")
    if range_set is None:
        return []
    ranges: List[ValueRange] = []
    for value_range in _children(range_set, "valueRange"):
        min_raw = _text(value_range, "min")
        max_raw = _text(value_range, "max")
        min_notation_raw = _text(value_range, "minMathNotation")
        max_notation_raw = _text(value_range, "maxMathNotation")
        ranges.append(
            ValueRange(
                min_notation=_parse_math_notation(min_notation_raw),
                min_notation_raw=min_notation_raw,
                min_value=parse_decimal(min_raw),
                min_value_raw=min_raw,
                max_notation=_parse_math_notation(max_notation_raw),
                max_notation_raw=max_notation_raw,
                max_value=parse_decimal(max_raw),
                max_value_raw=max_raw,
            )
        )
    return ranges


def _parse_concrete_values(value: ET.Element) -> List[str]:
    value_set = _child(value, "valueSet")
    if value_set is None:
        return []
    return [_clean(node.text) for node in _children(value_set, "concreteValue") if _clean(node.text) is not None]


def _parse_position_okeis(data: ET.Element) -> List[OkeiUnit]:
    container = _child(data, "OKEIs")
    if container is None:
        return []
    return [OkeiUnit(code=_text(okei, "code"), name=_text(okei, "name")) for okei in _children(container, "OKEI")]


def _parse_parent(data: ET.Element) -> Optional[ParentPositionInfo]:
    parent = _child(data, "parentPositionInfo")
    if parent is None:
        return None
    return ParentPositionInfo(
        code=_text(parent, "code"),
        version=_text(parent, "version"),
        external_code=_text(parent, "externalCode"),
    )


def _parse_classifiers(data: ET.Element) -> List[ClassifierInfo]:
    nsi = _child(data, "NSI")
    classifiers = _child(nsi, "classifiers") if nsi is not None else None
    if classifiers is None:
        return []
    result: List[ClassifierInfo] = []
    for classifier in _children(classifiers, "classifier"):
        values_container = _child(classifier, "values")
        values = []
        if values_container is not None:
            values = [
                ClassifierValue(
                    code=_text(value, "code"),
                    name=_text(value, "name"),
                    description=_text(value, "descriptionValue"),
                )
                for value in _children(values_container, "value")
            ]
        result.append(
            ClassifierInfo(
                name=_text(classifier, "name"),
                nsi_code=_text(classifier, "NSIcode"),
                code=_text(classifier, "code"),
                source=_text(classifier, "source"),
                values=values,
            )
        )
    return result


def _parse_products(data: ET.Element) -> List[ProductInfo]:
    container = _child(data, "products")
    if container is None:
        return []
    products: List[ProductInfo] = []
    for product in _children(container, "product"):
        chars = _child(product, "characteristics")
        products.append(
            ProductInfo(
                trade_name=_text(product, "tradeName"),
                manufacturer=_text(product, "manufacturer"),
                characteristics_count=len(_children(chars, "characteristic")) if chars is not None else 0,
                raw_children=[_local(child.tag) for child in list(product)],
            )
        )
    return products


def _parse_rubricators(data: ET.Element) -> List[RubricatorInfo]:
    container = _child(data, "rubricators")
    if container is None:
        return []
    result: List[RubricatorInfo] = []
    for rubricator in list(container):
        result.append(
            RubricatorInfo(
                code=_text(rubricator, "code"),
                name=_text(rubricator, "name"),
                raw_children=[_local(child.tag) for child in list(rubricator)],
            )
        )
    return result


def _parse_characteristic_type(raw: Optional[str]) -> CharacteristicType:
    if raw == "1":
        return CharacteristicType.QUALITATIVE
    if raw == "2":
        return CharacteristicType.QUANTITATIVE
    return CharacteristicType.UNKNOWN


def _parse_math_notation(raw: Optional[str]) -> Optional[MathNotation]:
    if raw is None:
        return None
    mapping = {
        "greater": MathNotation.GREATER,
        "greaterOrEqual": MathNotation.GREATER_OR_EQUAL,
        "less": MathNotation.LESS,
        "lessOrEqual": MathNotation.LESS_OR_EQUAL,
        "equal": MathNotation.EQUAL,
    }
    return mapping.get(raw, MathNotation.UNKNOWN)


def _parse_bool(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    value = raw.strip().lower()
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def _child(elem: Optional[ET.Element], name: str) -> Optional[ET.Element]:
    if elem is None:
        return None
    for child in list(elem):
        if _local(child.tag) == name:
            return child
    return None


def _children(elem: Optional[ET.Element], name: str) -> List[ET.Element]:
    if elem is None:
        return []
    return [child for child in list(elem) if _local(child.tag) == name]


def _text(elem: Optional[ET.Element], child_name: str) -> Optional[str]:
    child = _child(elem, child_name)
    if child is None:
        return None
    return _clean(child.text)


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(value.split())
    return cleaned or None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_zip_member(name: str) -> bool:
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts

