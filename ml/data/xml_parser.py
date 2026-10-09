from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set

from ml.data.category_filter import CategoryConfig, detect_category
from ml.data.domain import (
    CharacteristicDefinition,
    CharacteristicType,
    CharacteristicValue,
    KtruPosition,
    OkeiUnit,
)

LOGGER = logging.getLogger(__name__)

POSITION_TAG_CANDIDATES = {
    "nsiKTRUNew",
    "ktru",
    "ktruPosition",
    "position",
    "item",
}
CHARACTERISTICS_TAG_NAMES = {"characteristics", "characteristicList"}
CHARACTERISTIC_TAG_NAMES = {"characteristic", "characteristicInfo"}
VALUE_TAG_NAMES = {"value", "values", "characteristicValue"}
OKEI_TAG_NAMES = {"OKEI", "okei", "okeis"}
OKPD2_TAG_NAMES = {"OKPD2", "okpd2", "okpd2Code"}
RUBRICATOR_TAG_NAMES = {"rubricator", "rubricators"}
PRODUCT_TAG_NAMES = {"product", "products"}


def iter_ktru_positions(
    xml_path: Path,
    category_config: Optional[CategoryConfig] = None,
    include_uncategorized: bool = False,
) -> Iterator[KtruPosition]:
    """Stream KTRU positions from XML without loading the whole document."""
    context = ET.iterparse(xml_path, events=("end",))
    for _, elem in context:
        if _is_position_element(elem):
            position = parse_position_element(elem, category_config=category_config)
            if include_uncategorized or position.category is not None:
                yield position
            else:
                LOGGER.debug("Skipping KTRU position outside configured categories: %s", position.code)
            elem.clear()


def parse_position_element(
    elem: ET.Element,
    category_config: Optional[CategoryConfig] = None,
) -> KtruPosition:
    direct = _direct_child_text_map(elem)
    okpd2 = _collect_codes(elem, OKPD2_TAG_NAMES)
    rubricators = _collect_codes(elem, RUBRICATOR_TAG_NAMES)
    products = _collect_codes(elem, PRODUCT_TAG_NAMES)
    parent = _first_text_by_names(elem, {"parent", "parentCode", "parentKtruCode"})
    ancestors = _collect_codes(elem, {"ancestor", "ancestors"})
    descendants = _collect_codes(elem, {"descendant", "descendants"})
    parent_codes = [code for code in [parent, *ancestors] if code]

    category = detect_category(okpd2, rubricators, products, parent_codes, config=category_config)

    return KtruPosition(
        code=_first_from_map(direct, "code", "ktruCode"),
        name=_first_from_map(direct, "name", "positionName"),
        description=_first_from_map(direct, "description"),
        okpd2=okpd2,
        category=category,
        actual=_parse_bool(_first_from_map(direct, "actual")),
        version=_first_from_map(direct, "version"),
        start_date=_first_from_map(direct, "startDate", "start_date"),
        end_date=_first_from_map(direct, "endDate", "end_date"),
        parent=parent,
        ancestors=ancestors,
        descendants=descendants,
        rubricators=rubricators,
        products=products,
        okeis=_parse_okeis(elem),
        characteristics=_parse_characteristics(elem),
        raw=_raw_element(elem),
    )


def _is_position_element(elem: ET.Element) -> bool:
    name = _local_name(elem.tag)
    if name in POSITION_TAG_CANDIDATES:
        return _first_text_by_names(elem, {"code", "ktruCode"}) is not None and _first_text_by_names(
            elem, {"name", "positionName"}
        ) is not None
    return False


def _parse_characteristics(position_elem: ET.Element) -> List[CharacteristicDefinition]:
    result: List[CharacteristicDefinition] = []
    for container in _children_by_names(position_elem, CHARACTERISTICS_TAG_NAMES):
        for char_elem in _children_by_names(container, CHARACTERISTIC_TAG_NAMES):
            result.append(_parse_characteristic(char_elem))
    for char_elem in _children_by_names(position_elem, CHARACTERISTIC_TAG_NAMES):
        result.append(_parse_characteristic(char_elem))
    return result


def _parse_characteristic(elem: ET.Element) -> CharacteristicDefinition:
    direct = _direct_child_text_map(elem)
    units = _parse_okeis(elem)
    raw_type = _first_from_map(direct, "type")
    return CharacteristicDefinition(
        code=_first_from_map(direct, "code", "id"),
        canonical_name=_first_from_map(direct, "name", "canonicalName"),
        type=_parse_characteristic_type(raw_type, elem),
        kind=_first_from_map(direct, "kind"),
        actual=_parse_bool(_first_from_map(direct, "actual")),
        is_required=_parse_bool(_first_from_map(direct, "isRequired", "required")),
        choice_type=_first_from_map(direct, "choiceType"),
        allowed_values=_parse_values(elem),
        units=units,
        raw=_raw_element(elem),
    )


def _parse_characteristic_type(raw_type: Optional[str], elem: ET.Element) -> CharacteristicType:
    text = (raw_type or "").lower()
    if "qual" in text or "качеств" in text:
        return CharacteristicType.QUALITATIVE
    if "quant" in text or "колич" in text or _first_text_by_names(elem, {"lowerBound", "upperBound"}) is not None:
        return CharacteristicType.QUANTITATIVE
    return CharacteristicType.UNKNOWN


def _parse_values(elem: ET.Element) -> List[CharacteristicValue]:
    values: List[CharacteristicValue] = []
    for value_elem in _descendants_by_names(elem, VALUE_TAG_NAMES):
        if value_elem is elem:
            continue
        if _local_name(value_elem.tag) == "values":
            continue
        direct = _direct_child_text_map(value_elem)
        own_text = _clean_text(value_elem.text)
        value = _first_from_map(direct, "value", "name", "text") or own_text
        if value is None and not direct:
            continue
        values.append(
            CharacteristicValue(
                value=value,
                quality_description=_first_from_map(direct, "qualityDescription"),
                lower_bound=_first_from_map(direct, "lowerBound", "minValue", "from"),
                upper_bound=_first_from_map(direct, "upperBound", "maxValue", "to"),
                unit=next(iter(_parse_okeis(value_elem)), None),
                raw=_raw_element(value_elem),
            )
        )
    return values


def _parse_okeis(elem: ET.Element) -> List[OkeiUnit]:
    units: List[OkeiUnit] = []
    for okei_elem in _descendants_by_names(elem, OKEI_TAG_NAMES):
        direct = _direct_child_text_map(okei_elem)
        code = _first_from_map(direct, "code", "okeiCode") or _attr_value(okei_elem, "code")
        name = _first_from_map(direct, "name", "fullName") or _attr_value(okei_elem, "name")
        symbol = _first_from_map(direct, "symbol", "nationalSymbol") or _attr_value(okei_elem, "symbol")
        if code or name or symbol:
            units.append(OkeiUnit(code=code, name=name, symbol=symbol, raw=_raw_element(okei_elem)))
    return _dedupe_units(units)


def _collect_codes(elem: ET.Element, names: Set[str]) -> List[str]:
    values: List[str] = []
    for found in _descendants_by_names(elem, names):
        direct = _direct_child_text_map(found)
        value = (
            _first_from_map(direct, "code", "id", "name")
            or _attr_value(found, "code")
            or _attr_value(found, "id")
            or _clean_text(found.text)
        )
        if value:
            values.append(value)
    return _dedupe_strings(values)


def _children_by_names(elem: ET.Element, names: Set[str]) -> List[ET.Element]:
    return [child for child in list(elem) if _local_name(child.tag) in names]


def _descendants_by_names(elem: ET.Element, names: Set[str]) -> List[ET.Element]:
    return [child for child in elem.iter() if _local_name(child.tag) in names]


def _first_text_by_names(elem: ET.Element, names: Set[str]) -> Optional[str]:
    for child in elem.iter():
        if child is elem:
            continue
        if _local_name(child.tag) in names:
            text = _clean_text(child.text)
            if text:
                return text
    return None


def _direct_child_text_map(elem: ET.Element) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for child in list(elem):
        text = _clean_text(child.text)
        if text is not None:
            result[_local_name(child.tag)] = text
    return result


def _raw_element(elem: ET.Element) -> Dict[str, Any]:
    return {
        "tag": _local_name(elem.tag),
        "attributes": dict(elem.attrib),
        "children": [_local_name(child.tag) for child in list(elem)],
    }


def _first_from_map(values: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _attr_value(elem: ET.Element, name: str) -> Optional[str]:
    for key, value in elem.attrib.items():
        if _local_name(key) == name:
            return _clean_text(value)
    return None


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "да"}:
        return True
    if normalized in {"false", "0", "no", "n", "нет"}:
        return False
    return None


def _clean_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = " ".join(value.split())
    return stripped or None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _dedupe_units(units: Iterable[OkeiUnit]) -> List[OkeiUnit]:
    seen = set()
    result = []
    for unit in units:
        key = (unit.code, unit.name, unit.symbol)
        if key not in seen:
            seen.add(key)
            result.append(unit)
    return result

