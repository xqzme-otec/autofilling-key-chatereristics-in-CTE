from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional


class Category(str, Enum):
    CLOTHING = "clothing"
    FURNITURE = "furniture"
    TOOLS = "tools"
    OTHER = "other"
    UNKNOWN = "unknown"


class CharacteristicType(str, Enum):
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"
    UNKNOWN = "unknown"


class MathNotation(str, Enum):
    GREATER = "greater"
    GREATER_OR_EQUAL = "greaterOrEqual"
    LESS = "less"
    LESS_OR_EQUAL = "lessOrEqual"
    EQUAL = "equal"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ValueRange:
    min_notation: Optional[MathNotation] = None
    min_notation_raw: Optional[str] = None
    min_value: Optional[Decimal] = None
    min_value_raw: Optional[str] = None
    max_notation: Optional[MathNotation] = None
    max_notation_raw: Optional[str] = None
    max_value: Optional[Decimal] = None
    max_value_raw: Optional[str] = None


@dataclass(frozen=True)
class CharacteristicValue:
    value_code: Optional[str] = None
    quality_description: Optional[str] = None
    okei_code: Optional[str] = None
    okei_name: Optional[str] = None
    value_format: Optional[str] = None
    ranges: List[ValueRange] = field(default_factory=list)
    concrete_values: List[str] = field(default_factory=list)
    raw_children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CharacteristicDefinition:
    code: Optional[str] = None
    name: Optional[str] = None
    type: CharacteristicType = CharacteristicType.UNKNOWN
    type_raw: Optional[str] = None
    kind: Optional[str] = None
    actual: Optional[bool] = None
    is_required: Optional[bool] = None
    choice_type: Optional[str] = None
    values: List[CharacteristicValue] = field(default_factory=list)


@dataclass(frozen=True)
class OkeiUnit:
    code: Optional[str] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ClassifierValue:
    code: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class ClassifierInfo:
    name: Optional[str] = None
    nsi_code: Optional[str] = None
    code: Optional[str] = None
    source: Optional[str] = None
    values: List[ClassifierValue] = field(default_factory=list)


@dataclass(frozen=True)
class ProductInfo:
    trade_name: Optional[str] = None
    manufacturer: Optional[str] = None
    characteristics_count: int = 0
    raw_children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RubricatorInfo:
    code: Optional[str] = None
    name: Optional[str] = None
    raw_children: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ParentPositionInfo:
    code: Optional[str] = None
    version: Optional[str] = None
    external_code: Optional[str] = None


@dataclass(frozen=True)
class KtruPosition:
    code: str
    name: Optional[str]
    version_code: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    okpd2_code: Optional[str] = None
    okpd2_name: Optional[str] = None
    category: Category = Category.UNKNOWN
    inclusion_date: Optional[str] = None
    publish_date: Optional[str] = None
    update_date: Optional[str] = None
    application_date_start: Optional[str] = None
    application_date_end: Optional[str] = None
    nsi_description: Optional[str] = None
    is_template: Optional[bool] = None
    no_new_features: Optional[bool] = None
    characteristics: List[CharacteristicDefinition] = field(default_factory=list)
    products: List[ProductInfo] = field(default_factory=list)
    rubricators: List[RubricatorInfo] = field(default_factory=list)
    parent_position_info: Optional[ParentPositionInfo] = None
    position_okeis: List[OkeiUnit] = field(default_factory=list)
    classifiers: List[ClassifierInfo] = field(default_factory=list)
    source_zip: str = ""
    source_xml: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_decimal(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    try:
        return Decimal(value.replace(",", "."))
    except (InvalidOperation, AttributeError):
        return None

