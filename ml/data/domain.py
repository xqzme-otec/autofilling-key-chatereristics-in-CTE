from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Category(str, Enum):
    CLOTHING = "clothing"
    FURNITURE = "furniture"
    TOOLS = "tools"


class CharacteristicType(str, Enum):
    QUALITATIVE = "qualitative"
    QUANTITATIVE = "quantitative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OkeiUnit:
    code: Optional[str] = None
    name: Optional[str] = None
    symbol: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacteristicValue:
    value: Optional[str] = None
    quality_description: Optional[str] = None
    lower_bound: Optional[str] = None
    upper_bound: Optional[str] = None
    unit: Optional[OkeiUnit] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CharacteristicDefinition:
    code: Optional[str] = None
    canonical_name: Optional[str] = None
    type: CharacteristicType = CharacteristicType.UNKNOWN
    kind: Optional[str] = None
    actual: Optional[bool] = None
    is_required: Optional[bool] = None
    choice_type: Optional[str] = None
    allowed_values: List[CharacteristicValue] = field(default_factory=list)
    units: List[OkeiUnit] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KtruPosition:
    code: Optional[str]
    name: Optional[str]
    description: Optional[str] = None
    okpd2: List[str] = field(default_factory=list)
    category: Optional[Category] = None
    actual: Optional[bool] = None
    version: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    parent: Optional[str] = None
    ancestors: List[str] = field(default_factory=list)
    descendants: List[str] = field(default_factory=list)
    rubricators: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    okeis: List[OkeiUnit] = field(default_factory=list)
    characteristics: List[CharacteristicDefinition] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

