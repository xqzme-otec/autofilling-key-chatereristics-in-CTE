from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from ml.data.domain import Category


@dataclass(frozen=True)
class CategoryRule:
    okpd2_prefixes: List[str] = field(default_factory=list)
    rubricator_prefixes: List[str] = field(default_factory=list)
    product_prefixes: List[str] = field(default_factory=list)
    parent_prefixes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CategoryConfig:
    rules: Dict[Category, CategoryRule]

    @classmethod
    def default(cls) -> "CategoryConfig":
        return cls(
            rules={
                Category.CLOTHING: CategoryRule(okpd2_prefixes=["14"]),
                Category.FURNITURE: CategoryRule(okpd2_prefixes=["31"]),
                Category.TOOLS: CategoryRule(okpd2_prefixes=["25.73", "28.24"]),
            }
        )


def detect_category(
    okpd2: Iterable[str],
    rubricators: Iterable[str],
    products: Iterable[str],
    parent_codes: Iterable[str],
    config: Optional[CategoryConfig] = None,
) -> Optional[Category]:
    active_config = config or CategoryConfig.default()
    fields = {
        "okpd2_prefixes": list(okpd2),
        "rubricator_prefixes": list(rubricators),
        "product_prefixes": list(products),
        "parent_prefixes": list(parent_codes),
    }
    for category, rule in active_config.rules.items():
        if _rule_matches(fields, rule):
            return category
    return None


def _rule_matches(fields: Dict[str, List[str]], rule: CategoryRule) -> bool:
    for attr_name in ("okpd2_prefixes", "rubricator_prefixes", "product_prefixes", "parent_prefixes"):
        values = fields[attr_name]
        prefixes = getattr(rule, attr_name)
        if _any_prefix_match(values, prefixes):
            return True
    return False


def _any_prefix_match(values: Iterable[str], prefixes: Iterable[str]) -> bool:
    normalized_values = [value.strip() for value in values if value and value.strip()]
    normalized_prefixes = [prefix.strip() for prefix in prefixes if prefix and prefix.strip()]
    return any(value.startswith(prefix) for value in normalized_values for prefix in normalized_prefixes)

