from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ml.data.ktru_models import Category, KtruPosition


@dataclass(frozen=True)
class CategoryRule:
    okpd2_prefixes: List[str] = field(default_factory=list)
    rubricator_prefixes: List[str] = field(default_factory=list)
    parent_prefixes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CategoryConfig:
    rules: Dict[Category, CategoryRule]

    @classmethod
    def default(cls) -> "CategoryConfig":
        # Rules are based on official OKPD2 classification branches, not KTRU title keywords:
        # 14 - apparel; 31 - furniture; 25.73 - tools; 28.24 - power-driven hand tools.
        return cls(
            rules={
                Category.CLOTHING: CategoryRule(okpd2_prefixes=["14"]),
                Category.FURNITURE: CategoryRule(okpd2_prefixes=["31"]),
                Category.TOOLS: CategoryRule(okpd2_prefixes=["25.73", "28.24"]),
            }
        )


def classify_position(position: KtruPosition, config: Optional[CategoryConfig] = None) -> Category:
    active_config = config or CategoryConfig.default()
    okpd2_values = [position.okpd2_code] if position.okpd2_code else []
    rubricator_values = [r.code for r in position.rubricators if r.code]
    parent_values = []
    if position.parent_position_info and position.parent_position_info.code:
        parent_values.append(position.parent_position_info.code)

    for category, rule in active_config.rules.items():
        if _matches(okpd2_values, rule.okpd2_prefixes):
            return category
        if _matches(rubricator_values, rule.rubricator_prefixes):
            return category
        if _matches(parent_values, rule.parent_prefixes):
            return category
    return Category.OTHER if position.okpd2_code else Category.UNKNOWN


def load_category_config(path: Path) -> CategoryConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    rules = {}
    for category_name, raw_rule in data.items():
        category = Category(category_name)
        rules[category] = CategoryRule(
            okpd2_prefixes=list(raw_rule.get("okpd2_prefixes", [])),
            rubricator_prefixes=list(raw_rule.get("rubricator_prefixes", [])),
            parent_prefixes=list(raw_rule.get("parent_prefixes", [])),
        )
    return CategoryConfig(rules=rules)


def is_target_category(category: Category) -> bool:
    return category in {Category.CLOTHING, Category.FURNITURE, Category.TOOLS}


def _matches(values: Iterable[Optional[str]], prefixes: Iterable[str]) -> bool:
    clean_values = [value.strip() for value in values if value and value.strip()]
    clean_prefixes = [prefix.strip() for prefix in prefixes if prefix and prefix.strip()]
    return any(value.startswith(prefix) for value in clean_values for prefix in clean_prefixes)

