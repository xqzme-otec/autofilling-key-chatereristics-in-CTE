from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from ml.data.domain import Category, CharacteristicType, KtruPosition


@dataclass
class DataQualityReport:
    total_positions_seen: int = 0
    filtered_positions: Counter = field(default_factory=Counter)
    characteristic_count: Counter = field(default_factory=Counter)
    unique_characteristic_names: Dict[str, set] = field(default_factory=lambda: defaultdict(set))
    records_skipped: Counter = field(default_factory=Counter)
    positions_without_characteristics: List[str] = field(default_factory=list)
    characteristics_without_values: Counter = field(default_factory=Counter)
    characteristics_without_unit: Counter = field(default_factory=Counter)
    duplicate_ktru_codes: List[str] = field(default_factory=list)
    characteristic_types: Counter = field(default_factory=Counter)
    required_counts: Counter = field(default_factory=Counter)
    unknown_structures: Counter = field(default_factory=Counter)

    def as_text(self) -> str:
        lines = [f"TOTAL positions seen: {self.total_positions_seen}", "", "FILTERED:"]
        for category in Category:
            key = category.value
            lines.extend(
                [
                    f"{key}:",
                    f"  KTRU positions: {self.filtered_positions[key]}",
                    f"  characteristics: {self.characteristic_count[key]}",
                    f"  unique characteristic names: {len(self.unique_characteristic_names[key])}",
                ]
            )
        lines.extend(
            [
                "",
                f"records skipped: {dict(self.records_skipped)}",
                f"positions without characteristics: {len(self.positions_without_characteristics)}",
                f"characteristics without values: {dict(self.characteristics_without_values)}",
                f"characteristics without unit: {dict(self.characteristics_without_unit)}",
                f"duplicate KTRU codes: {self.duplicate_ktru_codes}",
                f"characteristic types: {dict(self.characteristic_types)}",
                f"required/optional: {dict(self.required_counts)}",
                f"unknown structures: {dict(self.unknown_structures)}",
            ]
        )
        return "\n".join(lines)


def build_report(all_positions: Iterable[KtruPosition], total_seen: int) -> DataQualityReport:
    report = DataQualityReport(total_positions_seen=total_seen)
    seen_codes = set()
    duplicate_codes = set()
    for position in all_positions:
        if position.category is None:
            report.records_skipped["outside_configured_categories"] += 1
            continue
        key = position.category.value
        report.filtered_positions[key] += 1
        if position.code:
            if position.code in seen_codes:
                duplicate_codes.add(position.code)
            seen_codes.add(position.code)
        if not position.characteristics:
            report.positions_without_characteristics.append(position.code or "<missing-code>")
        for characteristic in position.characteristics:
            report.characteristic_count[key] += 1
            if characteristic.canonical_name:
                report.unique_characteristic_names[key].add(characteristic.canonical_name)
            if not characteristic.allowed_values:
                report.characteristics_without_values[key] += 1
            if not characteristic.units and not any(value.unit for value in characteristic.allowed_values):
                report.characteristics_without_unit[key] += 1
            report.characteristic_types[characteristic.type.value] += 1
            if characteristic.is_required is True:
                report.required_counts["required"] += 1
            elif characteristic.is_required is False:
                report.required_counts["optional"] += 1
            else:
                report.required_counts["unknown"] += 1
            if characteristic.type == CharacteristicType.UNKNOWN:
                report.unknown_structures["unknown_characteristic_type"] += 1
    report.duplicate_ktru_codes = sorted(duplicate_codes)
    return report

