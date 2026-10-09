from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zipfile import BadZipFile, ZipFile

import pandas as pd

from ml.data.ktru_filter import CategoryConfig, classify_position, is_target_category, load_category_config
from ml.data.ktru_models import (
    Category,
    CharacteristicType,
    KtruPosition,
    MathNotation,
)
from ml.data.ktru_parser import find_ktru_xml_names, iter_ktru_zip_paths, iter_positions_from_xml

LOGGER = logging.getLogger(__name__)


@dataclass
class PrepareStats:
    zip_files_found: int = 0
    zip_files_processed: int = 0
    xml_files_processed: int = 0
    archives_without_actual_xml: List[str] = field(default_factory=list)
    failed_archives: List[str] = field(default_factory=list)
    positions_parsed: int = 0
    unique_ktru_codes: int = 0
    duplicate_codes: int = 0
    duplicate_code_version: int = 0
    duplicate_code_version_code: int = 0
    duplicate_ambiguities: int = 0
    status_counts: Counter = field(default_factory=Counter)
    category_counts: Counter = field(default_factory=Counter)
    total_characteristics: int = 0
    unique_characteristic_names: int = 0
    qualitative_count: int = 0
    quantitative_count: int = 0
    required_count: int = 0
    optional_count: int = 0
    missing_is_required: int = 0
    qualitative_values_count: int = 0
    quantitative_ranges_count: int = 0
    values_with_okei: int = 0
    values_without_okei: int = 0
    empty_values: int = 0
    positions_without_name: int = 0
    positions_without_okpd2: int = 0
    positions_without_characteristics: int = 0
    malformed_ranges: int = 0
    unknown_type_values: Counter = field(default_factory=Counter)
    unknown_math_notation: Counter = field(default_factory=Counter)
    parse_errors: List[str] = field(default_factory=list)
    output_records: int = 0
    output_path: Optional[str] = None
    output_size_bytes: Optional[int] = None
    total_parse_time_sec: float = 0.0
    avg_time_per_zip_sec: float = 0.0
    positions_per_sec: float = 0.0
    peak_rss_mb: Optional[float] = None
    category_rules: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def as_text(self) -> str:
        lines = ["CATEGORY RULES:"]
        for category, rule in self.category_rules.items():
            lines.append(f"  {category}: OKPD2 prefixes {', '.join(rule.get('okpd2_prefixes', [])) or '<none>'}")
        lines.extend(
            [
                "",
                "FILES:",
            f"  zip files found: {self.zip_files_found}",
            f"  zip files processed: {self.zip_files_processed}",
            f"  xml files processed: {self.xml_files_processed}",
            f"  archives without actual XML: {len(self.archives_without_actual_xml)}",
            f"  failed archives: {self.failed_archives}",
            "",
            "TOTAL:",
            f"  positions parsed: {self.positions_parsed}",
            f"  unique KTRU codes: {self.unique_ktru_codes}",
            f"  duplicate codes: {self.duplicate_codes}",
            f"  duplicate code+version: {self.duplicate_code_version}",
            f"  duplicate code+versionCode: {self.duplicate_code_version_code}",
            f"  duplicate ambiguities: {self.duplicate_ambiguities}",
            f"  status counts: {dict(self.status_counts)}",
            "",
            "CATEGORY COUNTS:",
            f"  clothing: {self.category_counts[Category.CLOTHING.value]}",
            f"  furniture: {self.category_counts[Category.FURNITURE.value]}",
            f"  tools: {self.category_counts[Category.TOOLS.value]}",
            f"  other: {self.category_counts[Category.OTHER.value]}",
            f"  unknown: {self.category_counts[Category.UNKNOWN.value]}",
            "",
            "CHARACTERISTICS:",
            f"  total characteristics: {self.total_characteristics}",
            f"  unique characteristic names: {self.unique_characteristic_names}",
            f"  qualitative count: {self.qualitative_count}",
            f"  quantitative count: {self.quantitative_count}",
            f"  required count: {self.required_count}",
            f"  optional count: {self.optional_count}",
            f"  missing isRequired: {self.missing_is_required}",
            "",
            "VALUES:",
            f"  qualitative values count: {self.qualitative_values_count}",
            f"  quantitative ranges count: {self.quantitative_ranges_count}",
            f"  values with OKEI: {self.values_with_okei}",
            f"  values without OKEI: {self.values_without_okei}",
            f"  empty values: {self.empty_values}",
            "",
            "DATA QUALITY:",
            f"  positions without name: {self.positions_without_name}",
            f"  positions without OKPD2: {self.positions_without_okpd2}",
            f"  positions without characteristics: {self.positions_without_characteristics}",
            f"  malformed ranges: {self.malformed_ranges}",
            f"  unknown type values: {dict(self.unknown_type_values)}",
            f"  unknown math notation: {dict(self.unknown_math_notation)}",
            f"  parse errors: {self.parse_errors[:10]}",
            "",
            "PERFORMANCE:",
            f"  total parse time seconds: {self.total_parse_time_sec:.3f}",
            f"  avg time per ZIP seconds: {self.avg_time_per_zip_sec:.3f}",
            f"  positions/sec: {self.positions_per_sec:.1f}",
            f"  resulting dataset size bytes: {self.output_size_bytes}",
            f"  peak RSS MB: {self.peak_rss_mb}",
            f"  output records: {self.output_records}",
                f"  output path: {self.output_path}",
            ]
        )
        return "\n".join(lines)


@dataclass
class PrepareResult:
    stats: PrepareStats
    examples: Dict[str, KtruPosition]


def prepare_dataset(
    input_dir: Path,
    output_path: Path,
    limit_zips: Optional[int] = None,
    category_config: Optional[CategoryConfig] = None,
) -> PrepareResult:
    started = time.perf_counter()
    config = category_config or CategoryConfig.default()
    stats = PrepareStats()
    stats.category_rules = _category_rules_for_report(config)
    zip_paths_all = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".zip")
    stats.zip_files_found = len(zip_paths_all)
    zip_paths = list(iter_ktru_zip_paths(input_dir, limit_zips=limit_zips))

    code_counts: Counter = Counter()
    code_version_counts: Counter = Counter()
    code_version_code_counts: Counter = Counter()
    characteristic_names = set()
    target_positions: List[KtruPosition] = []
    examples: Dict[str, KtruPosition] = {}

    for zip_index, zip_path in enumerate(zip_paths, start=1):
        LOGGER.info("Processing ZIP %s/%s: %s", zip_index, len(zip_paths), zip_path.name)
        try:
            with ZipFile(zip_path) as archive:
                xml_names = find_ktru_xml_names(archive)
                if not xml_names:
                    LOGGER.warning("Archive without nsiKTRUNew actual XML: %s", zip_path.name)
                    stats.archives_without_actual_xml.append(zip_path.name)
                    continue
                stats.zip_files_processed += 1
                for xml_name in sorted(xml_names):
                    stats.xml_files_processed += 1
                    with archive.open(xml_name) as stream:
                        for position in iter_positions_from_xml(stream, source_zip=zip_path.name, source_xml=xml_name):
                            category = classify_position(position, config=config)
                            position = replace(position, category=category)
                            _update_stats(stats, position, characteristic_names)
                            code_counts[position.code] += 1
                            code_version_counts[(position.code, position.version)] += 1
                            code_version_code_counts[(position.code, position.version_code)] += 1
                            if is_target_category(category):
                                target_positions.append(position)
                                if category.value not in examples or not examples[category.value].characteristics:
                                    examples[category.value] = position
        except BadZipFile as exc:
            message = f"{zip_path.name}: malformed ZIP ({exc})"
            LOGGER.warning(message)
            stats.failed_archives.append(message)
        except Exception as exc:  # noqa: BLE001 - keep full-run preprocessing resilient.
            message = f"{zip_path.name}: parse error ({type(exc).__name__}: {exc})"
            LOGGER.warning(message)
            stats.parse_errors.append(message)

    stats.unique_ktru_codes = len(code_counts)
    stats.duplicate_codes = sum(1 for count in code_counts.values() if count > 1)
    stats.duplicate_code_version = sum(1 for count in code_version_counts.values() if count > 1)
    stats.duplicate_code_version_code = sum(1 for count in code_version_code_counts.values() if count > 1)
    stats.unique_characteristic_names = len(characteristic_names)

    deduped_positions, duplicate_ambiguities = deduplicate_positions(target_positions)
    stats.duplicate_ambiguities = duplicate_ambiguities
    stats.output_records = len(deduped_positions)
    write_prepared_dataset(deduped_positions, output_path)
    stats.output_path = str(output_path)
    stats.output_size_bytes = output_path.stat().st_size if output_path.exists() else None

    elapsed = time.perf_counter() - started
    stats.total_parse_time_sec = elapsed
    stats.avg_time_per_zip_sec = elapsed / stats.zip_files_processed if stats.zip_files_processed else 0.0
    stats.positions_per_sec = stats.positions_parsed / elapsed if elapsed else 0.0
    stats.peak_rss_mb = _peak_rss_mb()
    return PrepareResult(stats=stats, examples=examples)


def deduplicate_positions(positions: Iterable[KtruPosition]) -> Tuple[List[KtruPosition], int]:
    by_code: Dict[str, List[KtruPosition]] = defaultdict(list)
    for position in positions:
        by_code[position.code].append(position)

    result: List[KtruPosition] = []
    ambiguities = 0
    for code, group in by_code.items():
        if len(group) == 1:
            result.append(group[0])
            continue
        selected = _select_latest_unambiguous(group)
        if selected is None:
            LOGGER.warning("Duplicate ambiguity for KTRU code %s; keeping all %s records", code, len(group))
            ambiguities += 1
            result.extend(group)
        else:
            LOGGER.info("Duplicate KTRU code %s resolved to version=%s versionCode=%s", code, selected.version, selected.version_code)
            result.append(selected)
    return result, ambiguities


def write_prepared_dataset(positions: List[KtruPosition], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_to_flat_row(position) for position in positions]
    if output_path.suffix.lower() == ".parquet":
        pd.DataFrame(rows).to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".jsonl":
        with output_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, default=_json_default))
                stream.write("\n")
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")


def _update_stats(stats: PrepareStats, position: KtruPosition, characteristic_names: set) -> None:
    stats.positions_parsed += 1
    stats.category_counts[position.category.value] += 1
    if position.status:
        stats.status_counts[position.status] += 1
    if not position.name:
        stats.positions_without_name += 1
    if not position.okpd2_code:
        stats.positions_without_okpd2 += 1
    if not position.characteristics:
        stats.positions_without_characteristics += 1

    for characteristic in position.characteristics:
        stats.total_characteristics += 1
        if characteristic.name:
            characteristic_names.add(characteristic.name)
        if characteristic.type == CharacteristicType.QUALITATIVE:
            stats.qualitative_count += 1
        elif characteristic.type == CharacteristicType.QUANTITATIVE:
            stats.quantitative_count += 1
        else:
            stats.unknown_type_values[characteristic.type_raw or "<missing>"] += 1
        if characteristic.is_required is True:
            stats.required_count += 1
        elif characteristic.is_required is False:
            stats.optional_count += 1
        else:
            stats.missing_is_required += 1
        if not characteristic.values:
            stats.empty_values += 1
        for value in characteristic.values:
            if characteristic.type == CharacteristicType.QUALITATIVE and value.quality_description:
                stats.qualitative_values_count += 1
            if value.okei_code or value.okei_name:
                stats.values_with_okei += 1
            else:
                stats.values_without_okei += 1
            stats.quantitative_ranges_count += len(value.ranges)
            for value_range in value.ranges:
                if value_range.min_value_raw and value_range.min_value is None:
                    stats.malformed_ranges += 1
                if value_range.max_value_raw and value_range.max_value is None:
                    stats.malformed_ranges += 1
                if value_range.min_notation == MathNotation.UNKNOWN:
                    stats.unknown_math_notation[value_range.min_notation_raw or "<missing>"] += 1
                if value_range.max_notation == MathNotation.UNKNOWN:
                    stats.unknown_math_notation[value_range.max_notation_raw or "<missing>"] += 1


def _select_latest_unambiguous(group: List[KtruPosition]) -> Optional[KtruPosition]:
    active_group = [position for position in group if position.status == "ACTIVE"] or group
    version_pairs = []
    for position in active_group:
        try:
            version_pairs.append((int(position.version or ""), position))
        except ValueError:
            return None
    if not version_pairs:
        return None
    max_version = max(version for version, _ in version_pairs)
    candidates = [position for version, position in version_pairs if version == max_version]
    if len(candidates) == 1:
        return candidates[0]
    version_codes = {position.version_code for position in candidates}
    if len(version_codes) == 1:
        return candidates[0]
    return None


def _to_flat_row(position: KtruPosition) -> Dict[str, Any]:
    return {
        "code": position.code,
        "name": position.name,
        "version_code": position.version_code,
        "version": position.version,
        "status": position.status,
        "okpd2_code": position.okpd2_code,
        "okpd2_name": position.okpd2_name,
        "category": position.category.value,
        "inclusion_date": position.inclusion_date,
        "publish_date": position.publish_date,
        "update_date": position.update_date,
        "application_date_start": position.application_date_start,
        "application_date_end": position.application_date_end,
        "nsi_description": position.nsi_description,
        "is_template": position.is_template,
        "no_new_features": position.no_new_features,
        "source_zip": position.source_zip,
        "source_xml": position.source_xml,
        "characteristics_json": json.dumps(position.characteristics, ensure_ascii=False, default=_json_default),
        "products_json": json.dumps(position.products, ensure_ascii=False, default=_json_default),
        "rubricators_json": json.dumps(position.rubricators, ensure_ascii=False, default=_json_default),
        "parent_position_info_json": json.dumps(position.parent_position_info, ensure_ascii=False, default=_json_default),
        "position_okeis_json": json.dumps(position.position_okeis, ensure_ascii=False, default=_json_default),
        "classifiers_json": json.dumps(position.classifiers, ensure_ascii=False, default=_json_default),
        "characteristics_count": len(position.characteristics),
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _peak_rss_mb() -> Optional[float]:
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes, Linux reports kilobytes.
        if rss > 10_000_000:
            return round(rss / (1024 * 1024), 2)
        return round(rss / 1024, 2)
    except Exception:
        return None


def print_examples(examples: Dict[str, KtruPosition]) -> None:
    print("\nSAMPLE POSITIONS:")
    for category in [Category.CLOTHING, Category.FURNITURE, Category.TOOLS]:
        position = examples.get(category.value)
        print(f"\n{category.value}:")
        if position is None:
            print("  not found")
            continue
        print(f"  code: {position.code}")
        print(f"  name: {position.name}")
        print(f"  OKPD2: {position.okpd2_code} {position.okpd2_name or ''}".rstrip())
        print(f"  category: {position.category.value}")
        for characteristic in position.characteristics[:5]:
            print(f"  - {characteristic.name} [{characteristic.type.value}] required={characteristic.is_required}")
            for value in characteristic.values[:3]:
                value_bits = []
                if value.quality_description:
                    value_bits.append(f"quality={value.quality_description}")
                if value.okei_code or value.okei_name:
                    value_bits.append(f"OKEI={value.okei_code} {value.okei_name}".strip())
                for value_range in value.ranges[:2]:
                    value_bits.append(
                        "range="
                        f"{value_range.min_notation_raw} {value_range.min_value_raw} "
                        f"{value_range.max_notation_raw} {value_range.max_value_raw}"
                    )
                if value_bits:
                    print("      " + "; ".join(value_bits))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KTRU dataset for hackathon categories.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-zips", type=int, default=None)
    parser.add_argument("--category-config", type=Path, default=Path("ml/config/ktru_categories.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s:%(name)s:%(message)s",
    )
    category_config = load_category_config(args.category_config)
    result = prepare_dataset(args.input_dir, args.output, limit_zips=args.limit_zips, category_config=category_config)
    print(result.stats.as_text())
    print_examples(result.examples)


def _category_rules_for_report(config: CategoryConfig) -> Dict[str, Dict[str, List[str]]]:
    return {
        category.value: {
            "okpd2_prefixes": rule.okpd2_prefixes,
            "rubricator_prefixes": rule.rubricator_prefixes,
            "parent_prefixes": rule.parent_prefixes,
        }
        for category, rule in config.rules.items()
    }


if __name__ == "__main__":
    main()

