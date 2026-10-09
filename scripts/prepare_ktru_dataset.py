from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.jsonl import write_positions_jsonl
from ml.data.report import build_report
from ml.data.xlsx_inspector import inspect_xlsx
from ml.data.xml_parser import iter_ktru_positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare compact KTRU dataset for configured hackathon categories.")
    parser.add_argument("--xml", type=Path, required=True, help="Path to nsiKTRUNew_all-actual_*.xml")
    parser.add_argument("--out", type=Path, default=Path("data/processed/ktru_three_categories.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("docs/reports/ktru_data_quality.txt"))
    parser.add_argument("--xlsx", type=Path, nargs="*", default=[], help="Optional supplier portal XLSX files to inspect.")
    args = parser.parse_args()

    start = time.perf_counter()
    positions_all = list(iter_ktru_positions(args.xml, include_uncategorized=True))
    filtered_positions = [position for position in positions_all if position.category is not None and position.actual is not False]
    written = write_positions_jsonl(filtered_positions, args.out)
    elapsed = time.perf_counter() - start

    report = build_report(positions_all, total_seen=len(positions_all))
    report_text = report.as_text()
    report_text += f"\n\nPERFORMANCE:\nparsing time seconds: {elapsed:.3f}\nresulting records: {written}\nrecords per second: {len(positions_all) / elapsed if elapsed else 0:.1f}\n"

    if args.xlsx:
        report_text += "\nXLSX INSPECTION:\n"
        for xlsx_path in args.xlsx:
            report_text += f"\n{xlsx_path}:\n"
            for sheet in inspect_xlsx(xlsx_path):
                report_text += json.dumps(sheet.__dict__, ensure_ascii=False, indent=2)
                report_text += "\n"

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report_text, encoding="utf-8")
    print(report_text)


if __name__ == "__main__":
    main()

