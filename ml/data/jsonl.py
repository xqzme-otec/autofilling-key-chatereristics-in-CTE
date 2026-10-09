from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from ml.data.domain import KtruPosition


def write_positions_jsonl(positions: Iterable[KtruPosition], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as stream:
        for position in positions:
            stream.write(json.dumps(position, ensure_ascii=False, default=_json_default))
            stream.write("\n")
            count += 1
    return count


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

