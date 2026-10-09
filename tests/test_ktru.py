import json

import pandas as pd
import pytest

from ml.ktru import DEFAULT_PATH, load_parquet


def _row(code, description):
    return {
        "code": code,
        "name": "Молоток",
        "nsi_description": description,
        "okpd2_code": "25.73.30.141",
        "category": "tools",
        "status": "ACTIVE",
        "is_template": False,
        "characteristics_json": json.dumps([]),
        "position_okeis_json": json.dumps([{"code": "796", "name": "Штука"}]),
        "parent_position_info_json": "null",
    }


def test_load_parquet_handles_missing_description(tmp_path):
    path = tmp_path / "ktru.parquet"
    pd.DataFrame([_row("1", None), _row("2", float("nan")), _row("3", "  Ударный инструмент  ")]).to_parquet(path)

    items = {it["code"]: it for it in load_parquet(path)}

    assert items["1"]["description"] == ""
    assert items["2"]["description"] == ""
    assert items["3"]["description"] == "Ударный инструмент"


@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="нет data/processed/ktru.parquet")
def test_load_real_ktru():
    items = load_parquet(DEFAULT_PATH)

    assert len(items) > 1000
    assert all(isinstance(it["description"], str) for it in items)
