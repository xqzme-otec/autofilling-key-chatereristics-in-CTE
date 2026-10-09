from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET


@dataclass
class SheetSummary:
    name: str
    rows: int
    columns: List[str]
    examples: List[Dict[str, str]]
    null_rate: Dict[str, float]


NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkgrel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def inspect_xlsx(path: Path, sample_size: int = 5) -> List[SheetSummary]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = _read_shared_strings(archive)
        sheets = _read_sheet_paths(archive)
        return [_summarize_sheet(archive, sheet_name, sheet_path, shared_strings, sample_size) for sheet_name, sheet_path in sheets.items()]


def _read_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("main:si", NS):
        texts = [node.text or "" for node in item.findall(".//main:t", NS)]
        values.append("".join(texts))
    return values


def _read_sheet_paths(archive: zipfile.ZipFile) -> Dict[str, str]:
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {
        rel.attrib["Id"]: "xl/" + rel.attrib["Target"].lstrip("/")
        for rel in rels.findall("pkgrel:Relationship", NS)
    }
    sheets: Dict[str, str] = {}
    for sheet in workbook.findall(".//main:sheet", NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib[f"{{{NS['rel']}}}id"]
        sheets[name] = rel_targets[rel_id]
    return sheets


def _summarize_sheet(
    archive: zipfile.ZipFile,
    sheet_name: str,
    sheet_path: str,
    shared_strings: List[str],
    sample_size: int,
) -> SheetSummary:
    root = ET.fromstring(archive.read(sheet_path))
    rows = root.findall(".//main:sheetData/main:row", NS)
    parsed_rows = [_parse_row(row, shared_strings) for row in rows]
    headers = _headers(parsed_rows)
    data_rows = [_row_to_dict(row, headers) for row in parsed_rows[1:]]
    return SheetSummary(
        name=sheet_name,
        rows=len(data_rows),
        columns=headers,
        examples=data_rows[:sample_size],
        null_rate=_null_rates(data_rows, headers),
    )


def _parse_row(row: ET.Element, shared_strings: List[str]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for cell in row.findall("main:c", NS):
        ref = cell.attrib.get("r", "")
        index = _column_index(ref)
        if index is None:
            continue
        value = _cell_value(cell, shared_strings)
        if value is not None:
            result[index] = value
    return result


def _cell_value(cell: ET.Element, shared_strings: List[str]) -> Optional[str]:
    value_node = cell.find("main:v", NS)
    inline_node = cell.find("main:is/main:t", NS)
    if inline_node is not None:
        return inline_node.text or ""
    if value_node is None:
        return None
    value = value_node.text or ""
    if cell.attrib.get("t") == "s":
        return shared_strings[int(value)]
    return value


def _headers(rows: List[Dict[int, str]]) -> List[str]:
    if not rows:
        return []
    max_index = max(rows[0].keys(), default=-1)
    return [rows[0].get(index, f"column_{index + 1}") for index in range(max_index + 1)]


def _row_to_dict(row: Dict[int, str], headers: List[str]) -> Dict[str, str]:
    return {header: row.get(index, "") for index, header in enumerate(headers)}


def _null_rates(rows: List[Dict[str, str]], headers: List[str]) -> Dict[str, float]:
    if not rows:
        return {header: 1.0 for header in headers}
    return {
        header: sum(1 for row in rows if not row.get(header)) / len(rows)
        for header in headers
    }


def _column_index(ref: str) -> Optional[int]:
    match = re.match(r"([A-Z]+)", ref)
    if not match:
        return None
    index = 0
    for char in match.group(1):
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1

