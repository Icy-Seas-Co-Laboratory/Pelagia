"""Small dependency-free XLSX writer with Excel-safe worksheet rollover.

This intentionally writes inline strings and native numbers/booleans.  It is a
product writer, not a generic database export mechanism.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Mapping, Sequence
from xml.sax.saxutils import escape as xml_escape

MAX_XLSX_ROWS = 1_048_576


def write_xlsx(sheets: Mapping[str, Iterable[Mapping[str, Any]]]) -> bytes:
    """Return a portable workbook, splitting oversized logical tables by sheet."""
    prepared: list[tuple[str, list[Mapping[str, Any]]]] = []
    max_data_rows = MAX_XLSX_ROWS - 1  # Reserve row one for a header.
    for name, source_rows in sheets.items():
        rows = list(source_rows)
        if not rows:
            prepared.append((name, rows))
            continue
        for part, start in enumerate(range(0, len(rows), max_data_rows), 1):
            sheet_name = str(name) if len(rows) <= max_data_rows else f"{name}_{part:03d}"
            prepared.append((sheet_name, rows[start:start + max_data_rows]))
    if not prepared:
        prepared = [("Data", [])]
    names = _sheet_names([name for name, _ in prepared])
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types(len(prepared)))
        archive.writestr("_rels/.rels", _root_rels())
        archive.writestr("xl/workbook.xml", _workbook(names))
        archive.writestr("xl/_rels/workbook.xml.rels", _workbook_rels(len(prepared)))
        archive.writestr("xl/styles.xml", _styles())
        for index, (_, rows) in enumerate(prepared, 1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet(rows))
    return output.getvalue()


def _worksheet(rows: list[Mapping[str, Any]]) -> str:
    columns = _columns(rows)
    if len(rows) + (1 if columns else 0) > MAX_XLSX_ROWS:
        raise ValueError("Internal error: XLSX sheet partition exceeds Excel's row limit.")
    rendered: list[str] = []
    if columns:
        rendered.append(_row(1, columns, header=True))
        rendered.extend(_row(index, [record.get(column) for column in columns]) for index, record in enumerate(rows, 2))
    dimension = "A1" if not columns else f"A1:{_column_name(len(columns))}{len(rows) + 1}"
    auto_filter = "" if not columns else f'<autoFilter ref="{dimension}"/>'
    pane = "" if not columns else '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f'<dimension ref="{dimension}"/>{pane}<sheetData>{"".join(rendered)}</sheetData>{auto_filter}</worksheet>')


def _row(number: int, values: Sequence[Any], *, header: bool = False) -> str:
    return f'<row r="{number}">' + "".join(_cell(f"{_column_name(i)}{number}", value, header) for i, value in enumerate(values, 1)) + "</row>"


def _cell(reference: str, value: Any, header: bool) -> str:
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{int(value)}</v></c>'
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return f'<c r="{reference}"><v>{value}</v></c>'
    if isinstance(value, (dict, list, tuple, set)):
        value = json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (datetime, date)):
        value = value.isoformat()
    text = _clean_text(str(value))
    # Formula-like content is data in scientific exports, never a formula.
    if not header and text[:1] in {"=", "+", "-", "@"}:
        text = "'" + text
    return f'<c r="{reference}" t="inlineStr"><is><t>{xml_escape(text)}</t></is></c>'


def _clean_text(value: str) -> str:
    return "".join(char for char in value if ord(char) in (9, 10, 13) or ord(char) >= 32)


def _columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                found.append(key); seen.add(key)
    return found


def _sheet_names(names: Sequence[str]) -> list[str]:
    result: list[str] = []; seen: set[str] = set()
    for index, raw in enumerate(names, 1):
        base = re.sub(r"[\[\]:*?/\\]+", "_", str(raw))[:31] or f"Sheet{index}"
        candidate = base; suffix = 1
        while candidate.casefold() in seen:
            suffix += 1; candidate = f"{base[:31-len(str(suffix))-1]}_{suffix}"
        seen.add(candidate.casefold()); result.append(candidate)
    return result


def _column_name(index: int) -> str:
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26); result = chr(65 + remainder) + result
    return result


def _content_types(count: int) -> str:
    worksheets = "".join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>' for i in range(1, count + 1))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' + worksheets + '</Types>')


def _root_rels() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'


def _workbook(names: Sequence[str]) -> str:
    sheets = "".join(f'<sheet name="{xml_escape(name)}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(names, 1))
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>' + sheets + '</sheets></workbook>'


def _workbook_rels(count: int) -> str:
    relationships = "".join(f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>' for i in range(1, count + 1))
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">' + relationships + f'<Relationship Id="rId{count+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'


def _styles() -> str:
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?><styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts><fills count="1"><fill><patternFill patternType="none"/></fill></fills><borders count="1"><border/></borders><cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs><cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs></styleSheet>'
