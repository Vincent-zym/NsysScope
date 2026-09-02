#!/usr/bin/env python3
"""Convert every CSV in a directory to a small, dependency-free XLSX workbook."""

from __future__ import annotations

import argparse
import csv
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape


def column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def cell_xml(reference: str, value: str, header: bool = False) -> str:
    style = ' s="1"' if header else ""
    clean = value.replace("\x00", "")[:32767]
    if not header:
        try:
            number = float(clean.rstrip("%"))
            if math.isfinite(number) and re.fullmatch(r"[+-]?\d+(?:\.\d+)?", clean):
                return f'<c r="{reference}"{style}><v>{clean}</v></c>'
        except ValueError:
            pass
    return (
        f'<c r="{reference}" t="inlineStr"{style}><is><t xml:space="preserve">'
        f"{escape(clean)}</t></is></c>"
    )


def worksheet_xml(rows: list[list[str]]) -> str:
    widths = []
    for index in range(max((len(row) for row in rows), default=0)):
        widths.append(min(60, max(10, max(
            (len(row[index]) if index < len(row) else 0 for row in rows[:500]),
            default=10,
        ) + 2)))
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, 1)
    )
    xml_rows = []
    for row_index, row in enumerate(rows, 1):
        cells = "".join(
            cell_xml(f"{column_name(column_index)}{row_index}", value, row_index == 1)
            for column_index, value in enumerate(row, 1)
        )
        xml_rows.append(f'<row r="{row_index}">{cells}</row>')
    last_column = column_name(len(widths)) or "A"
    last_row = max(1, len(rows))
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetViews><sheetView workbookViewId="0">'
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
        '</sheetView></sheetViews>'
        f"<cols>{columns}</cols><sheetData>{''.join(xml_rows)}</sheetData>"
        f'<autoFilter ref="A1:{last_column}{last_row}"/>'
        "</worksheet>"
    )


def write_workbook(csv_path: Path, output: Path) -> None:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [[value or "" for value in row] for row in csv.reader(handle)]
    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="analysis" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            "</Relationships>"
        ),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font><sz val="10"/><name val="Arial"/></font>'
            '<font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Arial"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FF27354F"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/></cellXfs>'
            "</styleSheet>"
        ),
        "xl/worksheets/sheet1.xml": worksheet_xml(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_dir", type=Path)
    parser.add_argument("xlsx_dir", type=Path)
    args = parser.parse_args()
    csv_files = sorted(args.csv_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"no CSV files found in {args.csv_dir}")
    for csv_path in csv_files:
        write_workbook(csv_path, args.xlsx_dir / f"{csv_path.stem}.xlsx")


if __name__ == "__main__":
    main()
