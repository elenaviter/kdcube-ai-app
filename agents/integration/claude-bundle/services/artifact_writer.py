# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Elena Viter
"""Create the demo PDF and XLSX with the Python standard library only."""

from __future__ import annotations

import argparse
import json
import textwrap
import zipfile
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Iterable, Mapping


PDF_NAME = "agent-harness-research.pdf"
XLSX_NAME = "agent-harness-research.xlsx"


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(
    commands: list[str],
    *,
    x: float,
    y: float,
    size: float,
    value: str,
    bold: bool = False,
    color: tuple[float, float, float] = (0.08, 0.15, 0.22),
) -> None:
    commands.append(
        f"BT /{'F2' if bold else 'F1'} {size:g} Tf "
        f"{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} rg "
        f"1 0 0 1 {x:g} {y:g} Tm ({_pdf_escape(value)}) Tj ET"
    )


def _pdf_bytes(data: Mapping[str, Any]) -> bytes:
    title = str(data.get("title") or "Python release research")
    version = str(data.get("version") or "Unknown")
    release_date = str(data.get("release_date") or "Unknown")
    highlights = [str(item) for item in data.get("highlights") or []]
    sources = [item for item in data.get("sources") or [] if isinstance(item, Mapping)]

    commands = [
        "0.965 0.976 0.984 rg 0 0 612 792 re f",
        "0.000 0.620 0.580 rg 0 748 612 44 re f",
        "0.345 0.255 0.875 rg 0 742 612 6 re f",
    ]
    _pdf_text(commands, x=42, y=765, size=10, value="KDCUBE AGENT HARNESS DEMONSTRATION", bold=True, color=(1, 1, 1))
    _pdf_text(commands, x=42, y=706, size=23, value=title, bold=True)
    _pdf_text(commands, x=42, y=674, size=11, value=f"Stable release: {version}", bold=True, color=(0.0, 0.45, 0.42))
    _pdf_text(commands, x=300, y=674, size=11, value=f"Released: {release_date}", color=(0.30, 0.36, 0.42))
    commands.append("0.840 0.870 0.900 RG 42 656 m 570 656 l S")

    y = 626.0
    _pdf_text(commands, x=42, y=y, size=13, value="Release highlights", bold=True)
    y -= 25
    for item in highlights[:6]:
        wrapped = textwrap.wrap(item, width=82) or [""]
        _pdf_text(commands, x=48, y=y, size=10, value="-", bold=True, color=(0.345, 0.255, 0.875))
        for index, line in enumerate(wrapped):
            _pdf_text(commands, x=62, y=y, size=10, value=line)
            if index < len(wrapped) - 1:
                y -= 14
        y -= 20

    y = max(y - 8, 250)
    _pdf_text(commands, x=42, y=y, size=13, value="Primary sources", bold=True)
    y -= 24
    link_specs: list[tuple[str, float]] = []
    for source in sources[:8]:
        label = str(source.get("label") or "Source")
        url = str(source.get("url") or "")
        _pdf_text(commands, x=48, y=y, size=9, value=label, bold=True)
        y -= 14
        if url:
            _pdf_text(commands, x=48, y=y, size=9, value=url, color=(0.0, 0.36, 0.70))
            link_specs.append((url, y))
        y -= 22
        if y < 82:
            break

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _pdf_text(commands, x=42, y=34, size=8, value=f"Generated through the KDCube conversation workspace - {generated}", color=(0.38, 0.43, 0.48))
    stream = ("\n".join(commands) + "\n").encode("latin-1", errors="replace")

    annotation_ids = list(range(7, 7 + len(link_specs)))
    annots = " ".join(f"{number} 0 R" for number in annotation_ids)
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R /F2 5 0 R >> >> "
            f"/Contents 6 0 R /Annots [{annots}] >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"endstream",
    ]
    for url, link_y in link_specs:
        width = min(510, max(80, len(url) * 4.7))
        objects.append(
            (
                "<< /Type /Annot /Subtype /Link "
                f"/Rect [48 {link_y - 2:g} {48 + width:g} {link_y + 10:g}] "
                "/Border [0 0 0] /A << /S /URI "
                f"/URI ({_pdf_escape(url)}) >> >>"
            ).encode("latin-1", errors="replace")
        )

    output = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode("ascii"))
        output.extend(body)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _cell(row: int, column: int, value: Any, style: int = 0) -> str:
    ref = f"{_column_name(column)}{row}"
    text = escape(str(value), quote=False)
    return f'<c r="{ref}" t="inlineStr" s="{style}"><is><t>{text}</t></is></c>'


def _sheet(
    rows: Iterable[Iterable[tuple[Any, int]]],
    *,
    widths: tuple[int, ...],
    freeze: bool = True,
    hyperlink_rows: tuple[int, ...] = (),
) -> str:
    rendered_rows = []
    for row_index, values in enumerate(rows, start=1):
        cells = "".join(
            _cell(row_index, column_index, value, style)
            for column_index, (value, style) in enumerate(values, start=1)
        )
        rendered_rows.append(f'<row r="{row_index}" ht="22" customHeight="1">{cells}</row>')
    columns = "".join(
        f'<col min="{index}" max="{index}" width="{width}" customWidth="1"/>'
        for index, width in enumerate(widths, start=1)
    )
    pane = '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>' if freeze else ""
    hyperlinks = ""
    if hyperlink_rows:
        links = "".join(
            f'<hyperlink ref="B{row}" r:id="rId{index}"/>'
            for index, row in enumerate(hyperlink_rows, start=1)
        )
        hyperlinks = f"<hyperlinks>{links}</hyperlinks>"
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"{pane}<cols>{columns}</cols><sheetData>{''.join(rendered_rows)}</sheetData>{hyperlinks}"
        '<autoFilter ref="A1:B1"/></worksheet>'
    )


def _xlsx_bytes(data: Mapping[str, Any]) -> bytes:
    title = str(data.get("title") or "Python release research")
    version = str(data.get("version") or "Unknown")
    release_date = str(data.get("release_date") or "Unknown")
    highlights = [str(item) for item in data.get("highlights") or []]
    sources = [item for item in data.get("sources") or [] if isinstance(item, Mapping)]

    summary_rows: list[list[tuple[Any, int]]] = [
        [("Field", 1), ("Value", 1)],
        [("Report", 2), (title, 0)],
        [("Stable version", 2), (version, 3)],
        [("Release date", 2), (release_date, 0)],
    ]
    for index, item in enumerate(highlights, start=1):
        summary_rows.append([(f"Highlight {index}", 2), (item, 0)])
    source_rows: list[list[tuple[Any, int]]] = [[("Source", 1), ("URL", 1)]]
    for source in sources:
        source_rows.append(
            [
                (str(source.get("label") or "Source"), 2),
                (str(source.get("url") or ""), 4),
            ]
        )

    files = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
            '</Types>'
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
            '</Relationships>'
        ),
        "docProps/core.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/">'
            '<dc:title>Agent Harness Research</dc:title><dc:creator>KDCube Agent Harness</dc:creator>'
            '</cp:coreProperties>'
        ),
        "xl/workbook.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Summary" sheetId="1" r:id="rId1"/>'
            '<sheet name="Sources" sheetId="2" r:id="rId2"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>'
            '<Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>'
        ),
        "xl/styles.xml": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="4"><font><sz val="11"/><name val="Aptos"/></font>'
            '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Aptos"/></font>'
            '<font><b/><color rgb="FF006F68"/><sz val="11"/><name val="Aptos"/></font>'
            '<font><u/><color rgb="FF0563C1"/><sz val="11"/><name val="Aptos"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FF17324D"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="5"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
            '<xf numFmtId="0" fontId="2" fillId="0" borderId="0" xfId="0" applyFont="1"/>'
            '<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"/>'
            '<xf numFmtId="0" fontId="3" fillId="0" borderId="0" xfId="0" applyFont="1"><alignment wrapText="1"/></xf>'
            '</cellXfs><cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>'
            '<dxfs count="0"/><tableStyles count="0" defaultTableStyle="TableStyleMedium2" defaultPivotStyle="PivotStyleLight16"/>'
            '</styleSheet>'
        ),
        "xl/worksheets/sheet1.xml": _sheet(summary_rows, widths=(22, 92)),
        "xl/worksheets/sheet2.xml": _sheet(
            source_rows,
            widths=(36, 94),
            hyperlink_rows=tuple(range(2, len(source_rows) + 1)),
        ),
        "xl/worksheets/_rels/sheet2.xml.rels": (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                '<Relationship '
                f'Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" '
                f'Target="{escape(str(source.get("url") or ""), quote=True)}" TargetMode="External"/>'
                for index, source in enumerate(sources, start=1)
            )
            + '</Relationships>'
        ),
    }
    from io import BytesIO

    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return target.getvalue()


def load_research(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("research input must be a JSON object")
    required = ("version", "release_date", "highlights", "sources")
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise ValueError("research input is missing: " + ", ".join(missing))
    if not isinstance(data["highlights"], list) or not isinstance(data["sources"], list):
        raise ValueError("highlights and sources must be arrays")
    return data


def write_artifacts(data: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / PDF_NAME
    xlsx_path = output_dir / XLSX_NAME
    pdf_path.write_bytes(_pdf_bytes(data))
    xlsx_path.write_bytes(_xlsx_bytes(data))
    return pdf_path, xlsx_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    paths = write_artifacts(load_research(args.input), args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
