from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "services" / "artifact_writer.py"
_SPEC = importlib.util.spec_from_file_location("harness_claude_artifact_writer", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PDF_NAME = _MODULE.PDF_NAME
XLSX_NAME = _MODULE.XLSX_NAME
load_research = _MODULE.load_research
write_artifacts = _MODULE.write_artifacts


def _research() -> dict:
    return {
        "title": "Python release research",
        "version": "3.14.0",
        "release_date": "2026-01-01",
        "highlights": ["A clear release highlight."],
        "sources": [{"label": "Python", "url": "https://www.python.org/"}],
    }


def test_writer_creates_readable_pdf_and_workbook(tmp_path: Path) -> None:
    pdf_path, xlsx_path = write_artifacts(_research(), tmp_path)

    assert pdf_path.name == PDF_NAME
    assert pdf_path.read_bytes().startswith(b"%PDF-1.7")
    assert b"https://www.python.org/" in pdf_path.read_bytes()
    assert xlsx_path.name == XLSX_NAME
    with zipfile.ZipFile(xlsx_path) as workbook:
        names = set(workbook.namelist())
        assert "xl/worksheets/sheet1.xml" in names
        assert "xl/worksheets/sheet2.xml" in names
        assert "xl/worksheets/_rels/sheet2.xml.rels" in names
        assert b"Stable version" in workbook.read("xl/worksheets/sheet1.xml")
        assert b"https://www.python.org/" in workbook.read("xl/worksheets/sheet2.xml")
        assert b"TargetMode=\"External\"" in workbook.read(
            "xl/worksheets/_rels/sheet2.xml.rels"
        )


def test_load_research_rejects_incomplete_input(tmp_path: Path) -> None:
    path = tmp_path / "research.json"
    path.write_text(json.dumps({"version": "3.14"}), encoding="utf-8")

    try:
        load_research(path)
    except ValueError as exc:
        assert "research input is missing" in str(exc)
    else:
        raise AssertionError("incomplete research must be rejected")
