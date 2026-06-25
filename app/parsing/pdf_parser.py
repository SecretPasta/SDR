from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

from app.domain.section import Location, ParsedDoc, Section, TableData

_HEADING_SCALE = 1.15
_HEADING_MAX_LEN = 100
_HEADING_NUM_RE = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.+)', re.DOTALL)
_BULLET_RE = re.compile(r'^[•\-*◦‣▪▸–]\s+')
_BOLD_FLAG = 1 << 4  # PyMuPDF span flags bit 4


# ── public entry point ────────────────────────────────────────────────────────

def parse_pdf(path: Path, doc_id: str) -> ParsedDoc:
    doc = fitz.open(str(path))
    try:
        median_size = _median_body_size(doc)
        sections = _extract_sections(doc, path, doc_id, median_size)
    finally:
        doc.close()
    return ParsedDoc(doc_id=doc_id, filename=path.name, sections=sections)


# ── section extraction ────────────────────────────────────────────────────────

@dataclass
class _Acc:
    """Mutable accumulator for one section being built."""
    heading: str
    heading_num: str | None
    heading_path: list[str]
    page_num: int
    idx: int
    body_lines: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)

    def to_section(self, doc_id: str, filename: str) -> Section:
        section_id = (
            f"{doc_id}::{self.heading_num}"
            if self.heading_num
            else f"{doc_id}::{self.idx}-{_slugify(self.heading)}"
        )
        return Section(
            id=section_id,
            location=Location(
                filename=filename,
                page_number=self.page_num,
                heading_number=self.heading_num,
                heading_path=list(self.heading_path),
            ),
            heading=self.heading,
            body_text="\n".join(self.body_lines).strip(),
            tables=list(self.tables),
            bullets=list(self.bullets),
        )


def _extract_sections(
    doc: fitz.Document,
    path: Path,
    doc_id: str,
    median_size: float,
) -> list[Section]:
    threshold = median_size * _HEADING_SCALE
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (depth, full_heading_text)
    acc: _Acc | None = None
    idx = 0

    for i in range(len(doc)):
        page_num = i + 1
        page: fitz.Page = doc[i]
        table_bboxes = [t.bbox for t in page.find_tables().tables]
        page_tables = _extract_page_tables(page, path, page_num)

        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            if _in_table(block["bbox"], table_bboxes):
                continue

            for line in block["lines"]:
                text, size, bold = _line_attrs(line)
                stripped = text.strip()
                if not stripped:
                    continue

                if _is_heading(stripped, size, bold, threshold):
                    if acc is not None:
                        sections.append(acc.to_section(doc_id, path.name))
                    heading_num, heading_clean = _parse_heading_num(stripped)
                    depth = _heading_depth(heading_num)
                    stack = [(d, t) for d, t in stack if d < depth]
                    stack.append((depth, stripped))
                    acc = _Acc(
                        heading=heading_clean,
                        heading_num=heading_num,
                        heading_path=[t for _, t in stack],
                        page_num=page_num,
                        idx=idx,
                    )
                    idx += 1
                elif acc is not None:
                    if _BULLET_RE.match(stripped):
                        acc.bullets.append(_BULLET_RE.sub("", stripped).strip())
                    else:
                        acc.body_lines.append(stripped)

        # Tables on this page belong to whichever section is open at page end.
        # Sections spanning multiple pages accumulate tables page-by-page.
        if acc is not None:
            acc.tables.extend(page_tables)

    if acc is not None:
        sections.append(acc.to_section(doc_id, path.name))

    return sections


# ── table extraction ──────────────────────────────────────────────────────────

def _extract_page_tables(
    page: fitz.Page,
    path: Path,
    page_num: int,
) -> list[TableData]:
    fitz_tabs = page.find_tables().tables
    if not fitz_tabs:
        return []

    results: list[TableData] = []
    for tab in fitz_tabs:
        rows = tab.extract()
        if not rows or not _table_valid(rows):
            # Any broken table on the page triggers a full-page pdfplumber fallback
            return _pdfplumber_tables(path, page_num)
        headers = [str(c) if c is not None else "" for c in rows[0]]
        body = [[str(c) if c is not None else "" for c in row] for row in rows[1:]]
        results.append(TableData(headers=headers, rows=body))

    return results


def _table_valid(rows: list[list[str | None]]) -> bool:
    if not rows:
        return False
    col_count = len(rows[0])
    if col_count == 0:
        return False
    if any(len(row) != col_count for row in rows):
        return False
    empty = sum(1 for row in rows if all((c or "").strip() == "" for c in row))
    return empty / len(rows) < 0.3


def _pdfplumber_tables(path: Path, page_num: int) -> list[TableData]:
    tables: list[TableData] = []
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[page_num - 1]
        for raw in page.extract_tables() or []:
            if not raw:
                continue
            headers = [str(c or "") for c in raw[0]]
            rows = [[str(c or "") for c in row] for row in raw[1:]]
            tables.append(TableData(headers=headers, rows=rows))
    return tables


# ── text analysis helpers ─────────────────────────────────────────────────────

def _median_body_size(doc: fitz.Document) -> float:
    sizes: list[float] = []
    for i in range(len(doc)):
        page: fitz.Page = doc[i]
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["text"].strip():
                        sizes.append(float(span["size"]))
    return statistics.median(sizes) if sizes else 12.0


def _line_attrs(line: dict) -> tuple[str, float, bool]:
    spans = line.get("spans", [])
    if not spans:
        return "", 0.0, False
    text = "".join(s["text"] for s in spans)
    size = max(float(s["size"]) for s in spans)
    bold = any(
        bool(s.get("flags", 0) & _BOLD_FLAG) or "bold" in s.get("font", "").lower()
        for s in spans
        if s["text"].strip()
    )
    return text, size, bold


def _is_heading(text: str, size: float, bold: bool, threshold: float) -> bool:
    return size >= threshold and bold and len(text) < _HEADING_MAX_LEN


def _parse_heading_num(text: str) -> tuple[str | None, str]:
    m = _HEADING_NUM_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()
    return None, text


def _heading_depth(heading_num: str | None) -> int:
    if heading_num:
        return heading_num.count(".") + 1
    return 1


def _in_table(bbox: tuple, table_bboxes: list, threshold: float = 0.5) -> bool:
    bx0, by0, bx1, by1 = (float(v) for v in bbox)
    block_area = (bx1 - bx0) * (by1 - by0)
    if block_area <= 0:
        return False
    for tb in table_bboxes:
        tx0, ty0, tx1, ty1 = (float(v) for v in tb)
        ix0, iy0 = max(bx0, tx0), max(by0, ty0)
        ix1, iy1 = min(bx1, tx1), min(by1, ty1)
        if ix0 < ix1 and iy0 < iy1 and (ix1 - ix0) * (iy1 - iy0) / block_area >= threshold:
            return True
    return False


def _slugify(text: str) -> str:
    return re.sub(r'\W+', '-', text.lower()).strip('-')[:40]