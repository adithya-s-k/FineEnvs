"""Readable previews of workspace files: spreadsheets as tables, documents and slides as text blocks.

Everything is capped (rows, columns, characters) so a 40 MB workbook previews as fast as a tiny one.
PDFs, images and HTML are served raw instead and shown by the browser itself.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

ROWS, COLS, CHARS = 60, 24, 60_000


def _cell(v) -> str:
    if v is None:
        return ""
    s = str(v)
    return s if len(s) <= 200 else s[:199] + "…"


def spreadsheet(p: Path) -> dict:
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            rows = [r[:COLS] for _, r in zip(range(ROWS + 1), csv.reader(f))]
        return {"type": "sheets", "sheets": [{"name": p.name, "rows": [[_cell(c) for c in r] for r in rows]}]}
    import openpyxl

    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    sheets = []
    for ws in wb.worksheets[:12]:
        rows = []
        for r in ws.iter_rows(max_row=ROWS, max_col=COLS, values_only=True):
            rows.append([_cell(c) for c in r])
        while rows and not any(rows[-1]):
            rows.pop()
        sheets.append({"name": ws.title, "rows": rows, "dims": ws.max_row and f"{ws.max_row} × {ws.max_column}"})
    wb.close()
    return {"type": "sheets", "sheets": sheets}


def document(p: Path) -> dict:
    import docx

    d = docx.Document(p)
    blocks, n = [], 0
    body = d.element.body
    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = docx.text.paragraph.Paragraph(child, d)
            t = para.text.strip()
            if not t:
                continue
            style = (para.style.name or "").lower() if para.style is not None else ""
            blocks.append({"h": 1 if "heading 1" in style or style == "title" else 2 if "heading" in style else 0, "text": t})
            n += len(t)
        elif tag == "tbl":
            tbl = docx.table.Table(child, d)
            rows = [[_cell(c.text.strip()) for c in r.cells][:COLS] for r in tbl.rows[:ROWS]]
            blocks.append({"table": rows})
            n += sum(len(c) for r in rows for c in r)
        if n > CHARS:
            blocks.append({"h": 0, "text": "… (truncated)"})
            break
    return {"type": "document", "blocks": blocks}


def slides(p: Path) -> dict:
    from pptx import Presentation

    prs = Presentation(p)
    out = []
    for i, s in enumerate(prs.slides, 1):
        title, texts = "", []
        for sh in s.shapes:
            if not sh.has_text_frame:
                if getattr(sh, "has_table", False) and sh.has_table:
                    texts.append(" | ".join(c.text for c in sh.table.rows[0].cells))
                continue
            t = sh.text_frame.text.strip()
            if not t:
                continue
            if sh == getattr(s.shapes, "title", None) and not title:
                title = t
            else:
                texts.append(t)
        out.append({"n": i, "title": title, "text": texts[:30]})
    return {"type": "slides", "slides": out[:60]}


def pdf_text(p: Path) -> dict:
    from pypdf import PdfReader

    r = PdfReader(p)
    pages = []
    total = 0
    for i, page in enumerate(r.pages[:12], 1):
        t = (page.extract_text() or "").strip()
        pages.append({"n": i, "text": t[:6000]})
        total += len(t)
        if total > CHARS:
            break
    return {"type": "pdf", "pages": pages, "page_count": len(r.pages)}


def text(p: Path) -> dict:
    return {"type": "text", "text": p.read_text(encoding="utf-8", errors="replace")[:CHARS]}


def archive(p: Path) -> dict:
    with zipfile.ZipFile(p) as z:
        return {"type": "archive", "entries": [{"name": i.filename, "size": i.file_size} for i in z.infolist()[:300]]}


def preview(p: Path) -> dict:
    ext = p.suffix.lower().lstrip(".")
    try:
        if ext in ("xlsx", "xlsm", "csv"):
            return spreadsheet(p)
        if ext == "docx":
            return document(p)
        if ext == "pptx":
            return slides(p)
        if ext == "pdf":
            return pdf_text(p)
        if ext in ("md", "txt", "json", "xml", "yaml", "yml", "py", "sql", "log"):
            return text(p)
        if ext == "zip":
            return archive(p)
    except Exception as e:  # a malformed file should not break the page
        return {"type": "error", "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return {"type": "raw"}
