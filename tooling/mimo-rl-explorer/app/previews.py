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
    # Formulas the file never had cached values for show as blanks in data_only mode; show the formula instead.
    wf = openpyxl.load_workbook(p, read_only=True, data_only=False)
    sheets = []
    for ws, wsf in zip(wb.worksheets[:12], wf.worksheets[:12]):
        rows = []
        for r, rf in zip(ws.iter_rows(max_row=ROWS, max_col=COLS, values_only=True),
                         wsf.iter_rows(max_row=ROWS, max_col=COLS, values_only=True)):
            rows.append([_cell(c if c is not None else (f if isinstance(f, str) and f.startswith("=") else None))
                         for c, f in zip(r, rf)])
        while rows and not any(rows[-1]):
            rows.pop()
        sheets.append({"name": ws.title, "rows": rows, "dims": ws.max_row and f"{ws.max_row} × {ws.max_column}"})
    wb.close()
    wf.close()
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
    """Each slide as a title plus its text, in reading order. The title is the title placeholder, or else the
    largest text on the slide. Text repeated on most slides (logos, confidentiality footers) is pulled out and
    shown once, and bare page labels ("01") are dropped, so each card shows what is different about that slide."""
    from collections import Counter

    from pptx import Presentation

    prs = Presentation(p)
    raw = []
    for i, s in enumerate(prs.slides, 1):
        tshape = s.shapes.title
        tid = tshape.shape_id if tshape is not None else None
        items, tables = [], []
        for sh in s.shapes:
            if getattr(sh, "has_table", False) and sh.has_table:
                tables.append([[c.text.strip() for c in r.cells] for r in list(sh.table.rows)[:12]])
                continue
            if not sh.has_text_frame:
                continue
            t = sh.text_frame.text.strip()
            if not t:
                continue
            sizes = [r.font.size.pt for para in sh.text_frame.paragraphs for r in para.runs if r.font.size]
            items.append({"t": t, "size": max(sizes) if sizes else 0, "top": sh.top or 0, "left": sh.left or 0,
                          "title": sh.shape_id == tid})
        items.sort(key=lambda x: (x["top"], x["left"]))
        raw.append((i, items, tables))
    counts = Counter(x["t"] for _, items, _ in raw for x in {x["t"]: x for x in items}.values())
    common = {t for t, c in counts.items() if len(raw) >= 3 and c >= max(3, 0.4 * len(raw))}
    label = lambda t: len(t) <= 3 and (t.isdigit() or t.isupper())   # "01", "ADC"
    out = []
    for i, items, tables in raw:
        keep = [x for x in items if x["t"] not in common and not label(x["t"])]
        pick = next((x for x in keep if x["title"]), None) or max(
            (x for x in keep if len(x["t"]) >= 4), key=lambda x: (x["size"], -x["top"]), default=None)
        title = pick["t"].split("\n")[0] if pick else ""
        texts = [x["t"] for x in keep if x is not pick]
        if pick and "\n" in pick["t"]:
            texts.insert(0, pick["t"].split("\n", 1)[1])
        out.append({"n": i, "title": title, "text": texts[:30], "tables": tables})
    return {"type": "slides", "slides": out[:60], "repeated": sorted(common, key=len)[:6]}


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
