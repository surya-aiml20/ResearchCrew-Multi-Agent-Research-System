"""PDF -> page text (PyMuPDF) -> overlapping, page-aware chunks."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    id: str
    text: str
    source: str       # file name
    page: int         # 1-based
    doc_id: str


def load_pdf(source: str | Path | bytes, name: str | None = None) -> tuple[str, str, list[tuple[int, str]]]:
    """Return (doc_id, file_name, [(page_no, text), ...]). Accepts a path or raw bytes."""
    try:
        import pymupdf as fitz  # PyMuPDF >= 1.24.3
    except ImportError:  # older releases
        import fitz

    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        name = name or path.name
    else:
        data = source
        name = name or "uploaded.pdf"
    doc_id = hashlib.sha1(data).hexdigest()[:12]
    pages: list[tuple[int, str]] = []
    with fitz.open(stream=data, filetype="pdf") as pdf:
        for i, page in enumerate(pdf, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            if text:
                pages.append((i, text))
    return doc_id, name, pages


def _split(text: str, size: int, overlap: int) -> list[str]:
    """Greedy window split that prefers sentence / word boundaries."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("? "), window.rfind("! "))
            if cut < size * 0.5:
                cut = window.rfind(" ")
            if cut > 0:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return out


def chunk_pages(doc_id: str, name: str, pages: list[tuple[int, str]],
                chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page_no, text in pages:
        for j, piece in enumerate(_split(text, chunk_size, overlap)):
            if len(piece) < 40:  # skip headers / page numbers
                continue
            chunks.append(Chunk(id=f"{doc_id}:{page_no}:{j}", text=piece,
                                source=name, page=page_no, doc_id=doc_id))
    return chunks
