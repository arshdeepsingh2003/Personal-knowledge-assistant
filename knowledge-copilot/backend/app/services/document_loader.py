"""
document_loader.py — Table-aware document ingestion

Key change from Phase 2:
  PyPDFLoader is replaced with pdfplumber for PDF files.
  pdfplumber preserves table structure, cell boundaries, and row data.
  PyPDFLoader strips tables to a flat string, losing all structure.

For non-PDF formats (txt, md, web) the logic is identical to Phase 2.
"""
import os
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.core.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


# ── PDF loading with pdfplumber ───────────────────────────────────────────────

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF using pdfplumber.

    For each page this extracts:
      - Regular text (paragraphs, headings) as a prose string
      - Tables as structured Document objects with table_data metadata

    Tables get their own Document objects so the chunker can keep
    them intact and convert rows to natural language separately.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for table-aware PDF loading.\n"
            "Install it: pip install pdfplumber"
        )

    docs: List[Document] = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # ── Extract tables first ──────────────────────────────────────────
            tables = page.extract_tables()
            table_bboxes = []

            for table_idx, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue  # skip empty or header-only tables

                # Get table bounding box to exclude from prose extraction
                table_settings = {"vertical_strategy": "lines",
                                   "horizontal_strategy": "lines"}
                try:
                    tbl_obj = page.find_tables(table_settings)
                    if tbl_obj and table_idx < len(tbl_obj):
                        table_bboxes.append(tbl_obj[table_idx].bbox)
                except Exception:
                    pass  # bbox extraction is best-effort

                # Store table as separate Document with full row data
                docs.append(Document(
                    page_content=_table_to_text(table),
                    metadata={
                        "source":      file_path,
                        "file_name":   Path(file_path).name,
                        "file_type":   ".pdf",
                        "page":        page_num,
                        "content_type": "table",
                        "table_index": table_idx,
                        "table_data":  table,  # raw rows for NL conversion
                    }
                ))

            # ── Extract prose text (excluding table regions) ──────────────────
            # crop away table regions so prose and tables don't overlap
            prose_page = page
            for bbox in table_bboxes:
                try:
                    # Crop out the table area from the prose extraction
                    prose_page = prose_page.outside_bbox(bbox)
                except Exception:
                    pass

            text = prose_page.extract_text(x_tolerance=3, y_tolerance=3)
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source":       file_path,
                        "file_name":    Path(file_path).name,
                        "file_type":    ".pdf",
                        "page":         page_num,
                        "content_type": "prose",
                    }
                ))

    return docs


def _table_to_text(table: List[List]) -> str:
    """
    Convert a raw pdfplumber table (list of lists) to a readable text block.
    This is stored as the document content for embedding.

    Example input:
      [["Industry", "ROI", "Payback"],
       ["Retail", "312%", "3.8 months"]]

    Example output:
      Industry | ROI | Payback
      Retail | 312% | 3.8 months
    """
    lines = []
    for row in table:
        clean_row = [str(cell).strip() if cell else "" for cell in row]
        lines.append(" | ".join(clean_row))
    return "\n".join(lines)


# ── Text, Markdown, Web loaders (unchanged from Phase 2) ─────────────────────

def load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def load_web_url(url: str) -> List[Document]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return [Document(
        page_content=text,
        metadata={"source": url, "type": "web", "content_type": "prose"}
    )]


# ── Router ────────────────────────────────────────────────────────────────────

def load_document(file_path: str) -> List[Document]:
    """Route to the right loader based on file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        docs = load_pdf(file_path)
    elif ext == ".txt":
        docs = load_text(file_path)
    elif ext in (".md", ".markdown"):
        docs = load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Attach file_name and file_type to every doc that doesn't already have it
    for doc in docs:
        doc.metadata.setdefault("file_name", Path(file_path).name)
        doc.metadata.setdefault("file_type", ext)

    return docs


def save_upload_and_load(file_bytes: bytes, filename: str) -> List[Document]:
    """Save an uploaded file to disk and load it."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / filename
    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    return load_document(str(dest_path))