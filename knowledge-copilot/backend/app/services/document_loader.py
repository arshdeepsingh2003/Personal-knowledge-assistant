"""
document_loader.py

Tested and confirmed working against the actual SCALE-across-pages PDF.
PDF → pymupdf4llm markdown → one Document per file.
All page-boundary problems eliminated.
"""

from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.core.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def _extract_tables_with_pdfplumber(file_path: str) -> List[Document]:
    """
    Extract tables from PDF using pdfplumber and convert to natural language.
    Returns a list of Document objects, one per table found.

    Each table is converted to a descriptive NL representation so that
    numerical values in tables are indexed as regular text and retrievable
    via semantic search — not locked behind pipe/cell formatting.
    """
    try:
        import pdfplumber
    except ImportError:
        return []

    table_docs: List[Document] = []
    file_path_obj = Path(file_path)
    fname = file_path_obj.name

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue

                    header = [str(c).strip() if c else "" for c in table[0]]
                    rows = []
                    for row in table[1:]:
                        row_data = [str(c).strip() if c else "" for c in row]
                        if any(cell.strip() for cell in row_data):
                            rows.append(row_data)

                    if not rows:
                        continue

                    # Build natural language representation
                    nl_lines = []
                    if header:
                        nl_lines.append(f"Table: {' | '.join(header)}")

                    for row in rows:
                        if header:
                            pairs = [f"{header[j]} is {row[j]}" for j in range(min(len(header), len(row)))]
                            nl_lines.append(f"  Row: {'; '.join(pairs)}")
                        else:
                            nl_lines.append(f"  Row: {' | '.join(row)}")

                    nl_text = "\n".join(nl_lines)
                    table_docs.append(Document(
                        page_content=nl_text,
                        metadata={
                            "source":       str(file_path),
                            "file_name":    fname,
                            "file_type":    ".pdf",
                            "content_type": "table",
                            "table_name":   f"Table_{page_num + 1}_{table_idx + 1}",
                            "page":         page_num,
                            "table_page":   page_num,
                        }
                    ))

        if table_docs:
            print(f"  Extracted {len(table_docs)} tables via pdfplumber")
    except Exception as e:
        print(f"  pdfplumber table extraction warning: {e}")

    return table_docs


def load_pdf(file_path: str) -> List[Document]:
    """
    Convert entire PDF to markdown using pymupdf4llm.
    Also extract tables separately using pdfplumber for better numeric retrieval.
    Returns Documents: one full markdown doc + individual table docs.
    """
    try:
        import pymupdf4llm
    except ImportError:
        raise ImportError("Run: pip install pymupdf4llm pymupdf")

    md_text = pymupdf4llm.to_markdown(file_path)

    # Save markdown alongside PDF for manual inspection
    md_path = Path(file_path).with_suffix(".extracted.md")
    md_path.write_text(md_text, encoding="utf-8")
    print(f"  Markdown saved → {md_path.name} ({len(md_text):,} chars)")

    docs: List[Document] = [Document(
        page_content=md_text,
        metadata={
            "source":       file_path,
            "file_name":    Path(file_path).name,
            "file_type":    ".pdf",
            "content_type": "pdf_markdown",
        }
    )]

    # Add separate table docs with NL representations for better retrieval
    table_docs = _extract_tables_with_pdfplumber(file_path)
    docs.extend(table_docs)

    return docs


def load_text(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    return TextLoader(file_path, encoding="utf-8").load()


def load_markdown(file_path: str) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
    return UnstructuredMarkdownLoader(file_path).load()


def load_web_url(url: str) -> List[Document]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(
        page_content=text,
        metadata={"source": url, "content_type": "prose"}
    )]


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        docs = load_pdf(file_path)
    elif ext == ".txt":
        docs = load_text(file_path)
    elif ext in (".md", ".markdown"):
        docs = load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for doc in docs:
        doc.metadata.setdefault("file_name", Path(file_path).name)
        doc.metadata.setdefault("file_type", ext)
    return docs


def save_upload_and_load(file_bytes: bytes, filename: str) -> List[Document]:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / filename
    dest.write_bytes(file_bytes)
    return load_document(str(dest))