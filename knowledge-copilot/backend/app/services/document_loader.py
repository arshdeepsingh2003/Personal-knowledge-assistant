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


def load_pdf(file_path: str) -> List[Document]:
    """
    Convert entire PDF to markdown using pymupdf4llm.
    Returns ONE Document containing the full markdown.
    The chunker handles splitting on ## boundaries.
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

    return [Document(
        page_content=md_text,
        metadata={
            "source":       file_path,
            "file_name":    Path(file_path).name,
            "file_type":    ".pdf",
            "content_type": "pdf_markdown",
        }
    )]


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