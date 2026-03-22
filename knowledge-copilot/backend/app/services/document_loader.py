import os
import shutil
import tempfile
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

from app.core.config import settings


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_pdf(file_path: str) -> List[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> List[Document]:
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def load_markdown(file_path: str) -> List[Document]:
    loader = UnstructuredMarkdownLoader(file_path)
    return loader.load()


def load_web_url(url: str) -> List[Document]:
    # ✅ FIX: Add headers to avoid 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    return [Document(
        page_content=text,
        metadata={"source": url, "type": "web"}
    )]


def load_document(file_path: str) -> List[Document]:
    """Route to the right loader based on file extension."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        docs = load_pdf(file_path)
    elif ext in (".txt",):
        docs = load_text(file_path)
    elif ext in (".md", ".markdown"):
        docs = load_markdown(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Attach extra metadata to every page/chunk
    for doc in docs:
        doc.metadata["file_name"] = Path(file_path).name
        doc.metadata["file_type"] = ext

    return docs


def save_upload_and_load(file_bytes: bytes, filename: str) -> List[Document]:
    """
    Save an uploaded file to disk, load it, return Documents.
    Keeps the file on disk for later re-use (e.g. re-chunking).
    """
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    dest_path = upload_dir / filename

    with open(dest_path, "wb") as f:
        f.write(file_bytes)

    return load_document(str(dest_path))