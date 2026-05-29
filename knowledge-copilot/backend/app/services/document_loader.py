"""
document_loader.py

Refactored to load documents from bytes in memory (no local temp files).
Files are persisted to Supabase Storage; metadata is stored in MongoDB.
"""

import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from app.core.config import settings
from app.services.supabase_storage import (
    delete_file,
    generate_storage_path,
    get_signed_url,
    upload_file,
)
from app.models.database import get_db

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


# ── In-memory loaders (no local filesystem writes) ─────────────────────────────


def load_pdf_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    import fitz

    doc = fitz.open(stream=file_bytes, filetype="pdf")

    import pymupdf4llm

    md_text = pymupdf4llm.to_markdown(doc)

    docs: List[Document] = [
        Document(
            page_content=md_text,
            metadata={
                "source": filename,
                "file_name": filename,
                "file_type": ".pdf",
                "content_type": "pdf_markdown",
            },
        )
    ]

    table_docs = _extract_tables_from_bytes(file_bytes, filename)
    docs.extend(table_docs)

    return docs


def _extract_tables_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    try:
        import pdfplumber
    except ImportError:
        return []

    table_docs: List[Document] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
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

                    nl_lines = []
                    if header:
                        nl_lines.append(f"Table: {' | '.join(header)}")

                    for row in rows:
                        if header:
                            pairs = [
                                f"{header[j]} is {row[j]}"
                                for j in range(min(len(header), len(row)))
                            ]
                            nl_lines.append(f"  Row: {'; '.join(pairs)}")
                        else:
                            nl_lines.append(f"  Row: {' | '.join(row)}")

                    nl_text = "\n".join(nl_lines)
                    table_docs.append(
                        Document(
                            page_content=nl_text,
                            metadata={
                                "source": filename,
                                "file_name": filename,
                                "file_type": ".pdf",
                                "content_type": "table",
                                "table_name": f"Table_{page_num + 1}_{table_idx + 1}",
                                "page": page_num,
                                "table_page": page_num,
                            },
                        )
                    )

        if table_docs:
            print(f"  Extracted {len(table_docs)} tables via pdfplumber")
    except Exception as e:
        print(f"  pdfplumber table extraction warning: {e}")

    return table_docs


def load_text_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    text = file_bytes.decode("utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "file_name": filename,
                "file_type": ".txt",
                "content_type": "text",
            },
        )
    ]


def load_markdown_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    text = file_bytes.decode("utf-8")
    return [
        Document(
            page_content=text,
            metadata={
                "source": filename,
                "file_name": filename,
                "file_type": ".md",
                "content_type": "markdown",
            },
        )
    ]


def load_document_from_bytes(file_bytes: bytes, filename: str) -> List[Document]:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        docs = load_pdf_from_bytes(file_bytes, filename)
    elif ext == ".txt":
        docs = load_text_from_bytes(file_bytes, filename)
    elif ext in (".md", ".markdown"):
        docs = load_markdown_from_bytes(file_bytes, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    for doc in docs:
        doc.metadata.setdefault("file_name", filename)
        doc.metadata.setdefault("file_type", ext)
    return docs


# ── Web loader (unchanged) ─────────────────────────────────────────────────────


def load_web_url(url: str) -> List[Document]:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [
        Document(
            page_content=text,
            metadata={"source": url, "content_type": "prose"},
        )
    ]


# ── Upload + load (Supabase Storage + MongoDB metadata) ────────────────────────


async def save_upload_and_load(
    file_bytes: bytes,
    filename: str,
    user_id: Optional[str] = None,
) -> List[Document]:
    ext = Path(filename).suffix.lower()
    content_type_map = {
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".markdown": "text/markdown",
    }
    content_type = content_type_map.get(ext, "application/octet-stream")

    storage_path = generate_storage_path(filename, user_id)

    upload_file(
        file_bytes=file_bytes,
        storage_path=storage_path,
        content_type=content_type,
    )

    db = get_db()
    await db.file_uploads.insert_one({
        "filename": filename,
        "storage_path": storage_path,
        "content_type": content_type,
        "file_size": len(file_bytes),
        "uploaded_by": user_id,
        "created_at": datetime.utcnow(),
    })

    return load_document_from_bytes(file_bytes, filename)


# ── Cleanup ────────────────────────────────────────────────────────────────────


async def delete_uploaded_file(file_id: str, user_id: Optional[str] = None) -> bool:
    db = get_db()
    from bson import ObjectId

    query: dict = {"_id": ObjectId(file_id)}
    if user_id:
        query["uploaded_by"] = user_id

    record = await db.file_uploads.find_one(query)
    if not record:
        return False

    try:
        delete_file(record["storage_path"])
    except Exception as e:
        print(f"Warning: failed to delete file from Supabase: {e}")

    await db.file_uploads.delete_one({"_id": ObjectId(file_id)})
    return True


async def get_signed_download_url(file_id: str, user_id: Optional[str] = None) -> Optional[str]:
    db = get_db()
    from bson import ObjectId

    query: dict = {"_id": ObjectId(file_id)}
    if user_id:
        query["uploaded_by"] = user_id

    record = await db.file_uploads.find_one(query)
    if not record:
        return None

    return get_signed_url(record["storage_path"], expiry_seconds=3600)


async def list_uploaded_files(user_id: Optional[str] = None) -> List[dict]:
    db = get_db()
    query: dict = {}
    if user_id:
        query["uploaded_by"] = user_id

    cursor = db.file_uploads.find(query, sort=[("created_at", -1)], limit=100)
    files = []
    async for f in cursor:
        files.append({
            "id": str(f["_id"]),
            "filename": f["filename"],
            "content_type": f.get("content_type", ""),
            "file_size": f.get("file_size", 0),
            "uploaded_by": f.get("uploaded_by"),
            "created_at": f["created_at"],
        })
    return files
