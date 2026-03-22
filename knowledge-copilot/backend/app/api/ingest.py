from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import List

from app.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    load_web_url,
    save_upload_and_load,
)

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a file upload, extract its text, return metadata."""
    ext = "." + file.filename.split(".")[-1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )

    file_bytes = await file.read()
    docs = save_upload_and_load(file_bytes, file.filename)

    return {
        "filename": file.filename,
        "pages_loaded": len(docs),
        "preview": docs[0].page_content[:300] if docs else "",
        "metadata": docs[0].metadata if docs else {},
    }


@router.post("/url")
async def ingest_url(url: str = Form(...)):
    """Fetch a webpage and extract its text."""
    try:
        docs = load_web_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "url": url,
        "pages_loaded": len(docs),
        "preview": docs[0].page_content[:300] if docs else "",
    }