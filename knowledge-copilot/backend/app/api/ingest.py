from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from typing import List

from app.services.document_loader import (
    SUPPORTED_EXTENSIONS,
    load_web_url,
    save_upload_and_load,
)
from app.services.chunker import chunk_documents, get_chunk_stats   # ← ADD

router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
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
    try:
        docs = load_web_url(url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "url": url,
        "pages_loaded": len(docs),
        "preview": docs[0].page_content[:300] if docs else "",
    }


# Upload file → Read it → Break into chunks → Show stats + sample output
@router.post("/chunk")
async def chunk_uploaded_file(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    strategy: str = Form("recursive"),
):
    """Upload a file, load it, chunk it, and return stats + preview."""
    "Validate file type : Is file allowed? (.pdf, .txt, etc.)"
    
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")

    #Convert file → Documents
    """
    File → Text → Document objects
    PDF → pages → Document list
    
    """
    file_bytes = await file.read()
    docs   = save_upload_and_load(file_bytes, file.filename)
    chunks = chunk_documents(docs, chunk_size, chunk_overlap, strategy) #Chunk the document
    stats  = get_chunk_stats(chunks)
    
    """
    Generate stats  
    {
       "total_chunks": 10,
       "avg_length": 850
    }
    """

    return {
        "filename":      file.filename,
        "strategy":      strategy,
        "chunk_size":    chunk_size,
        "chunk_overlap": chunk_overlap,
        "stats":         stats,
        "preview_chunks": [
            {
                "index":   c.metadata.get("chunk_index"),
                "length":  len(c.page_content),
                "content": c.page_content[:200],
                "metadata": c.metadata,
            }
            for c in chunks[:3]   #Only shows first 3 chunks (to avoid overload)
        ],
    }