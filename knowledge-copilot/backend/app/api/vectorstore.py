from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.document_loader import SUPPORTED_EXTENSIONS, save_upload_and_load
from app.services.chunker import chunk_documents
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/vectorstore", tags=["vector store"])


@router.get("/stats")
def store_stats():
    """How many vectors are indexed right now?"""
    return get_vector_store().stats()


@router.post("/index")
async def index_document(
    file:          UploadFile = File(...),
    chunk_size:    int  = Form(1000),
    chunk_overlap: int  = Form(200),
    strategy:      str  = Form("recursive"),
):
    """Full pipeline: upload → chunk → embed → store."""
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")

    file_bytes = await file.read()
    docs       = save_upload_and_load(file_bytes, file.filename)
    chunks     = chunk_documents(docs, chunk_size, chunk_overlap, strategy)
    added      = get_vector_store().add_chunks(chunks)

    return {
        "filename":     file.filename,
        "chunks_added": added,
        "store_stats":  get_vector_store().stats(),
    }


@router.post("/search")
def search_store(
    query: str = Form(...),
    k:     int = Form(5),
):
    """Search the vector store with a plain-text query."""
    results = get_vector_store().search(query, k=k)

    if not results:
        return {"query": query, "results": [], "message": "No documents indexed yet."}

    return {
        "query":   query,
        "k":       k,
        "results": [
            {
                "rank":     i + 1,
                "score":    r["score"],
                "preview":  r["text"][:300],
                "metadata": r["metadata"],
            }
            for i, r in enumerate(results)
        ],
    }


@router.delete("/clear")
def clear_store():
    """Wipe the entire vector store. Irreversible."""
    get_vector_store().clear()
    return {"message": "Vector store cleared.", "stats": get_vector_store().stats()}