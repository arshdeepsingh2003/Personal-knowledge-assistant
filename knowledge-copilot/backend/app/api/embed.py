from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.document_loader import SUPPORTED_EXTENSIONS, save_upload_and_load
from app.services.chunker import chunk_documents
from app.services.embedder import embed_chunks, embed_query, get_embedding_dimension

router = APIRouter(prefix="/embed", tags=["embeddings"])

# Check which embedding model is running
'''
{
  "provider": "local", <-- model
  "model": "all-MiniLM-L6-v2",
  "dimension": 384
}

'''
@router.get("/info")
def embedding_info():
    """Return which model is active and its vector dimension."""
    from app.core.config import settings
    dim = get_embedding_dimension()
    return {
        "provider":  settings.embedding_provider,
        "model":     (
            settings.embedding_model_local
            if settings.embedding_provider == "local"
            else settings.embedding_model_openai
        ),
        "dimension": dim,
    }



#Convert user query → embedding vector
'''
Input (Form):query = "What is AI?"
Output:
    {
  "query": "What is AI?",
  "dimension": 384,
  "preview": [0.12, -0.45, ...]
    }
'''
@router.post("/query")
def embed_single_query(query: str = Form(...)):
    """Embed one query string — useful for debugging retrieval."""
    vector = embed_query(query)
    return {
        "query":     query,
        "dimension": len(vector),
        "preview":   vector[:8],   # first 8 values — full vector is huge
    }

# This is your full pipeline API
'''
Step 1: Upload file
Step 2: Check extension : Prevents invalid files (like .exe)
Step 3: Load document
Step 4: Chunking : Breaks large text into smaller pieces
Step 5: Embedding : Converts each chunk → vector
Step 6: Return preview

{
  "filename": "notes.pdf",
  "total_chunks": 25,
  "dimension": 384,
  "preview": {
    "text": "First 200 chars...",
    "metadata": {...},
    "vector_preview": [0.12, -0.34, ...]
  }
}
'''

@router.post("/document")
async def embed_document(
    file: UploadFile = File(...),
    chunk_size:    int = Form(1000),
    chunk_overlap: int = Form(200),
):
    """Upload → chunk → embed. Returns stats and a preview of first chunk."""
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported type: {ext}")

    file_bytes     = await file.read()
    docs           = save_upload_and_load(file_bytes, file.filename)
    chunks         = chunk_documents(docs, chunk_size, chunk_overlap)
    embedded       = embed_chunks(chunks)

    first = embedded[0]
    return {
        "filename":       file.filename,
        "total_chunks":   len(embedded),
        "dimension":      len(first["embedding"]),
        "preview": {
            "text":       first["text"][:200],
            "metadata":   first["metadata"],
            "vector_preview": first["embedding"][:6],
        },
    }