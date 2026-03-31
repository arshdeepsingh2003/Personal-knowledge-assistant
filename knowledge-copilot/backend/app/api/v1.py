import json
import time
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.services.chat_session import (
    add_message, create_session,
    delete_session, get_history,
    get_session, list_sessions,
)
from app.services.document_loader import SUPPORTED_EXTENSIONS, save_upload_and_load
from app.services.chunker import chunk_documents
from app.services.llm import generate_answer, stream_answer
from app.services.retriever import format_context_for_llm, retrieve
from app.services.vector_store import get_vector_store

router  = APIRouter(prefix="/api/v1", tags=["v1"])
limiter = Limiter(key_func=get_remote_address)

'''
This file defines your entire backend API for a RAG (AI chat + documents) system

In simple words:

👉 Upload documents → store them
👉 Ask questions → AI answers using those documents
👉 Manage chat sessions'''


# Request / Response models
# Used when user asks a question:
'''
It also ensures:
query is not empty
limits are safe
'''
class AskRequest(BaseModel):
    session_id:      str
    query:           str   = Field(..., min_length=1, max_length=2000)
    k:               int   = Field(5,   ge=1, le=20)
    score_threshold: float = Field(0.30, ge=0.0, le=1.0)
    stream:          bool  = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

# Used when uploading documents
class IndexRequest(BaseModel):
    chunk_size:    int = Field(1000, ge=100,  le=8000)
    chunk_overlap: int = Field(200,  ge=0,    le=2000)
    strategy:      str = Field("recursive", pattern="^(recursive|markdown)$")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size", 1000)
        if v >= size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


# Health check — reports real system status
def health_check():
    """
    Deep health check — verifies every subsystem is reachable.
    Returns 200 if healthy, 503 if any critical component is down.
    """
    status_report = {
        "api":          "ok",
        "vector_store": "ok",
        "llm":          "ok",
        "embedding":    "ok",
    }
    issues = []

    # Check vector store
    try:
        stats = get_vector_store().stats()
        status_report["vector_store"] = {
            "status":     "ok",
            "provider":   stats["provider"],
            "total_docs": stats["total_docs"],
        }
    except Exception as e:
        status_report["vector_store"] = "error"
        issues.append(f"vector_store: {e}")

    # Check embedding model
    try:
        from app.services.embedder import get_embedding_dimension
        dim = get_embedding_dimension()
        status_report["embedding"] = {
            "status":    "ok",
            "provider":  settings.embedding_provider,
            "dimension": dim,
        }
    except Exception as e:
        status_report["embedding"] = "error"
        issues.append(f"embedding: {e}")

    # Check LLM config (we don't call it — just verify the key exists)
    try:
        from app.services.llm import get_llm
        get_llm()
        status_report["llm"] = {
            "status":   "ok",
            "provider": settings.llm_provider,
            "model":    settings.llm_model,
        }
    except Exception as e:
        status_report["llm"] = "error"
        issues.append(f"llm: {e}")

    overall = "healthy" if not issues else "degraded"
    http_code = 200 if not issues else 503

    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=http_code,
        content={
            "status":  overall,
            "version": settings.app_version,
            "checks":  status_report,
            "issues":  issues,
        }
    )


# Document management
# Upload file + store in vector DB

@router.post("/documents")
@limiter.limit("10/minute")
async def upload_and_index(
    request:       Request,
    file:          UploadFile = File(...),
    chunk_size:    int  = Form(1000),
    chunk_overlap: int  = Form(200),
    strategy:      str  = Form("recursive"),
):
    """
    Upload a document and immediately index it into the vector store.
    Single endpoint replaces the old /ingest/upload + /vectorstore/index pair.
    """
    # Validate params via pydantic manually
    try:
        params = IndexRequest(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    file_bytes = await file.read()

    if len(file_bytes) > 50 * 1024 * 1024:     # 50 MB hard limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 50 MB.",
        )

    docs    = save_upload_and_load(file_bytes, file.filename)
    chunks  = chunk_documents(docs, params.chunk_size, params.chunk_overlap, params.strategy)
    added   = get_vector_store().add_chunks(chunks)

    return {
        "filename":     file.filename,
        "pages_loaded": len(docs),
        "chunks_added": added,
        "strategy":     params.strategy,
        "store_stats":  get_vector_store().stats(),
    }


@router.get("/documents/status")
def documents_status():
    """Quick summary of what's currently indexed."""
    stats = get_vector_store().stats()
    return {
        "indexed":    stats["total_docs"] > 0,
        "total_docs": stats["total_docs"],
        "provider":   stats["provider"],
    }


# Session management
@router.post("/sessions")
def new_session():
    """Create a new chat session."""
    session_id = create_session()
    return {"session_id": session_id}


@router.get("/sessions")
def all_sessions():
    return {"sessions": list_sessions()}


@router.get("/sessions/{session_id}")
def session_detail(session_id: str):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}



# Core ask endpoint — the only endpoint the frontend needs to call

@router.post("/ask")
@limiter.limit("30/minute")
async def ask(request: Request, body: AskRequest):
    """
    The unified RAG endpoint.

    Accepts JSON body (not form fields) so the frontend can send
    { session_id, query, k, score_threshold, stream } cleanly.

    When stream=false  → returns complete JSON response
    When stream=true   → returns SSE stream (text/event-stream)
    """
    # Guard: session must exist
    session = get_session(body.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /api/v1/sessions first.",
        )

    # Guard: something must be indexed
    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first.",
        )

    # Retrieve context
    result  = retrieve(body.query, k=body.k, score_threshold=body.score_threshold)
    context = format_context_for_llm(result)
    history = get_history(body.session_id)

    sources_payload = [
        {
            "file_name": s.file_name,
            "page":      s.page,
            "score":     s.score,
            "preview":   s.preview,
        }
        for s in result.sources
    ]

    # ── Streaming response ────────────────────────────────────────────────────
    if body.stream:
        add_message(body.session_id, "user", body.query)

        def event_gen():
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

            full = []
            for token in stream_answer(body.query, context, history):
                full.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            add_message(body.session_id, "assistant", "".join(full))
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Blocking response ─────────────────────────────────────────────────────
    answer = generate_answer(body.query, context, history)

    add_message(body.session_id, "user",      body.query)
    add_message(body.session_id, "assistant", answer)

    return {
        "session_id":   body.session_id,
        "query":        body.query,
        "answer":       answer,
        "sources":      sources_payload,
        "context_used": result.total_found > 0,
        "meta": {
            "chunks_retrieved": result.total_found,
            "model":            settings.llm_model,
            "provider":         settings.llm_provider,
        },
    }