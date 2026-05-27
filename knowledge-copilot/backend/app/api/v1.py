import json
import logging
import time
from typing import Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings
from app.middleware.auth_middleware import get_current_user
from app.services.chat_history import (
    create_session, delete_session, get_session,
    get_history_for_llm, get_messages, list_sessions,
    save_user_message, save_assistant_message,
)
from app.services.document_loader import SUPPORTED_EXTENSIONS, save_upload_and_load
from app.services.chunker import chunk_documents
from app.services.llm import generate_answer, generate_answer_with_meta, stream_answer
from app.services.retriever import format_context_for_llm, retrieve, RetrievalResult
from app.services.summarizer import hierarchical_summarize
from app.services.vector_store import get_vector_store
from app.services.query_analyzer import analyze_query, clarify_query
from app.services.confidence import estimate_confidence, check_citation_grounding
from app.services.metrics import evaluate_retrieval_quality, evaluate_response_quality
from app.models.database import get_db
from datetime import datetime

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
    k:               int   = Field(settings.retrieval_k, ge=1, le=20)
    score_threshold: float = Field(settings.retrieval_score_threshold, ge=0.0, le=1.0)
    source_files:    Optional[list[str]] = None
    stream:          bool  = False

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()

# Used when uploading documents
class IndexRequest(BaseModel):
    chunk_size:    int = Field(settings.chunking_default_size, ge=100,  le=8000)
    chunk_overlap: int = Field(settings.chunking_default_overlap, ge=0, le=2000)
    strategy:      str = Field(settings.chunking_default_strategy, pattern="^(recursive|markdown|structure_aware|semantic)$")

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
    chunk_size:    int  = Form(settings.chunking_default_size),
    chunk_overlap: int  = Form(settings.chunking_default_overlap),
    strategy:      str  = Form(settings.chunking_default_strategy),
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


@router.get("/documents/sources")
def list_sources():
    """List all indexed document sources with chunk counts."""
    sources = get_vector_store().list_sources()
    total = sum(sources.values())
    return {
        "total_chunks": total,
        "total_sources": len(sources),
        "sources": [{"file_name": k, "chunks": v} for k, v in sources.items()],
    }


# ── Summarization ──────────────────────────────────────────────────────────────

class SummarizeRequest(BaseModel):
    source_files: Optional[list[str]] = None


@router.post("/summarize")
@limiter.limit("10/minute")
async def summarize_document(
    request:      Request,
    body:         SummarizeRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Hierarchical document summarization.

    Generates a structured technical summary using:
      1. Full-document chunk retrieval (all chunks, no top-k cutoff)
      2. Concept extraction via term-frequency analysis
      3. Concept-weighted importance scoring (repeated technical concepts
         like embeddings, pipelines, validation receive higher weight)
      4. Section-aware chunk selection (round-robin across all sections)
      5. Per-chunk summarization (first-pass LLM)
      6. Global summary generation (second-pass LLM merge)

    The summary prioritizes core contributions — systems, pipelines,
    algorithms, and AI features — across the entire document, avoiding
    over-focus on the last section.
    """
    if not settings.summarization_enabled:
        raise HTTPException(status_code=400, detail="Summarization is disabled")

    if get_vector_store().stats()["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. Upload a document first.",
        )

    summary = hierarchical_summarize(source_files=body.source_files)

    return {
        "summary":     summary,
        "char_count":  len(summary),
        "source_files": body.source_files or "all",
    }


# Session management
@router.post("/sessions")
async def new_session(
    current_user: dict = Depends(get_current_user)
):
    """Create a new chat session."""
    session_id = await create_session(user_id=current_user["id"])
    return {"session_id": session_id}


@router.get("/sessions")
async def all_sessions(
    current_user: dict = Depends(get_current_user)
):
    sessions = await list_sessions(user_id=current_user["id"])
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
async def session_detail(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    session = await get_session(session_id, user_id=current_user["id"])
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = await get_messages(session_id, user_id=current_user["id"])
    return {**session, "messages": messages}


@router.get("/sessions/{session_id}/messages")
async def session_messages(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    """Fetch all messages for a session (oldest first)."""
    messages = await get_messages(session_id, user_id=current_user["id"])
    return {"messages": messages}


@router.delete("/sessions/{session_id}")
async def remove_session(
    session_id:   str,
    current_user: dict = Depends(get_current_user),
):
    ok = await delete_session(session_id, user_id=current_user["id"])
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session deleted"}


class RenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id:   str,
    body:         RenameRequest,
    current_user: dict = Depends(get_current_user),
):
    """Rename a session."""
    db = get_db()
    result = await db.chat_sessions.find_one_and_update(
        {"_id": ObjectId(session_id), "user_id": current_user["id"]},
        {"$set": {"title": body.title.strip()[:200], "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "id":         str(result["_id"]),
        "title":      result["title"],
        "created_at": result["created_at"],
    }



# Core ask endpoint — the only endpoint the frontend needs to call

@router.post("/ask")
@limiter.limit("30/minute")
async def ask(
    request:      Request,
    body:         AskRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    The unified RAG endpoint.

    Accepts JSON body (not form fields) so the frontend can send
    { session_id, query, k, score_threshold, stream } cleanly.

    When stream=false  → returns complete JSON response
    When stream=true   → returns SSE stream (text/event-stream)
    """
    # Guard: session must exist and belong to this user
    session = await get_session(body.session_id, user_id=current_user["id"])
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

    # ── Query analysis: ambiguity & adversarial detection ─────────────────────
    query_analysis = analyze_query(body.query)
    if query_analysis.get("is_adversarial"):
        logger = logging.getLogger("knowledge_copilot.api")
        logger.warning(
            f"Adversarial query blocked: "
            f"reasons={query_analysis['adversarial_reasons']}, "
            f"score={query_analysis['adversarial_score']}"
        )
        raise HTTPException(
            status_code=400,
            detail="Query was flagged as potentially malicious and was blocked.",
        )

    effective_query = body.query
    if settings.query_ambiguity_detection and query_analysis.get("is_ambiguous"):
        clarified = clarify_query(body.query, query_analysis)
        if clarified != body.query:
            effective_query = clarified
            logger = logging.getLogger("knowledge_copilot.api")
            logger.info(f"Query clarified: '{body.query}' → '{effective_query}'")

    # Detect summarization intent for retrieval mode
    summarization_mode = (
        query_analysis.get("intent") == "summarization"
        if settings.retrieval_hybrid_search
        else False
    )
    # Retrieve context (summarization mode adjusts parameters for broader coverage)
    result  = retrieve(
        effective_query, k=body.k,
        score_threshold=body.score_threshold,
        source_files=body.source_files,
        summarization_mode=summarization_mode,
    )
    context = format_context_for_llm(result)
    history = await get_history_for_llm(body.session_id, user_id=current_user["id"])

    sources_payload = [
        {
            "file_name": s.file_name,
            "page":      s.page,
            "score":     s.score,
            "preview":   s.preview,
        }
        for s in result.sources
    ]

    sources_for_llm = [
        {"file_name": s.file_name, "page": s.page, "score": s.score}
        for s in result.sources
    ]

    # ── Streaming response ────────────────────────────────────────────────────
    if body.stream:
        await save_user_message(
            session_id=body.session_id,
            user_id=current_user["id"],
            content=body.query,
        )

        async def event_gen():
            full = []

            yield f"data: {json.dumps({'type': 'sources', 'sources': sources_payload})}\n\n"

            for token in stream_answer(
                effective_query, context, history,
                chunks=result.chunks, sources=sources_for_llm,
            ):
                full.append(token)
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            full_answer = "".join(full)

            # Run confidence check on streaming response
            confidence_result = {}
            if settings.confidence_enabled and result.chunks:
                try:
                    confidence_result = estimate_confidence(full_answer, result.chunks)
                    cit_check = check_citation_grounding(full_answer, sources_for_llm)
                    confidence_result["citation_check"] = cit_check
                except Exception:
                    pass

            await save_assistant_message(
                session_id     = body.session_id,
                user_id        = current_user["id"],
                content        = full_answer,
                sources        = sources_payload,
                context_chunks = result.chunks,
                model          = settings.llm_model,
            )

            if confidence_result:
                yield f"data: {json.dumps({'type': 'confidence', 'confidence': confidence_result})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ── Blocking response ─────────────────────────────────────────────────────
    answer_meta = generate_answer_with_meta(
        effective_query, context, history,
        chunks=result.chunks, sources=sources_for_llm,
    )
    answer = answer_meta["answer"]
    confidence_result = answer_meta.get("confidence", {})

    # ── Retrieval quality evaluation ──────────────────────────────────────────
    quality_eval = {}
    if settings.eval_trace_enabled:
        try:
            quality_eval = evaluate_response_quality(answer, body.query, result)
        except Exception:
            pass

    await save_user_message(
        session_id=body.session_id,
        user_id=current_user["id"],
        content=body.query,
    )
    await save_assistant_message(
        session_id     = body.session_id,
        user_id        = current_user["id"],
        content        = answer,
        sources        = sources_payload,
        context_chunks = result.chunks,
        model          = settings.llm_model,
    )

    response = {
        "session_id":   body.session_id,
        "query":        body.query,
        "answer":       answer,
        "sources":      sources_payload,
        "context_used": result.total_found > 0,
        "meta": {
            "chunks_retrieved":  result.total_found,
            "model":             settings.llm_model,
            "provider":          settings.llm_provider,
            "expanded_queries":  result.expanded_queries,
            "retrieval_metrics": result.retrieval_metrics,
            "source_files":      body.source_files,
        },
    }

    if confidence_result:
        response["confidence"] = confidence_result
    if quality_eval:
        response["quality_eval"] = quality_eval
    if query_analysis.get("is_ambiguous") and effective_query != body.query:
        response["meta"]["original_query"] = body.query
        response["meta"]["clarified_query"] = effective_query

    return response


# Admin / maintenance

@router.delete("/vectorstore/clear")
def clear_vector_store(current_user: dict = Depends(get_current_user)):
    """Wipe the entire vector store. All indexed documents will be removed."""
    store = get_vector_store()
    before = store.stats()["total_docs"]
    store.clear()
    return {
        "message":     f"Vector store cleared. Removed {before} chunks.",
        "before":      before,
        "after":       0,
    }