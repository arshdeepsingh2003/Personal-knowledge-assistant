from fastapi import APIRouter, Form, HTTPException

from app.services.retriever import retrieve_as_dict, format_context_for_llm, retrieve
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/retrieve", tags=["retrieval"])


@router.post("/search")
def retrieval_search(
    query:           str   = Form(...),
    k:               int   = Form(5),
    score_threshold: float = Form(0.30),
):
    """
    Retrieve relevant chunks for a query.
    Returns filtered, deduplicated chunks with source references.
    """
    stats = get_vector_store().stats()
    if stats["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Index at least one document first."
        )

    return retrieve_as_dict(
        query           = query,
        k               = k,
        score_threshold = score_threshold,
    )


@router.post("/context")
def get_llm_context(
    query:           str   = Form(...),
    k:               int   = Form(5),
    score_threshold: float = Form(0.30),
):
    """
    Returns the exact context string that will be injected into the LLM prompt.
    Useful for debugging what the LLM will actually see.
    """
    stats = get_vector_store().stats()
    if stats["total_docs"] == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Index at least one document first."
        )

    result  = retrieve(query, k=k, score_threshold=score_threshold)
    context = format_context_for_llm(result)

    return {
        "query":         query,
        "total_found":   result.total_found,
        "context_chars": len(context),
        "llm_context":   context,
    }