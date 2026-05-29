import logging
import traceback

from fastapi import APIRouter, Form, HTTPException

from app.services.retriever import retrieve_as_dict, format_context_for_llm, retrieve
from app.services.vector_store import get_vector_store
from app.services.embedder import embed_query, get_embedding_dimension

logger = logging.getLogger("knowledge_copilot.api")
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
    try:
        logger.info("Retrieval search — query: '%s' (k=%d, threshold=%.2f)", query[:120], k, score_threshold)

        stats = get_vector_store().stats()
        if stats["total_docs"] == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Index at least one document first."
            )

        result = retrieve_as_dict(
            query           = query,
            k               = k,
            score_threshold = score_threshold,
        )
        logger.info("Search complete — total_found=%d", result.get("total_found", 0))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Retrieval search failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {type(e).__name__}: {e}",
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
    try:
        logger.info("Retrieval context — query received: '%s' (k=%d, threshold=%.2f)", query[:120], k, score_threshold)

        stats = get_vector_store().stats()
        logger.info("Vector store stats: %s", stats)
        if stats["total_docs"] == 0:
            raise HTTPException(
                status_code=400,
                detail="Vector store is empty. Index at least one document first."
            )

        result  = retrieve(query, k=k, score_threshold=score_threshold)
        logger.info(
            "Retrieval complete — total_found=%d, sources=%d, expanded_queries=%d",
            result.total_found, len(result.sources), len(result.expanded_queries),
        )

        context = format_context_for_llm(result)

        return {
            "query":         query,
            "total_found":   result.total_found,
            "context_chars": len(context),
            "llm_context":   context,
            "expanded_queries": result.expanded_queries,
            "retrieval_metrics": result.retrieval_metrics,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Retrieval context failed: %s\n%s",
            e, traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Retrieval failed: {type(e).__name__}: {e}",
        )


@router.post("/debug")
def debug_retrieval(
    query: str = Form(...),
    k:     int = Form(5),
):
    """Debug endpoint that returns detailed diagnostic info about the retrieval pipeline."""
    try:
        logger.info("Debug retrieval — query: '%s', k=%d", query[:120], k)

        store = get_vector_store()
        stats = store.stats()
        dim = get_embedding_dimension()
        q_emb = embed_query(query)

        debug_info = {
            "query": query,
            "query_embedding_dim": len(q_emb),
            "expected_dim": dim,
            "vector_store_stats": stats,
            "bm25_index_size": len(store.bm25._docs),
            "bm25_point_ids_count": len(store.bm25._point_ids),
        }

        # Test Qdrant search
        try:
            import numpy as np
            q_vec = np.array([q_emb], dtype="float32")
            q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)
            qdrant_resp = store.client.query_points(
                collection_name=store.collection_name,
                query=q_vec[0].tolist(),
                limit=k,
                with_payload=True,
                with_vectors=False,
            )
            qdrant_results = qdrant_resp.points
            debug_info["qdrant_search_count"] = len(qdrant_results)
            if qdrant_results:
                first = qdrant_results[0]
                debug_info["qdrant_first_result"] = {
                    "id": str(first.id),
                    "score": float(first.score),
                    "payload_keys": list(dict(first.payload or {}).keys()),
                    "has_text": bool(dict(first.payload or {}).get("_text", "")),
                }
        except Exception as e:
            debug_info["qdrant_search_error"] = str(e)

        # Test BM25 search
        try:
            bm25_results = store.bm25.search(query, top_k=k)
            debug_info["bm25_search_count"] = len(bm25_results)
            if bm25_results:
                first_idx, first_score = bm25_results[0]
                debug_info["bm25_first_result"] = {
                    "index": first_idx,
                    "score": float(first_score),
                    "point_id": store.bm25._point_ids[first_idx] if first_idx < len(store.bm25._point_ids) else "N/A",
                }
        except Exception as e:
            debug_info["bm25_search_error"] = str(e)

        # Test hybrid search
        try:
            hybrid_results = store.search_hybrid(query, k=k, fetch_k=k * 4)
            debug_info["hybrid_search_count"] = len(hybrid_results)
            if hybrid_results:
                first = hybrid_results[0]
                debug_info["hybrid_first_result"] = {
                    "id": first.get("id", ""),
                    "score": float(first.get("score", 0)),
                    "source": first.get("_source", ""),
                    "text_preview": first.get("text", "")[:120],
                    "metadata_keys": list(first.get("metadata", {}).keys()),
                    "has_bm25_score": "bm25_score" in first,
                }
        except Exception as e:
            debug_info["hybrid_search_error"] = str(e)

        return debug_info

    except Exception as e:
        logger.error("Debug retrieval failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Debug retrieval failed: {type(e).__name__}: {e}")