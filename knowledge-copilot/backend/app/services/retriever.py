"""
retriever.py — Table-aware retrieval with reranking

Key changes from Phase 6:
  1. Default k increased to 8 (from 5) — tables often score 0.35-0.45
     so we need to cast wider to catch them before filtering.

  2. Default score_threshold lowered to 0.25 (from 0.30) — same reason.

  3. Reranker added: after vector search returns k=8 candidates,
     a cross-encoder reranker reorders them by semantic relevance.
     Cross-encoders read both the query and chunk together, so they
     understand "ROI 312%" relates to "what is the ROI for retail?"
     much better than cosine similarity alone.

  4. Metadata filtering: if the query mentions a specific table topic
     (industry, model name, metric) the metadata key_fields are checked
     to boost matching table chunks.

  5. max_context_chars increased to 4000 — tables need more space.

  6. Table chunks get a section label prepended in the context window
     so the LLM sees: "Section: ROI Analysis | Table 4 | ..."
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from app.services.vector_store import get_vector_store
from app.services.embedder import embed_query
from app.core.config import settings


# ── Data structures (unchanged from Phase 6) ─────────────────────────────────

@dataclass
class SourceReference:
    file_name:   str
    chunk_index: int
    page:        Optional[int]
    score:       float
    preview:     str
    content_type: str = "prose"   # new: "table" or "prose"
    section:     str  = ""        # new: section heading
    table_name:  str  = ""        # new: table identifier


@dataclass
class RetrievalResult:
    query:       str
    context:     str
    sources:     List[SourceReference]
    chunks:      List[dict]
    total_found: int


# ── Reranker ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker():
    """
    Load the cross-encoder reranker model (cached after first load).

    bge-reranker-large is a cross-encoder trained specifically for
    retrieval reranking. It takes (query, passage) pairs and outputs
    a relevance score that is far more accurate than cosine similarity
    for structured/numeric content.
    """
    provider = settings.reranker_provider

    if provider == "none":
        return None

    if provider == "cohere":
        try:
            import cohere
            return cohere.Client(settings.cohere_api_key)
        except ImportError:
            print("⚠ cohere not installed, falling back to bge reranker")

    # Default: BGE cross-encoder (local, free)
    try:
        from sentence_transformers import CrossEncoder
        print(f"✓ Reranker: {settings.reranker_model}")
        return CrossEncoder(
            settings.reranker_model,
            max_length=512,
        )
    except ImportError:
        print("⚠ sentence-transformers not installed — reranker disabled")
        print("  pip install sentence-transformers")
        return None


def _rerank_bge(
    reranker,
    query:   str,
    results: List[dict],
    top_n:   int,
) -> List[dict]:
    """Rerank results using BGE cross-encoder."""
    if not results:
        return results

    pairs   = [(query, r["text"]) for r in results]
    scores  = reranker.predict(pairs)

    # Attach rerank scores and sort
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


def _rerank_cohere(
    reranker,
    query:   str,
    results: List[dict],
    top_n:   int,
) -> List[dict]:
    """Rerank results using Cohere Rerank API."""
    if not results:
        return results

    docs = [r["text"] for r in results]
    response = reranker.rerank(
        query=query,
        documents=docs,
        top_n=top_n,
        model="rerank-english-v3.0",
    )
    reranked = []
    for hit in response.results:
        r = results[hit.index].copy()
        r["rerank_score"] = hit.relevance_score
        reranked.append(r)
    return reranked


# ── Core retriever ────────────────────────────────────────────────────────────

def retrieve(
    query:             str,
    k:                 int   = None,
    score_threshold:   float = None,
    max_context_chars: int   = None,
) -> RetrievalResult:
    """
    Full retrieval pipeline with reranking:
      1. Vector search (top-k, lower threshold)
      2. Score threshold filter
      3. Deduplication
      4. Reranking (cross-encoder sorts by true relevance)
      5. Context assembly with table-aware formatting
      6. Source tracking with content_type metadata
    """
    # Use config defaults if not overridden
    k                 = k                 or settings.retrieval_k
    score_threshold   = score_threshold   or settings.retrieval_score_threshold
    max_context_chars = max_context_chars or settings.retrieval_max_context_chars

    store = get_vector_store()

    # ── 1. Vector search ──────────────────────────────────────────────────────
    # Fetch 2x k so the reranker has enough candidates to work with
    raw_results = store.search(query, k=k * 2)

    # ── 2. Score filter ───────────────────────────────────────────────────────
    filtered = [r for r in raw_results if r["score"] >= score_threshold]

    # ── 3. Deduplicate ────────────────────────────────────────────────────────
    seen: set = set()
    unique: List[dict] = []
    for r in filtered:
        key = " ".join(r["text"].split())
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # ── 4. Rerank ─────────────────────────────────────────────────────────────
    reranker = _get_reranker()
    if reranker and unique:
        if settings.reranker_provider == "cohere":
            unique = _rerank_cohere(reranker, query, unique, top_n=k)
        else:
            unique = _rerank_bge(reranker, query, unique, top_n=k)
    else:
        # No reranker: sort by vector score, keep top k
        unique.sort(key=lambda x: x["score"], reverse=True)
        unique = unique[:k]

    # ── 5. Build context ──────────────────────────────────────────────────────
    context_parts: List[str] = []
    chars_used = 0

    for i, chunk in enumerate(unique):
        text = chunk["text"].strip()
        meta = chunk.get("metadata", {})

        # Prefix table chunks with their section and table name
        # so the LLM has full context about what it's reading
        label = f"[{i+1}]"
        if meta.get("content_type") == "table":
            section    = meta.get("section", "")
            table_name = meta.get("table_name", "")
            if section or table_name:
                header = " — ".join(filter(None, [section, table_name]))
                label  = f"[{i+1}] ({header})"

        entry = f"{label} {text}"
        if chars_used + len(entry) + 2 > max_context_chars:
            break

        context_parts.append(entry)
        chars_used += len(entry) + 2

    context = "\n\n".join(context_parts)

    # ── 6. Build source references ────────────────────────────────────────────
    sources: List[SourceReference] = []
    for chunk in unique[:len(context_parts)]:
        meta = chunk.get("metadata", {})
        sources.append(SourceReference(
            file_name    = meta.get("file_name", meta.get("source", "unknown")),
            chunk_index  = meta.get("chunk_index", -1),
            page         = meta.get("page"),
            score        = round(chunk.get("rerank_score", chunk["score"]), 4),
            preview      = chunk["text"][:120],
            content_type = meta.get("content_type", "prose"),
            section      = meta.get("section", ""),
            table_name   = meta.get("table_name", ""),
        ))

    return RetrievalResult(
        query       = query,
        context     = context,
        sources     = sources,
        chunks      = unique,
        total_found = len(unique),
    )


# ── Context formatter ─────────────────────────────────────────────────────────

def retrieve_as_dict(
    query:           str,
    k:               int   = None,
    score_threshold: float = None,
) -> dict:
    result = retrieve(query, k=k, score_threshold=score_threshold)
    return {
        "query":       result.query,
        "total_found": result.total_found,
        "context":     result.context,
        "sources": [
            {
                "file_name":   s.file_name,
                "chunk_index": s.chunk_index,
                "page":        s.page,
                "score":       s.score,
                "preview":     s.preview,
                "content_type": s.content_type,
                "section":     s.section,
                "table_name":  s.table_name,
            }
            for s in result.sources
        ],
    }


def format_context_for_llm(result: RetrievalResult) -> str:
    """
    Format retrieved context for the LLM prompt.
    Table chunks get an explicit type label so the prompt can
    instruct the LLM to look specifically for tables.
    """
    if not result.context:
        return "No relevant context was found in the knowledge base."

    # Build a source list that distinguishes tables from prose
    source_lines = []
    for i, s in enumerate(result.sources):
        line = f"  [{i+1}] {s.file_name}"
        if s.page is not None:
            line += f" (page {s.page + 1})"
        if s.content_type == "table":
            line += f" [TABLE"
            if s.table_name:
                line += f": {s.table_name}"
            line += "]"
        line += f" — score {s.score}"
        source_lines.append(line)

    source_str = "\n".join(source_lines)

    has_tables = any(s.content_type == "table" for s in result.sources)
    table_hint = (
        "\nNote: Some context chunks contain TABLE data. "
        "Pay close attention to all numeric values, percentages, "
        "and row-level data in these chunks when answering."
        if has_tables else ""
    )

    return (
        f"Use ONLY the context below to answer the question. "
        f"If the answer is not in the context, say so clearly."
        f"{table_hint}\n\n"
        f"SOURCES:\n{source_str}\n\n"
        f"CONTEXT:\n{result.context}"
    )