"""
retriever.py — Multi-stage RAG pipeline with MMR, query expansion, section diversity, and reranking.

Pipeline:
  1. Query expansion (optional) — generate variant queries for broader recall
  2. MMR retrieval — fetch diverse candidates via Maximal Marginal Relevance
  3. Score threshold filter
  4. Deduplication (by normalized text)
  5. Cross-encoder reranking with section diversity enforcement
  6. Context assembly with table-aware formatting
  7. Source tracking
  8. Debug logging
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional

from app.services.vector_store import get_vector_store
from app.services.embedder import embed_query
from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.retriever")


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class SourceReference:
    file_name:     str
    chunk_index:   int
    score:         float
    preview:       str
    page:          Optional[int] = None
    rerank_score:  float = 0.0
    content_type:  str = "prose"
    section:       str  = ""
    table_name:    str  = ""
    source_number: int = 0


@dataclass
class RetrievalResult:
    query:            str
    context:          str
    sources:          List[SourceReference]
    chunks:           List[dict]
    total_found:      int
    expanded_queries: List[str] = field(default_factory=lambda: [])
    retrieval_metrics: dict      = field(default_factory=lambda: {})


# ── Query Expansion ─────────────────────────────────────────────────────────

def _expand_query(query: str) -> List[str]:
    """Generate variant queries to improve recall for entity names, synonyms, metrics."""
    if not settings.query_expansion_enabled:
        return [query]

    try:
        from app.services.llm import get_llm
        llm = get_llm()

        prompt = (
            f"Given the user question below, rewrite it into up to {settings.query_expansion_max_terms - 1} "
            f"alternative phrasings that would help find relevant information in a knowledge base. "
            f"Use synonyms, rephrase numeric/metric terms, expand acronyms, and add related concepts. "
            f"Return each variant on a separate line. Do NOT include the original question.\n\n"
            f"Question: {query}"
        )

        response = llm.invoke(prompt)
        variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        variants = variants[:settings.query_expansion_max_terms - 1]

        # Filter out lines that look like meta-commentary
        variants = [v for v in variants if not v.lower().startswith(("here", "sure", "option", "variant"))]

        queries = [query] + variants
        logger.info(f"Query expansion: {len(queries)} queries (original + {len(variants)} variants)")
        for i, q in enumerate(queries):
            logger.debug(f"  Query [{i}]: {q[:120]}")

        return queries
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]


def _search_with_expansion(
    query: str,
    k: int,
    fetch_k: int,
    mmr_lambda: float,
) -> tuple[List[dict], List[str]]:
    """Search with query expansion, merging results from all variants."""
    queries = _expand_query(query)
    store = get_vector_store()

    all_results: List[dict] = []
    seen_texts: set = set()

    for q in queries:
        if hasattr(store, "search_mmr"):
            raw = store.search_mmr(q, k=fetch_k, fetch_k=fetch_k * 2, mmr_lambda=mmr_lambda)
        else:
            raw = store.search(q, k=fetch_k)

        # Deduplicate across query variants
        for r in raw:
            normalized = " ".join(r["text"].split())[:200]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                r["_source_query"] = q
                all_results.append(r)

    return all_results, queries


# ── Reranker ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_reranker():
    provider = settings.reranker_provider

    if provider == "none":
        return None

    if provider == "cohere":
        try:
            import cohere
            return cohere.Client(settings.cohere_api_key)
        except ImportError:
            logger.warning("cohere not installed, falling back to bge reranker")

    try:
        from sentence_transformers import CrossEncoder
        logger.info(f"Reranker: {settings.reranker_model}")
        return CrossEncoder(
            settings.reranker_model,
            max_length=512,
        )
    except ImportError:
        logger.warning("sentence-transformers not installed — reranker disabled")
        return None


def _rerank_bge(reranker, query: str, results: List[dict], top_n: int) -> List[dict]:
    """Rerank results using BGE cross-encoder."""
    if not results:
        return results

    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)

    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)

    if settings.eval_log_reranking:
        logger.info(f"Reranking scores for '{query[:80]}':")
        for i, r in enumerate(sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]):
            logger.info(f"  [{i+1}] score={r['rerank_score']:.4f} | preview={r['text'][:120]}")

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]


def _rerank_cohere(reranker, query: str, results: List[dict], top_n: int) -> List[dict]:
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

    if settings.eval_log_reranking:
        logger.info(f"Cohere reranking scores for '{query[:80]}':")
        for i, r in enumerate(reranked):
            logger.info(f"  [{i+1}] score={r['rerank_score']:.4f} | preview={r['text'][:120]}")

    return reranked


# ── Section diversity enforcement ──────────────────────────────────────────

def _enforce_section_diversity(
    chunks: List[dict],
    k: int,
    min_sections: int,
) -> List[dict]:
    """Ensure the final set of chunks spans multiple sections.

    After reranking, if too many chunks come from the same section,
    this promotes chunks from under-represented sections up in rank
    so the LLM sees diverse context.
    """
    if not settings.retrieval_section_diversity:
        return chunks[:k]

    section_counts: dict = {}
    for c in chunks:
        section = c.get("metadata", {}).get("heading", "") or c.get("metadata", {}).get("section", "")
        if not section:
            section = "__prose__"
        section_counts.setdefault(section, [])
        section_counts[section].append(c)

    # If we already have enough sections, take top k
    if len(section_counts) >= min(min_sections, len(chunks)):
        return chunks[:k]

    logger.info(f"Section diversity: only {len(section_counts)} sections found, enforcing minimum {min_sections}")

    selected: List[dict] = []
    selected_sections: set = set()

    # First pass: pick the top chunk from each section, cycling through under-represented ones
    for c in chunks:
        section = c.get("metadata", {}).get("heading", "") or c.get("metadata", {}).get("section", "") or "__prose__"
        if section not in selected_sections:
            selected.append(c)
            selected_sections.add(section)
            if len(selected) >= k:
                break

    # Second pass: fill remaining slots with best remaining chunks by rerank score
    if len(selected) < k:
        for c in chunks:
            section = c.get("metadata", {}).get("heading", "") or c.get("metadata", {}).get("section", "") or "__prose__"
            if c not in selected:
                selected.append(c)
                if len(selected) >= k:
                    break

    return selected[:k]


# ── Core retriever ──────────────────────────────────────────────────────────

def retrieve(
    query:             str,
    k:                 int   = None,
    score_threshold:   float = None,
    max_context_chars: int   = None,
    mmr_lambda:        float = None,
) -> RetrievalResult:
    """
    Full retrieval pipeline:

    1. Query expansion (optional)
    2. MMR vector search (fetch_k candidates for diversity)
    3. Score threshold filter
    4. Deduplication
    5. Cross-encoder reranking
    6. Section diversity enforcement
    7. Context assembly with table-aware formatting
    8. Source tracking
    9. Debug logging
    """
    k                 = k                 or settings.retrieval_k
    fetch_k           = settings.retrieval_fetch_k
    score_threshold   = score_threshold   or settings.retrieval_score_threshold
    max_context_chars = max_context_chars or settings.retrieval_max_context_chars
    mmr_lambda        = mmr_lambda        or settings.retrieval_mmr_lambda

    # ── 1. Query expansion + search ────────────────────────────────────────
    raw_results, expanded_queries = _search_with_expansion(query, k, fetch_k, mmr_lambda)

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Retrieved {len(raw_results)} raw candidates for query: '{query[:120]}'")
        for i, r in enumerate(raw_results[:10]):
            score_info = f"vector_score={r.get('score', 0):.4f}"
            if settings.eval_log_scores:
                section = r.get("metadata", {}).get("heading", "") or r.get("metadata", {}).get("section", "")
                logger.info(f"  [{i+1}] {score_info} | section='{section}' | preview={r['text'][:100]}")

    # ── 2. Score filter ────────────────────────────────────────────────────
    filtered = [r for r in raw_results if r["score"] >= score_threshold]

    if settings.eval_log_scores:
        logger.info(f"After score filter (≥{score_threshold}): {len(filtered)} / {len(raw_results)} chunks")

    # ── 3. Deduplicate ─────────────────────────────────────────────────────
    seen: set = set()
    unique: List[dict] = []
    for r in filtered:
        key = " ".join(r["text"].split())[:300]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if settings.eval_log_retrieved_chunks:
        logger.info(f"After dedup: {len(unique)} unique chunks")

    # ── 4. Rerank ──────────────────────────────────────────────────────────
    reranker = _get_reranker()
    if reranker and unique:
        if settings.reranker_provider == "cohere":
            reranked = _rerank_cohere(reranker, query, unique, top_n=min(k * 2, len(unique)))
        else:
            reranked = _rerank_bge(reranker, query, unique, top_n=min(k * 2, len(unique)))
    else:
        reranked = sorted(unique, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        reranked = reranked[:k]

    if settings.eval_log_reranking:
        logger.info(f"After reranking: top {len(reranked)} chunks")

    # ── 5. Section diversity ──────────────────────────────────────────────
    diversified = _enforce_section_diversity(reranked, k, settings.retrieval_min_sections)

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Final {len(diversified)} chunks sent to LLM:")
        for i, c in enumerate(diversified):
            section = c.get("metadata", {}).get("heading", "") or c.get("metadata", {}).get("section", "")
            score = c.get("rerank_score", c.get("score", 0))
            logger.info(f"  [{i+1}] score={score:.4f} | section='{section}' | len={len(c['text'])} | preview={c['text'][:80]}")

    # ── 6. Build context ───────────────────────────────────────────────────
    context_parts: List[str] = []
    chars_used = 0
    source_number = 1

    for chunk in diversified:
        text = chunk["text"].strip()
        meta = chunk.get("metadata", {})

        label = f"[{source_number}]"
        if meta.get("content_type") == "table":
            section    = meta.get("section", "")
            table_name = meta.get("table_name", "")
            if section or table_name:
                header = " — ".join(filter(None, [section, table_name]))
                label  = f"[{source_number}] ({header})"

        # Track source number on chunk for reference matching
        chunk["_source_number"] = source_number

        entry = f"{label} {text}"
        if chars_used + len(entry) + 2 > max_context_chars:
            break

        context_parts.append(entry)
        chars_used += len(entry) + 2
        source_number += 1

    context = "\n\n".join(context_parts)

    # ── 7. Build source references ─────────────────────────────────────────
    sources: List[SourceReference] = []
    for chunk in diversified[:len(context_parts)]:
        meta = chunk.get("metadata", {})
        src_num = chunk.get("_source_number", 0)
        sources.append(SourceReference(
            file_name     = meta.get("file_name", meta.get("source", "unknown")),
            chunk_index   = meta.get("chunk_index", -1),
            page          = meta.get("page"),
            score         = round(chunk.get("rerank_score", chunk.get("score", 0)), 4),
            rerank_score  = round(chunk.get("rerank_score", 0), 4),
            preview       = chunk["text"][:120],
            content_type  = meta.get("content_type", "prose"),
            section       = meta.get("heading", meta.get("section", "")),
            table_name    = meta.get("table_name", ""),
            source_number = src_num,
        ))

    # ── 8. Retrieval metrics ──────────────────────────────────────────────
    sections_found = set()
    for c in diversified:
        sec = c.get("metadata", {}).get("heading", "") or c.get("metadata", {}).get("section", "")
        if sec:
            sections_found.add(sec)

    metrics = {
        "total_candidates": len(raw_results),
        "after_filter":     len(filtered),
        "after_dedup":      len(unique),
        "after_rerank":     len(reranked),
        "final_chunks":     len(diversified),
        "sections_covered": len(sections_found),
        "expanded_queries": len(expanded_queries),
    }

    return RetrievalResult(
        query            = query,
        context          = context,
        sources          = sources,
        chunks           = diversified,
        total_found      = len(diversified),
        expanded_queries = expanded_queries,
        retrieval_metrics = metrics,
    )


# ── Context formatter ───────────────────────────────────────────────────────

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
        "expanded_queries": result.expanded_queries,
        "retrieval_metrics": result.retrieval_metrics,
        "sources": [
            {
                "file_name":   s.file_name,
                "chunk_index": s.chunk_index,
                "page":        s.page,
                "score":       s.score,
                "rerank_score": s.rerank_score,
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
    Format retrieved context for the LLM prompt with multi-section synthesis support.
    """
    if not result.context:
        return "No relevant context was found in the knowledge base."

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
        if s.section:
            line += f" [Section: {s.section}]"
        line += f" — score {s.score}"
        source_lines.append(line)

    source_str = "\n".join(source_lines)

    has_tables = any(s.content_type == "table" for s in result.sources)
    multi_section = len(set(s.section for s in result.sources if s.section)) > 1

    hints = []
    if has_tables:
        hints.append("Some context chunks contain TABLE data. Pay close attention to all numeric values, percentages, and row-level data.")
    if multi_section:
        hints.append("The context spans MULTIPLE SECTIONS. You may need to combine information across different sections to fully answer the question.")

    hint_str = "\n".join(f"Note: {h}" for h in hints)

    return (
        f"Use ONLY the context below to answer the question. "
        f"If the answer is not in the context, say so clearly."
        f"{chr(10) + hint_str if hints else ''}\n\n"
        f"SOURCES:\n{source_str}\n\n"
        f"CONTEXT:\n{result.context}"
    )
