from dataclasses import dataclass
from typing import List, Optional

from app.services.vector_store import get_vector_store


# =========================
# Data structures
# =========================

@dataclass
class SourceReference:
    """Tracks exactly where a retrieved chunk came from."""
    file_name: str
    chunk_index: int
    page: Optional[int]
    score: float
    preview: str  # first 120 chars


@dataclass
class RetrievalResult:
    """
    context  → injected into LLM
    sources  → citations
    chunks   → raw debug data
    """
    query: str
    context: str
    sources: List[SourceReference]
    chunks: List[dict]
    total_found: int


# =========================
# Core retriever
# =========================

def retrieve(
    query: str,
    k: int = 5,
    score_threshold: float = 0.30,
    max_context_chars: int = 3000,
) -> RetrievalResult:

    store = get_vector_store()

    # 1. Vector search
    raw_results = store.search(query, k=k) or []

    # 2. Score filtering
    filtered = [
        r for r in raw_results
        if r and r.get("score", 0) >= score_threshold
    ]

    # 3. Deduplication
    unique: List[dict] = []
    seen_texts = set()

    for r in filtered:
        text = r.get("text", "").strip()
        if not text:
            continue

        key = hash(text)  # faster + safer

        if key not in seen_texts:
            seen_texts.add(key)
            unique.append(r)

    # 4. Sort by score
    unique.sort(key=lambda x: x.get("score", 0), reverse=True)

    # 5. Build context (with limit)
    context_parts: List[str] = []
    chars_used = 0

    for i, chunk in enumerate(unique):
        text = chunk.get("text", "").strip()
        if not text:
            continue

        if chars_used + len(text) + 2 > max_context_chars:
            break

        context_parts.append(f"[{i+1}] {text}")
        chars_used += len(text) + 2

    context = "\n\n".join(context_parts)

    # 6. Source references
    sources: List[SourceReference] = []

    for chunk in unique[:len(context_parts)]:
        meta = chunk.get("metadata", {})

        sources.append(SourceReference(
            file_name=meta.get("file_name", meta.get("source", "unknown")),
            chunk_index=meta.get("chunk_index", -1),
            page=meta.get("page"),
            score=round(chunk.get("score", 0), 4),
            preview=chunk.get("text", "")[:120],
        ))

    return RetrievalResult(
        query=query,
        context=context,
        sources=sources,
        chunks=unique,
        total_found=len(unique),
    )


# =========================
# API helper
# =========================

def retrieve_as_dict(
    query: str,
    k: int = 5,
    score_threshold: float = 0.30,
) -> dict:

    result = retrieve(query, k=k, score_threshold=score_threshold)

    return {
        "query": result.query,
        "total_found": result.total_found,
        "context": result.context,
        "sources": [
            {
                "file_name": s.file_name,
                "chunk_index": s.chunk_index,
                "page": s.page,
                "score": s.score,
                "preview": s.preview,
            }
            for s in result.sources
        ],
    }


# =========================
# LLM formatting
# =========================

def format_context_for_llm(result: RetrievalResult) -> str:

    if not result.context:
        return "No relevant context was found in the knowledge base."

    source_lines = "\n".join(
        f"[{i+1}] {s.file_name}"
        + (f" (page {s.page})" if s.page is not None else "")
        + f" — score {s.score}"
        for i, s in enumerate(result.sources)
    )

    return (
        "Use ONLY the context below to answer the question.\n"
        "If the answer is not present, clearly say so.\n\n"
        f"SOURCES:\n{source_lines}\n\n"
        f"CONTEXT:\n{result.context}"
    )