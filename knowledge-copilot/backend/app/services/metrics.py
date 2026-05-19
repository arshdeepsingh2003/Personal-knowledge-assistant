"""
metrics.py — Automated retrieval quality metrics and chunk coverage validation.

Provides functions to evaluate the quality of the RAG pipeline output:
  - Chunk coverage: what fraction of the query's information is covered
  - Retrieval diversity: how many distinct sections/headings are represented
  - Score distribution: stats on similarity/rerank scores
  - Context utilization: whether context fits within token limits
  - Section span: how many different document sections are retrieved
"""

import logging
from typing import List

from app.services.retriever import RetrievalResult

logger = logging.getLogger("knowledge_copilot.metrics")


def compute_chunk_coverage(result: RetrievalResult) -> dict:
    """Analyze how well the retrieved chunks cover the query's needs.

    Returns:
      total_chunks: Number of chunks in final result
      sections_covered: Number of distinct document sections represented
      content_types: Breakdown of content types (prose, table, section)
      avg_chunk_length: Average character length of chunks
      total_context_chars: Total characters in assembled context
    """
    if not result.chunks:
        return {
            "total_chunks": 0,
            "sections_covered": 0,
            "content_types": {},
            "avg_chunk_length": 0,
            "total_context_chars": 0,
        }

    sections = set()
    content_types = {}
    total_len = 0

    for chunk in result.chunks:
        meta = chunk.get("metadata", {})
        section = meta.get("heading", meta.get("section", ""))
        if section:
            sections.add(section)

        ct = meta.get("content_type", "prose")
        content_types[ct] = content_types.get(ct, 0) + 1
        total_len += len(chunk.get("text", ""))

    coverage = {
        "total_chunks": len(result.chunks),
        "sections_covered": len(sections),
        "content_types": content_types,
        "avg_chunk_length": round(total_len / max(len(result.chunks), 1)),
        "total_context_chars": len(result.context),
    }

    logger.info(f"Chunk coverage: {coverage}")
    return coverage


def compute_diversity_score(result: RetrievalResult) -> float:
    """Score from 0-1 indicating how diverse the retrieved chunks are.

    1.0 = chunks come from many different sections with balanced representation.
    0.0 = all chunks from the same section.
    """
    if not result.chunks or len(result.chunks) < 2:
        return 0.0

    section_counts = {}
    for chunk in result.chunks:
        meta = chunk.get("metadata", {})
        section = meta.get("heading", meta.get("section", "__unknown__"))
        section_counts[section] = section_counts.get(section, 0) + 1

    num_sections = len(section_counts)
    max_possible = min(len(result.chunks), max(section_counts.values()))

    if num_sections <= 1:
        return 0.0

    # Normalized entropy: higher = more balanced distribution
    import math
    total = len(result.chunks)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in section_counts.values()
    )
    max_entropy = math.log2(min(num_sections, total))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    return round(normalized_entropy, 4)


def compute_score_distribution(result: RetrievalResult) -> dict:
    """Statistics on score distribution across retrieved chunks."""
    if not result.chunks:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}

    scores = sorted([
        chunk.get("rerank_score", chunk.get("score", 0))
        for chunk in result.chunks
    ], reverse=True)

    n = len(scores)
    return {
        "min": round(scores[-1], 4) if scores else 0,
        "max": round(scores[0], 4) if scores else 0,
        "avg": round(sum(scores) / n, 4) if n > 0 else 0,
        "median": round(scores[n // 2], 4) if n > 0 else 0,
        "count": n,
    }


def validate_context_fit(result: RetrievalResult, max_chars: int) -> dict:
    """Check if context fits within the LLMs token limit (est. 4 chars/token)."""
    if not result.context:
        return {"fits": True, "context_chars": 0, "estimated_tokens": 0}

    context_len = len(result.context)
    estimated_tokens = context_len / 4

    return {
        "fits": context_len <= max_chars,
        "context_chars": context_len,
        "estimated_tokens": round(estimated_tokens),
        "max_chars": max_chars,
    }


def evaluate_retrieval_quality(result: RetrievalResult) -> dict:
    """Run all quality metrics and return a combined report."""
    coverage = compute_chunk_coverage(result)
    diversity = compute_diversity_score(result)
    scores = compute_score_distribution(result)
    context_fit = validate_context_fit(result, 8000)

    report = {
        "query": result.query[:120],
        "coverage": coverage,
        "diversity_score": diversity,
        "score_distribution": scores,
        "context_fit": context_fit,
        "expanded_queries": len(result.expanded_queries),
        "pipeline_metrics": result.retrieval_metrics,
    }

    logger.info(f"Retrieval quality report: {report}")
    return report
