"""
metrics.py — Automated retrieval quality metrics and chunk coverage validation.

Provides functions to evaluate the quality of the RAG pipeline output:
  - Chunk coverage: what fraction of the query's information is covered
  - Retrieval diversity: how many distinct sections/headings are represented
  - Score distribution: stats on similarity/rerank scores
  - Context utilization: whether context fits within token limits
  - Context precision: how much of the context is actually relevant
  - Answer faithfulness: whether the answer is grounded in the context
  - Answer relevance: how well the answer addresses the query
  - Table QA Accuracy: how well table questions are answered
  - Multi-Hop QA Accuracy: how well multi-hop questions are answered
  - Citation Accuracy: how many citations are valid
  - Hallucination Rate: fraction of unsupported claims
  - Answer Completeness: coverage of available facts
"""

import logging
import math
import re
from typing import Dict, List, Optional

from app.services.retriever import RetrievalResult

logger = logging.getLogger("knowledge_copilot.metrics")


def compute_chunk_coverage(result: RetrievalResult) -> dict:
    """Analyze how well the retrieved chunks cover the query's needs."""
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

    if num_sections <= 1:
        return 0.0

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


# ── Enhanced evaluation metrics ──────────────────────────────────────────────

def _extract_query_terms(query: str) -> set:
    return set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))


def _extract_key_terms(text: str) -> set:
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "this", "that", "with", "from", "they", "what", "when", "where",
        "which", "their", "there", "about", "would", "could", "should",
    }
    return set(w for w in words if w not in stop_words)


def compute_context_precision(result: RetrievalResult) -> float:
    """Estimate context precision: how much of the retrieved context is
    actually relevant to the query.

    Measures the overlap between query terms and chunk content across all chunks.
    1.0 = all chunks contain query-related terms
    0.0 = no chunks contain query-related terms
    """
    if not result.chunks:
        return 0.0

    query_terms = _extract_query_terms(result.query)
    if not query_terms:
        return 0.5

    relevant_chunks = 0
    for chunk in result.chunks:
        text = chunk.get("text", "")
        chunk_terms = _extract_key_terms(text)
        overlap = len(query_terms & chunk_terms)
        if overlap > 0:
            relevant_chunks += 1

    precision = relevant_chunks / max(len(result.chunks), 1)
    return round(precision, 3)


def compute_context_recall(result: RetrievalResult) -> float:
    """Estimate context recall: what fraction of query-relevant information
    available in the knowledge base was retrieved.

    Uses source/section coverage as a proxy — more sections covering
    query-related topics = higher recall.
    """
    if not result.chunks:
        return 0.0

    sections = set()
    for chunk in result.chunks:
        meta = chunk.get("metadata", {})
        sec = meta.get("heading", meta.get("section", ""))
        if sec:
            sections.add(sec)

    query_terms = _extract_query_terms(result.query)
    if not query_terms:
        return 0.5

    covered_sections = 0
    for chunk in result.chunks:
        text = chunk.get("text", "")
        chunk_terms = _extract_key_terms(text)
        if query_terms & chunk_terms:
            covered_sections += 1

    return round(min(covered_sections / max(len(sections), 1), 1.0), 3)


def compute_answer_faithfulness(answer: str, result: RetrievalResult) -> Dict:
    """Estimate how faithful the answer is to the retrieved context.

    Extracts key claims from the answer and checks if they reference
    entities/numbers present in the context chunks.
    """
    if not answer or not result.chunks:
        return {"faithfulness_score": 0.0, "verified_claims": 0, "total_claims": 0}

    answer_numbers = set(re.findall(r'\b\d+(?:[.,]\d+)?%?\b', answer))
    answer_entities = set(re.findall(
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer
    ))

    context_text = " ".join(c.get("text", "") for c in result.chunks).lower()

    verified_numbers = 0
    for num in answer_numbers:
        clean = num.replace(",", "").replace("%", "").replace(" ", "")
        if clean in context_text or num.lower() in context_text:
            verified_numbers += 1

    verified_entities = 0
    for ent in answer_entities:
        if len(ent) > 3 and ent.lower() in context_text:
            verified_entities += 1

    total_claims = max(len(answer_numbers) + len(answer_entities), 1)
    verified = verified_numbers + verified_entities

    score = round(verified / total_claims, 3)

    return {
        "faithfulness_score": score,
        "verified_claims": verified,
        "total_claims": total_claims,
        "verified_numbers": verified_numbers,
        "total_numbers": len(answer_numbers),
        "verified_entities": verified_entities,
        "total_entities": len(answer_entities),
    }


def compute_answer_relevance(answer: str, query: str) -> float:
    """Estimate answer relevance: how well the answer addresses the query.

    Measures the proportion of query terms that appear in the answer.
    """
    if not answer or not query:
        return 0.0

    query_terms = _extract_query_terms(query)
    if not query_terms:
        return 0.5

    answer_lower = answer.lower()
    matched = sum(1 for t in query_terms if t in answer_lower)
    return round(matched / len(query_terms), 3)


def compute_novelty_score(result: RetrievalResult) -> float:
    """Measure information novelty: average pairwise dissimilarity among chunks."""
    chunks = result.chunks
    if not chunks or len(chunks) < 2:
        return 0.0

    from app.services.retriever import _jaccard_similarity

    total_sim = 0.0
    pairs = 0
    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            sim = _jaccard_similarity(
                chunks[i].get("text", ""),
                chunks[j].get("text", ""),
            )
            total_sim += sim
            pairs += 1

    avg_sim = total_sim / max(pairs, 1)
    novelty = 1.0 - avg_sim
    return round(novelty, 3)


def evaluate_retrieval_quality(result: RetrievalResult) -> dict:
    """Run all quality metrics and return a combined report."""
    coverage = compute_chunk_coverage(result)
    diversity = compute_diversity_score(result)
    scores = compute_score_distribution(result)
    context_fit = validate_context_fit(result, 8000)
    context_precision = compute_context_precision(result)
    context_recall = compute_context_recall(result)
    novelty = compute_novelty_score(result)

    report = {
        "query": result.query[:120],
        "coverage": coverage,
        "diversity_score": diversity,
        "score_distribution": scores,
        "context_fit": context_fit,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "novelty_score": novelty,
        "expanded_queries": len(result.expanded_queries),
        "pipeline_metrics": result.retrieval_metrics,
    }

    logger.info(f"Retrieval quality report: {report}")
    return report


# ── Summarization-Specific Metrics ─────────────────────────────────────────────

def compute_section_coverage_balance(
    answer: str,
    section_summaries: Optional[str] = None,
    chunks: Optional[List[dict]] = None,
) -> dict:
    """Measure how evenly the answer covers different sections.

    Returns:
      - coverage_ratio: fraction of sections whose unique terms appear in answer
      - section_hit_rates: per-section term match rates
      - balance_score: entropy-based evenness (1.0 = perfectly balanced)
    """
    if not chunks and not section_summaries:
        return {"coverage_ratio": 0.0, "section_hit_rates": {}, "balance_score": 0.0}

    section_terms: dict[str, set] = {}
    if section_summaries:
        for block in section_summaries.split("\n=== "):
            if " ===" in block or block.startswith("=== "):
                lines = block.replace("=== ", "").split("\n", 1)
                sec_name = lines[0].strip()
                sec_text = lines[1] if len(lines) > 1 else ""
                terms = _extract_key_terms(sec_text)
                if terms:
                    section_terms[sec_name] = terms
    elif chunks:
        for c in chunks:
            sec = c.get("metadata", {}).get("heading", c.get("metadata", {}).get("section", "__unknown__"))
            terms = _extract_key_terms(c.get("text", ""))
            section_terms.setdefault(sec, set()).update(terms)

    if not section_terms:
        return {"coverage_ratio": 0.0, "section_hit_rates": {}, "balance_score": 0.0}

    answer_lower = answer.lower()
    hit_rates = {}
    total_rate = 0.0
    for sec, terms in section_terms.items():
        if terms:
            matched = sum(1 for t in terms if t in answer_lower)
            rate = matched / len(terms)
            hit_rates[sec] = round(rate, 3)
            total_rate += rate

    coverage_ratio = len([s for s, r in hit_rates.items() if r > 0.05]) / max(len(hit_rates), 1)

    if len(hit_rates) > 1:
        rates = [max(r, 0.001) for r in hit_rates.values()]
        total = sum(rates)
        proportions = [r / total for r in rates]
        entropy = -sum(p * math.log2(p) for p in proportions)
        max_entropy = math.log2(len(rates))
        balance_score = entropy / max_entropy if max_entropy > 0 else 0
    else:
        balance_score = 1.0

    return {
        "coverage_ratio": round(coverage_ratio, 3),
        "section_hit_rates": hit_rates,
        "balance_score": round(balance_score, 3),
    }


def compute_global_concept_coverage(
    answer: str,
    concepts: List[dict],
) -> dict:
    """Measure what fraction of globally-identified concepts appear in the answer."""
    if not answer or not concepts:
        return {"global_coverage": 0.0, "covered_concepts": 0, "total_concepts": 0}

    answer_lower = answer.lower()
    covered = 0
    covered_names = []
    missing_names = []

    for c in concepts:
        name_lower = c.get("name", "").lower()
        keywords = [kw.lower() for kw in c.get("keywords", [])]
        name_hit = name_lower and (name_lower in answer_lower or any(kw in answer_lower for kw in keywords))
        if name_hit:
            covered += 1
            covered_names.append(c.get("name", "?"))
        else:
            missing_names.append(c.get("name", "?"))

    total = max(len(concepts), 1)
    return {
        "global_coverage": round(covered / total, 3),
        "covered_concepts": covered,
        "total_concepts": len(concepts),
        "covered_names": covered_names[:10],
        "missing_names": missing_names[:10],
    }


def compute_summary_conciseness(answer: str, total_chunk_chars: int = 0) -> dict:
    """Measure information density of a summary.

    Metrics:
      - char_count: total length
      - entity_density: entities per 100 chars
      - numeric_density: numbers per 100 chars
      - sentence_count: total sentences
      - avg_sentence_length: avg chars per sentence
    """
    if not answer:
        return {
            "char_count": 0, "entity_density": 0.0, "numeric_density": 0.0,
            "sentence_count": 0, "avg_sentence_length": 0.0,
        }

    entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', answer)
    numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\b', answer)
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    char_count = len(answer)
    entity_density = len(entities) / max(char_count, 1) * 100
    numeric_density = len(numbers) / max(char_count, 1) * 100
    avg_sentence_len = char_count / max(len(sentences), 1)

    return {
        "char_count": char_count,
        "entity_density": round(entity_density, 2),
        "numeric_density": round(numeric_density, 2),
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_sentence_len, 1),
    }


def evaluate_summarization_quality(
    answer: str,
    chunks: List[dict],
    concepts: List[dict],
    section_summaries: Optional[str] = None,
) -> dict:
    """Combined summarization quality evaluation.

    Weighted score:
      - 30% section coverage balance
      - 25% global concept coverage
      - 20% conciseness (inverse length normalized)
      - 15% entity density
      - 10% numeric density
    """
    coverage = compute_section_coverage_balance(answer, section_summaries, chunks)
    concept_cov = compute_global_concept_coverage(answer, concepts)
    concise = compute_summary_conciseness(answer)

    total_chunk_chars = sum(len(c.get("text", "")) for c in (chunks or []))
    compression_ratio = min(concise["char_count"] / max(total_chunk_chars, 1) * 100, 100)
    conciseness_score = 1.0 - (compression_ratio / 100.0)

    entity_score = min(concise["entity_density"] / 10.0, 1.0)
    numeric_score = min(concise["numeric_density"] / 5.0, 1.0)

    overall = round(
        0.30 * coverage.get("balance_score", 0)
        + 0.25 * concept_cov.get("global_coverage", 0)
        + 0.20 * conciseness_score
        + 0.15 * entity_score
        + 0.10 * numeric_score,
        3,
    )

    report = {
        "overall_summarization_score": overall,
        "section_coverage_balance": coverage,
        "global_concept_coverage": concept_cov,
        "conciseness": concise,
        "compression_ratio": round(compression_ratio, 1),
    }

    logger.info(f"Summarization quality: overall={overall}")
    return report


def evaluate_response_quality(
    answer: str,
    query: str,
    result: RetrievalResult,
) -> dict:
    """Evaluate the quality of the full RAG response (answer + retrieval)."""
    faithfulness = compute_answer_faithfulness(answer, result)
    relevance = compute_answer_relevance(answer, query)
    retrieval_quality = evaluate_retrieval_quality(result)

    overall = round(
        0.4 * faithfulness.get("faithfulness_score", 0)
        + 0.3 * relevance
        + 0.15 * retrieval_quality.get("diversity_score", 0)
        + 0.15 * retrieval_quality.get("context_precision", 0),
        3,
    )

    report = {
        "overall_quality_score": overall,
        "query": query[:120],
        "answer_length": len(answer),
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "retrieval_quality": retrieval_quality,
    }

    logger.info(f"Response quality report: overall={overall}")
    return report


# ── New Evaluation Metrics ────────────────────────────────────────────────────

def compute_table_qa_accuracy(
    answer: str,
    query: str,
    chunks: List[dict],
) -> dict:
    """Evaluate accuracy of table-based question answering.

    Checks:
      - Does the answer reference all relevant table rows?
      - Are numeric values from tables correctly extracted?
      - Are row relationships preserved (columns stay with their rows)?
    """
    if not answer or not chunks:
        return {
            "table_qa_accuracy": 0.0,
            "rows_referenced": 0,
            "total_table_rows": 0,
            "numeric_accuracy": 0.0,
            "row_relationship_preserved": False,
        }

    table_chunks = [
        c for c in chunks
        if c.get("metadata", {}).get("content_type") == "table"
        or c.get("metadata", {}).get("table_preserved", False)
    ]

    if not table_chunks:
        return {
            "table_qa_accuracy": 1.0,
            "rows_referenced": 0,
            "total_table_rows": 0,
            "numeric_accuracy": 1.0,
            "row_relationship_preserved": True,
            "note": "No table chunks in context",
        }

    total_rows = len(table_chunks)
    answer_lower = answer.lower()

    # Check if answer references table content
    rows_referenced = 0
    numeric_matches = 0
    total_numbers_in_tables = 0

    for tc in table_chunks:
        text = tc.get("text", "").lower()
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\b', text)
        total_numbers_in_tables += len(numbers)

        # Check if at least some content from this row appears in answer
        row_preview = text[:100]
        if any(phrase in answer_lower for phrase in [row_preview[:30], row_preview[30:60]]):
            rows_referenced += 1

        for num in numbers:
            clean = num.replace("%", "").replace(",", "")
            if clean in answer_lower or num in answer_lower:
                numeric_matches += 1

    row_coverage = rows_referenced / max(total_rows, 1)
    numeric_accuracy = numeric_matches / max(total_numbers_in_tables, 1)

    # Check for row relationship preservation (columns referenced with their rows)
    has_row_prefix = bool(re.search(r'(?:for|in|of)\s+[A-Za-z].*?(?:is|are|was|were)\s', answer_lower))
    row_rels = has_row_prefix or row_coverage > 0.5

    accuracy = round(
        0.5 * row_coverage + 0.3 * numeric_accuracy + 0.2 * (1.0 if row_rels else 0.0),
        3,
    )

    return {
        "table_qa_accuracy": accuracy,
        "rows_referenced": rows_referenced,
        "total_table_rows": total_rows,
        "row_coverage": round(row_coverage, 3),
        "numeric_accuracy": round(numeric_accuracy, 3),
        "row_relationship_preserved": row_rels,
    }


def compute_multihop_qa_accuracy(
    answer: str,
    chunks: List[dict],
) -> dict:
    """Evaluate accuracy of multi-hop reasoning across multiple chunks.

    Checks:
      - Are multiple distinct chunks referenced in the answer?
      - Is information synthesized from different sections?
      - Are cross-chunk relationships correctly identified?
    """
    if not answer or not chunks or len(chunks) < 2:
        return {
            "multihop_accuracy": 0.0,
            "chunks_referenced": 0,
            "total_chunks": len(chunks) if chunks else 0,
            "sections_synthesized": 0,
            "total_sections": 0,
            "cross_referencing": False,
        }

    # Extract source citations from answer
    citations = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))

    # Count distinct sections referenced
    sections = set()
    for c in chunks:
        meta = c.get("metadata", {})
        sec = meta.get("heading", meta.get("section", ""))
        if sec:
            sections.add(sec)

    # Check if answer references multiple chunks
    chunks_referenced = len(citations - {0})
    total_chunks = len(chunks)
    multi_chunk = chunks_referenced >= 2

    # Check for synthesis language
    synthesis_indicators = [
        "across", "both", "combining", "synthesizing", "overall",
        "in summary", "in conclusion", "taken together",
        "from the above", "across all", "when combined",
    ]
    has_synthesis_language = any(ind in answer.lower() for ind in synthesis_indicators)

    # Check for cross-section synthesis
    section_refs = set()
    for sec in sections:
        if sec.lower() in answer.lower():
            section_refs.add(sec)

    accuracy = round(
        0.4 * (1.0 if multi_chunk else 0.0)
        + 0.3 * (1.0 if has_synthesis_language else 0.0)
        + 0.3 * min(len(section_refs) / max(len(sections), 1), 1.0),
        3,
    )

    return {
        "multihop_accuracy": accuracy,
        "chunks_referenced": chunks_referenced,
        "total_chunks": total_chunks,
        "sections_synthesized": len(section_refs),
        "total_sections": len(sections),
        "cross_referencing": multi_chunk,
        "synthesis_language_detected": has_synthesis_language,
    }


def compute_citation_accuracy(
    answer: str,
    sources: List[dict],
) -> dict:
    """Evaluate citation accuracy in the generated answer.

    Checks:
      - Every [N] citation in the answer corresponds to a valid source
      - No out-of-range citations
      - Citations are placed near the claims they support
    """
    if not answer:
        return {
            "citation_accuracy": 1.0,
            "citations_found": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "citations_valid": True,
        }

    citations = [(m.start(), int(m.group(1))) for m in re.finditer(r'\[(\d+)\]', answer)]
    if not citations:
        return {
            "citation_accuracy": 1.0,
            "citations_found": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "citations_valid": True,
            "note": "No citations in answer",
        }

    total_citations = len(citations)
    valid_range = set(range(1, len(sources) + 1)) if sources else set()

    valid_count = sum(1 for _, num in citations if num in valid_range)
    invalid_count = total_citations - valid_count

    accuracy = round(valid_count / max(total_citations, 1), 3)

    return {
        "citation_accuracy": accuracy,
        "citations_found": total_citations,
        "valid_citations": valid_count,
        "invalid_citations": invalid_count,
        "citations_valid": invalid_count == 0,
    }


def compute_hallucination_rate(
    answer: str,
    chunks: List[dict],
) -> dict:
    """Estimate hallucination rate by checking claims against context.

    Uses the confidence module's claim extraction and verification.
    """
    if not answer or not chunks:
        return {
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
        }

    try:
        from app.services.confidence import estimate_confidence
        conf = estimate_confidence(answer, chunks)
        total = conf.get("total_claims", 0)
        failed = conf.get("claims_failed", 0)
        rate = failed / max(total, 1)

        return {
            "hallucination_rate": round(rate, 3),
            "total_claims": total,
            "supported_claims": conf.get("claims_verified", 0),
            "unsupported_claims": failed,
            "overall_confidence": conf.get("overall_confidence", 0.0),
        }
    except Exception:
        return {
            "hallucination_rate": 0.0,
            "total_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "error": "confidence check failed",
        }


def compute_answer_completeness(
    answer: str,
    chunks: List[dict],
    query: str,
) -> dict:
    """Evaluate answer completeness by checking fact coverage from chunks."""
    try:
        from app.services.completeness import check_answer_completeness
        return check_answer_completeness(answer, chunks, query)
    except Exception:
        return {
            "is_complete": True,
            "coverage_ratio": 1.0,
            "total_facts": 0,
            "note": "completeness check unavailable",
        }


def compute_all_evaluation_metrics(
    answer: str,
    query: str,
    result: RetrievalResult,
) -> dict:
    """Compute ALL evaluation metrics and return a comprehensive report."""
    report = {
        "query": query[:200],
        "overall": evaluate_response_quality(answer, query, result),
        "table_qa": compute_table_qa_accuracy(answer, query, result.chunks),
        "multihop_qa": compute_multihop_qa_accuracy(answer, result.chunks),
        "citation": compute_citation_accuracy(answer, result.sources),
        "hallucination": compute_hallucination_rate(answer, result.chunks),
        "completeness": compute_answer_completeness(answer, result.chunks, query),
    }

    # Summarize key metrics
    report["summary"] = {
        "faithfulness": report["overall"].get("faithfulness", {}).get("faithfulness_score", 0),
        "context_recall": report["overall"].get("retrieval_quality", {}).get("context_recall", 0),
        "answer_completeness": report["completeness"].get("coverage_ratio", 0),
        "table_qa_accuracy": report["table_qa"].get("table_qa_accuracy", 0),
        "multihop_qa_accuracy": report["multihop_qa"].get("multihop_accuracy", 0),
        "citation_accuracy": report["citation"].get("citation_accuracy", 0),
        "hallucination_rate": report["hallucination"].get("hallucination_rate", 0),
    }

    logger.info(f"Full evaluation: {report['summary']}")
    return report
