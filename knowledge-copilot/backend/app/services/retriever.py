"""
retriever.py — Multi-stage RAG pipeline with hybrid search, query expansion,
MMR diversity, cross-encoder reranking, section + source balancing, and metadata-aware retrieval.

Pipeline:
  1. Query expansion (optional) with domain-term injection
  2. Hybrid search (BM25 + semantic) with MMR diversity
  3. Score threshold filter
  4. Deduplication (by normalized text)
  5. Cross-encoder reranking
  6. Source diversity enforcement (balance across documents)
  7. Section diversity enforcement (balance across headings)
  8. Context assembly with table-aware formatting
  9. Source tracking & metrics
"""

import logging
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Set

from app.services.vector_store import get_vector_store
from app.services.embedder import embed_query
from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.retriever")


# ── Domain term expansion map ──────────────────────────────────────────────────
# Maps high-level query concepts to concrete lexical terms that may appear in docs
# but are semantically distant from the query embedding.  BM25 catches these.

DOMAIN_TERM_MAP: dict[str, list[str]] = {
    # Financial / Revenue
    "revenue":       ["arr", "mrr", "pricing", "subscription", "enterprise pricing",
                       "annual recurring", "monthly recurring", "revenue model",
                       "revenue growth", "top line", "monetization", "license revenue",
                       "revenue stream"],
    "pricing":       ["price", "cost", "subscription fee", "enterprise pricing",
                       "per-seat", "pricing model", "tier", "plan", "billing",
                       "annual", "monthly", "arr", "mrr", "price point"],
    "subscription":  ["subscriptions", "subscribers", "recurring", "saas",
                       "membership", "plan", "tier", "billing cycle",
                       "annual renew", "monthly fee", "license"],
    "financial":     ["revenue", "profit", "cost", "margin", "expense", "budget",
                       "forecast", "financial statement", "balance sheet", "income",
                       "arr", "mrr", "ebitda", "valuation", "funding"],

    # Privacy / Compliance / Governance
    "privacy":       ["gdpr", "hipaa", "data protection", "pii", "personal data",
                       "consent", "data governance", "privacy policy", "data subject",
                       "soc2", "data privacy", "privacy regulation", "ccpa"],
    "compliance":    ["regulatory", "regulation", "standard", "certification", "audit",
                       "gdpr", "hipaa", "soc2", "iso", "data protection", "privacy law",
                       "compliance framework", "regulatory requirement"],
    "governance":    ["ethics", "compliance", "regulatory", "audit", "policy",
                       "responsible ai", "oversight", "board", "framework",
                       "corporate governance", "data governance"],
    "ethics":        ["responsible ai", "ethical", "bias", "fairness", "transparency",
                       "accountability", "safety", "alignment", "ethical ai",
                       "ai ethics", "trustworthy"],

    # Security
    "security":      ["api security", "authentication", "authorization", "encryption",
                       "access control", "vulnerability", "threat", "compliance",
                       "zero trust", "iam", "cybersecurity", "risk assessment",
                       "security posture", "data breach"],

    # Cost / Economics
    "cost":          ["pricing", "total cost", "tco", "roi", "break-even",
                       "payback period", "cost analysis", "budget", "expense",
                       "cost saving", "cost efficiency", "operating cost"],

    # Market / Competitive
    "market":        ["market share", "market size", "competitor", "competitive",
                       "industry analysis", "segment", "tam", "sam", "som",
                       "market position", "market trend", "market landscape"],

    # Performance
    "performance":   ["speed", "latency", "throughput", "benchmark", "efficiency",
                       "response time", "qps", "scalability", "performance metric",
                       "optimization", "bottleneck"],

    # Enterprise / Business
    "enterprise":    ["enterprise pricing", "business plan", "corporate", "organization",
                       "company-wide", "deployment", "integration", "business model",
                       "go-to-market", "business strategy"],

    # Comparison / Synthesis
    "comparison":    ["compare", "comparison", "vs", "versus", "compared to",
                       "differences", "better than", "pros and cons", "advantages",
                       "disadvantages", "alternative", "trade-off", "differentiate"],

    # Architecture / Technical
    "architecture":  ["system design", "infrastructure", "data pipeline", "microservices",
                       "api", "database", "deployment", "cloud architecture",
                       "technical stack", "architecture pattern"],

    # AI / ML
    "ai":            ["machine learning", "deep learning", "llm", "neural network",
                       "training", "inference", "model", "algorithm", "artificial intelligence",
                       "nlp", "transformer", "embedding"],

    # Growth / Metrics
    "growth":        ["growth rate", "adoption", "user base", "customer acquisition",
                       "retention", "churn", "expansion", "scaling", "market penetration"],
}


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

def _inject_domain_terms(query: str) -> List[str]:
    """Generate per-domain query variants by injecting domain-specific lexical synonyms.

    Each matched domain produces its own variant with only its terms, giving BM25
    better precision per variant (e.g., "privacy + governance" query produces two
    variants instead of one big combined variant).

    Matching is multi-strategy:
      1. Check if a domain KEY appears in the query (e.g., "privacy" in "data privacy")
      2. Check if any domain TERM appears in the query (e.g., "GDPR" in "GDPR requirements")
      3. Check if any multi-word KEY appears as a phrase in the query
    """
    q_lower = query.lower()
    matched_domains: dict[str, set[str]] = {}

    def _match_domain(domain: str, terms: list[str]):
        """Record a domain match, merging terms."""
        if domain not in matched_domains:
            matched_domains[domain] = set()
        matched_domains[domain].update(terms)

    for domain, terms in DOMAIN_TERM_MAP.items():
        # Strategy 1 — domain key appears as a whole word or phrase in query
        domain_words = domain.split()
        if any(dw in q_lower for dw in domain_words):
            _match_domain(domain, terms)
            continue

        # Strategy 2 — any domain term appears in query
        for term in terms:
            if term in q_lower:
                _match_domain(domain, terms)
                break

    if not matched_domains:
        return []

    # Build ONE variant per matched domain (not one giant combined variant).
    # Cap at 5 domain variants to avoid search fragmentation.
    MAX_DOMAIN_VARIANTS = 5
    domains = sorted(matched_domains.keys())[:MAX_DOMAIN_VARIANTS]

    variants: List[str] = []
    for domain in domains:
        extra = sorted(matched_domains[domain])
        variant = f"{query} {' '.join(extra)}"
        variants.append(variant)

    logger.info(
        f"Domain term expansion: {len(variants)} variants "
        f"[{', '.join(domains)}]"
    )
    for i, v in enumerate(variants):
        logger.debug(f"  Variant [{i}]: {v[:120]}")
    return variants


def _expand_query(query: str) -> List[str]:
    """Generate variant queries to improve recall for entity names, synonyms, metrics.

    Two strategies:
      1. Domain-term injection (fixed synonym map) — catches lexical gaps like
         "privacy" → "GDPR", "revenue" → "ARR pricing subscription"
      2. LLM-based rephrasing — generates contextual variants
    """
    queries: List[str] = [query]

    # Strategy 1 — Domain term injection (lexical, no LLM call needed)
    if settings.query_expansion_domain_terms:
        domain_variants = _inject_domain_terms(query)
        queries.extend(domain_variants)

    # Strategy 2 — LLM-based rephrasing
    if settings.query_expansion_enabled:
        try:
            from app.services.llm import get_llm
            llm = get_llm()

            domain_hint = ""
            if settings.query_expansion_domain_terms:
                matched = [d for d in DOMAIN_TERM_MAP if d in query.lower()]
                if matched:
                    domain_hint = (
                        f"\nThe question involves these domains: {', '.join(matched)}. "
                        f"Use synonyms and related terminology for these domains."
                    )

            prompt = (
                f"Given the user question below, rewrite it into up to {settings.query_expansion_max_terms - 1} "
                f"alternative phrasings that would help find relevant information in a knowledge base. "
                f"Use synonyms, rephrase numeric/metric terms, expand acronyms, and add related concepts."
                f"{domain_hint}"
                f"\nReturn each variant on a separate line. Do NOT include the original question.\n\n"
                f"Question: {query}"
            )

            response = llm.invoke(prompt)
            variants = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
            variants = variants[:settings.query_expansion_max_terms - 1]

            # Filter out lines that look like meta-commentary
            variants = [v for v in variants if not v.lower().startswith(("here", "sure", "option", "variant"))]
            queries.extend(variants)

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")

    # Cap total variants (original + domain + LLM) to limit search fragmentation
    MAX_TOTAL_VARIANTS = 8
    queries = queries[:MAX_TOTAL_VARIANTS]

    # Deduplicate while preserving order
    seen: Set[str] = set()
    unique: List[str] = []
    for q in queries:
        key = q.strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(q)

    logger.info(f"Query expansion: {len(unique)} queries (original + {len(unique) - 1} variants)")
    for i, q in enumerate(unique):
        logger.debug(f"  Query [{i}]: {q[:120]}")

    return unique


def _search_with_expansion(
    query: str,
    k: int,
    fetch_k: int,
    mmr_lambda: float,
    use_hybrid: bool = True,
) -> tuple[List[dict], List[str]]:
    """Search with query expansion, merging results from all variants.

    When use_hybrid is True and the store supports it, uses BM25 + vector
    hybrid search.  Otherwise falls back to pure vector MMR search.
    """
    queries = _expand_query(query)
    store = get_vector_store()
    has_hybrid = settings.retrieval_hybrid_search and hasattr(store, "search_hybrid")

    all_results: List[dict] = []
    seen_texts: set = set()

    for q in queries:
        if has_hybrid:
            raw = store.search_hybrid(
                q,
                k=fetch_k,
                fetch_k=fetch_k * 2,
                mmr_lambda=mmr_lambda,
                alpha=settings.retrieval_hybrid_alpha,
            )
        elif hasattr(store, "search_mmr"):
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

    logger.info(
        f"Search: {len(queries)} queries, {len(all_results)} unique candidates "
        f"({'hybrid' if has_hybrid else 'vector'})"
    )
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

def _get_source_key(chunk: dict) -> str:
    return (
        chunk.get("metadata", {}).get("file_name", "")
        or chunk.get("metadata", {}).get("source", "")
        or "__unknown__"
    )

def _get_section_key(chunk: dict) -> str:
    return (
        chunk.get("metadata", {}).get("heading", "")
        or chunk.get("metadata", {}).get("section", "")
        or "__prose__"
    )


def _cap_per_source(chunks: List[dict], max_per_doc: int) -> List[dict]:
    """Apply max-per-doc cap; keeps top-ranked chunks per source."""
    counts: dict[str, int] = {}
    capped: List[dict] = []
    for c in chunks:
        src = _get_source_key(c)
        counts[src] = counts.get(src, 0) + 1
        if counts[src] <= max_per_doc:
            capped.append(c)
    return capped


def _cap_per_section(chunks: List[dict], max_per_section: int) -> List[dict]:
    """Apply max-per-section cap; keeps top-ranked chunks per section."""
    counts: dict[str, int] = {}
    capped: List[dict] = []
    for c in chunks:
        sec = _get_section_key(c)
        counts[sec] = counts.get(sec, 0) + 1
        if counts[sec] <= max_per_section:
            capped.append(c)
    return capped


def _enforce_source_diversity(
    chunks: List[dict],
    k: int,
    min_sources: int,
    max_per_doc: int,
) -> List[dict]:
    """Ensure the final set of chunks spans multiple documents/sources.

    1. Always enforce max_per_doc cap (prevents any single source from dominating)
    2. If minimum sources not met, round-robin promote underrepresented sources
    """
    if not settings.retrieval_source_balancing:
        return chunks[:k]

    # Step 1 — Always cap chunks per source (most impactful fix)
    capped = _cap_per_source(chunks, max_per_doc)

    # Step 2 — Check if min sources met
    sources_present = set(_get_source_key(c) for c in capped)
    if len(sources_present) >= min_sources:
        return capped[:k]

    logger.info(
        f"Source diversity: only {len(sources_present)} sources (need {min_sources}), "
        f"max {max_per_doc} per doc — promoting underrepresented"
    )

    # Step 3 — Round-robin promote underrepresented sources
    source_groups: dict[str, list] = {}
    for c in chunks:
        src = _get_source_key(c)
        source_groups.setdefault(src, []).append(c)

    source_names = list(source_groups.keys())
    selected: List[dict] = []
    counts: dict[str, int] = {s: 0 for s in source_names}

    pool = {s: iter(g) for s, g in source_groups.items()}
    exhausted = set()

    while len(selected) < k:
        added = False
        for src in source_names:
            if src in exhausted:
                continue
            if counts[src] >= max_per_doc:
                continue
            try:
                chunk = next(pool[src])
                selected.append(chunk)
                counts[src] += 1
                added = True
                if len(selected) >= k:
                    break
            except StopIteration:
                exhausted.add(src)
        if not added:
            break

    return selected[:k]


# ── Section diversity enforcement ──────────────────────────────────────────

def _enforce_section_diversity(
    chunks: List[dict],
    k: int,
    min_sections: int,
    max_per_section: int,
) -> List[dict]:
    """Ensure the final set of chunks spans multiple sections.

    1. Always enforce max_per_section cap
    2. If minimum sections not met, round-robin promote underrepresented sections
    """
    if not settings.retrieval_section_diversity:
        return chunks[:k]

    # Step 1 — Always cap chunks per section
    capped = _cap_per_section(chunks, max_per_section)

    # Step 2 — Check if min sections met
    sections_present = set(_get_section_key(c) for c in capped)
    if len(sections_present) >= min_sections:
        return capped[:k]

    logger.info(
        f"Section diversity: only {len(sections_present)} sections (need {min_sections}), "
        f"max {max_per_section} per section — promoting underrepresented"
    )

    # Step 3 — Round-robin promote underrepresented sections
    section_groups: dict[str, list] = {}
    for c in chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    section_names = list(section_groups.keys())
    selected: List[dict] = []
    counts: dict[str, int] = {s: 0 for s in section_names}

    pool = {s: iter(g) for s, g in section_groups.items()}
    exhausted = set()

    while len(selected) < k:
        added = False
        for sec in section_names:
            if sec in exhausted:
                continue
            if counts[sec] >= max_per_section:
                continue
            try:
                chunk = next(pool[sec])
                selected.append(chunk)
                counts[sec] += 1
                added = True
                if len(selected) >= k:
                    break
            except StopIteration:
                exhausted.add(sec)
        if not added:
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

    1. Query expansion — domain-term injection + LLM rephrasing
    2. Hybrid search (BM25 + semantic) with MMR diversity  ← NEW
    3. Score threshold filter
    4. Deduplication (by normalized text)
    5. Cross-encoder reranking
    6. Source diversity enforcement (balance across documents)  ← NEW
    7. Section diversity enforcement (balance across headings)
    8. Context assembly with document-grouped formatting  ← IMPROVED
    9. Source tracking & metrics
    """
    k                 = k                 or settings.retrieval_k
    fetch_k           = settings.retrieval_fetch_k
    score_threshold   = score_threshold   or settings.retrieval_score_threshold
    max_context_chars = max_context_chars or settings.retrieval_max_context_chars
    mmr_lambda        = mmr_lambda        or settings.retrieval_mmr_lambda

    # ── 1. Query expansion + hybrid search ──────────────────────────────────
    raw_results, expanded_queries = _search_with_expansion(query, k, fetch_k, mmr_lambda)

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Retrieved {len(raw_results)} raw candidates for query: '{query[:120]}'")
        for i, r in enumerate(raw_results[:10]):
            score_info = f"score={r.get('score', 0):.4f}"
            bm25_info = f" bm25={r.get('bm25_score', 0):.4f}" if "bm25_score" in r else ""
            source_info = r.get("_source", "")
            if settings.eval_log_scores:
                section = r.get("metadata", {}).get("heading", "") or r.get("metadata", {}).get("section", "")
                source = r.get("metadata", {}).get("file_name", r.get("metadata", {}).get("source", ""))
                logger.info(f"  [{i+1}] {score_info}{bm25_info} | src='{source}' | section='{section}' | preview={r['text'][:80]}")

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
        reranked = reranked[:k * 2]

    if settings.eval_log_reranking:
        logger.info(f"After reranking: top {len(reranked)} chunks")

    # ── 5. Source diversity ───────────────────────────────────────────────
    source_balanced = _enforce_source_diversity(
        reranked, k,
        settings.retrieval_min_sources,
        settings.retrieval_max_chunks_per_doc,
    )
    if settings.eval_log_retrieved_chunks:
        sources_in_balanced = set(
            _get_source_key(c) for c in source_balanced
        )
        counts_in_balanced = {}
        for c in source_balanced:
            s = _get_source_key(c)
            counts_in_balanced[s] = counts_in_balanced.get(s, 0) + 1
        logger.info(
            f"After source balancing: {len(sources_in_balanced)} sources in "
            f"top {len(source_balanced)} chunks — per-source: {counts_in_balanced}"
        )

    # ── 6. Section diversity ──────────────────────────────────────────────
    diversified = _enforce_section_diversity(
        source_balanced, k,
        settings.retrieval_min_sections,
        settings.retrieval_max_chunks_per_section,
    )

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Final {len(diversified)} chunks sent to LLM:")
        for i, c in enumerate(diversified):
            section = _get_section_key(c)
            source = _get_source_key(c)
            score = c.get("rerank_score", c.get("score", 0))
            logger.info(f"  [{i+1}] score={score:.4f} | src='{source}' | section='{section}' | len={len(c['text'])} | preview={c['text'][:80]}")

    # ── 7. Build context with document-grouped formatting ─────────────────
    # Group chunks by source document for cross-document awareness
    doc_groups: dict[str, list[dict]] = {}
    for chunk in diversified:
        src = _get_source_key(chunk)
        doc_groups.setdefault(src, []).append(chunk)

    context_parts: List[str] = []
    sources: List[SourceReference] = []
    chars_used = 0
    source_number = 1

    for doc_name in sorted(doc_groups.keys(), key=lambda x: x.lower()):
        group_chunks = doc_groups[doc_name]
        for chunk in group_chunks:
            text = chunk["text"].strip()
            meta = chunk.get("metadata", {})

            # Build label with source + optional section/table annotation
            section_label = meta.get("heading", meta.get("section", ""))
            if meta.get("content_type") == "table":
                table_name = meta.get("table_name", "")
                annotations = " — ".join(filter(None, [section_label, table_name]))
                if annotations:
                    label = f"[{source_number}] ({src} — {annotations})"
                else:
                    label = f"[{source_number}] ({src})" if src else f"[{source_number}]"
            else:
                label = f"[{source_number}] ({src})" if src else f"[{source_number}]"

            entry = f"{label} {text}"
            if chars_used + len(entry) + 2 > max_context_chars:
                break

            context_parts.append(entry)
            sources.append(SourceReference(
                file_name     = meta.get("file_name", meta.get("source", "unknown")),
                chunk_index   = meta.get("chunk_index", -1),
                page          = meta.get("page"),
                score         = round(chunk.get("rerank_score", chunk.get("score", 0)), 4),
                rerank_score  = round(chunk.get("rerank_score", 0), 4),
                preview       = text[:120],
                content_type  = meta.get("content_type", "prose"),
                section       = meta.get("heading", meta.get("section", "")),
                table_name    = meta.get("table_name", ""),
                source_number = source_number,
            ))
            chars_used += len(entry) + 2
            source_number += 1

        if chars_used >= max_context_chars:
            break

    context = "\n\n".join(context_parts)

    # ── 9. Retrieval metrics ──────────────────────────────────────────────
    sections_found = set()
    sources_found = set()
    content_type_counts = {}
    for c in diversified:
        sec = _get_section_key(c)
        if sec != "__prose__":
            sections_found.add(sec)
        src = _get_source_key(c)
        if src != "__unknown__":
            sources_found.add(src)
        ct = c.get("metadata", {}).get("content_type", "prose")
        content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

    # Per-source/per-section counts for diagnostics
    per_source_counts = {}
    per_section_counts = {}
    for c in diversified:
        s = _get_source_key(c)
        per_source_counts[s] = per_source_counts.get(s, 0) + 1
        sec = _get_section_key(c)
        per_section_counts[sec] = per_section_counts.get(sec, 0) + 1

    metrics = {
        "total_candidates":    len(raw_results),
        "after_filter":        len(filtered),
        "after_dedup":         len(unique),
        "after_rerank":        len(reranked),
        "final_chunks":        len(diversified),
        "sections_covered":    len(sections_found),
        "sources_covered":     len(sources_found),
        "content_types":       content_type_counts,
        "expanded_queries":    len(expanded_queries),
        "hybrid_search":       settings.retrieval_hybrid_search,
        "source_balancing":    settings.retrieval_source_balancing,
        "per_source":          per_source_counts,
        "per_section":         per_section_counts,
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
    Format retrieved context for the LLM prompt with cross-document and
    multi-section synthesis support.
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
    multi_source = len(set(s.file_name for s in result.sources if s.file_name)) > 1

    hints = []
    if has_tables:
        hints.append("Some context chunks contain TABLE data. Pay close attention to all numeric values, percentages, and row-level data.")
    if multi_section:
        hints.append("The context spans MULTIPLE SECTIONS. You may need to combine information across different sections to fully answer the question.")
    if multi_source:
        hints.append("The context spans MULTIPLE DOCUMENTS. You MUST aggregate and synthesize information across ALL documents. Compare and contrast evidence from each source. Look for similarities, differences, and complementary information. If different documents present different data points on the same topic, include ALL of them with their source numbers.")
    if multi_source and multi_section:
        hints.append("This is a CROSS-DOCUMENT, CROSS-SECTION query. For the best answer, look for complementary information across sources and sections before concluding an answer. Synthesize partial evidence from multiple chunks.")

    hint_str = "\n".join(f"Note: {h}" for h in hints)

    return (
        f"Use ONLY the context below to answer the question. "
        f"If the answer is not in the context, say so clearly."
        f"{chr(10) + hint_str if hints else ''}\n\n"
        f"SOURCES:\n{source_str}\n\n"
        f"CONTEXT:\n{result.context}"
    )
