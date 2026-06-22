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
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Set

from app.services.vector_store import get_vector_store
from app.services.embedder import embed_query
from app.core.config import settings
from qdrant_client.http import models as qdrant_models

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
    expanded:      bool = False


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
    variants instead of one giant combined variant).

    Matching is multi-strategy:
      1. Check if a domain KEY appears in the query (e.g., "privacy" in "data privacy")
         → inject ALL domain terms (full synonym expansion)
      2. Check if any domain TERM appears in the query (e.g., "GDPR" in "GDPR requirements")
         → inject ONLY the matching term(s), NOT all domain terms.
           This prevents generic/technical words like "api" from pulling in
           the entire architecture domain term list.
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
        # Inject ALL domain terms (desired: full synonym expansion)
        domain_words = domain.split()
        if any(dw in q_lower for dw in domain_words):
            _match_domain(domain, terms)
            continue

        # Strategy 2 — any domain term appears in query
        # Inject ONLY the specific matching term(s), not all domain terms.
        # This avoids pollution from short/generic terms like "api".
        matched_terms = [term for term in terms if term in q_lower]
        if matched_terms:
            _match_domain(domain, matched_terms)

    if not matched_domains:
        return []

    # Build ONE variant per matched domain (not one giant combined variant).
    # Cap at 5 domain variants to avoid search fragmentation.
    # Also cap total variant length to 300 chars to avoid BM25 pollution.
    MAX_DOMAIN_VARIANTS = 5
    MAX_VARIANT_LENGTH = 300
    domains = sorted(matched_domains.keys())[:MAX_DOMAIN_VARIANTS]

    variants: List[str] = []
    for domain in domains:
        extra = sorted(matched_domains[domain])
        injection = ' '.join(extra)
        if len(injection) > MAX_VARIANT_LENGTH:
            injection = injection[:MAX_VARIANT_LENGTH]
        variant = f"{query} {injection}"
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
    filter_condition: Optional[qdrant_models.Filter] = None,
) -> tuple[List[dict], List[str]]:
    """Search with query expansion, merging results from all variants.

    When use_hybrid is True and the store supports it, uses BM25 + vector
    hybrid search.  Otherwise falls back to pure vector MMR search.

    When filter_condition is provided, it is passed to every Qdrant query_points
    call so only chunks matching the filter (e.g. by document_id or
    conversation_id) are considered during retrieval.
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
                filter_condition=filter_condition,
            )
        elif hasattr(store, "search_mmr"):
            raw = store.search_mmr(q, k=fetch_k, fetch_k=fetch_k * 2, mmr_lambda=mmr_lambda, filter_condition=filter_condition)
        else:
            raw = store.search(q, k=fetch_k, filter_condition=filter_condition)

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

    # Step 3 — Round-robin promote underrepresented sources.
    # Only promote chunks whose score is at least 60% of the best score
    # to avoid filling the context with near-zero-relevance content.
    best_score = max((c.get("rerank_score", c.get("score", 0)) for c in chunks if c), default=0.001)
    QUALITY_FLOOR = best_score * 0.6

    source_groups: dict[str, list] = {}
    for c in chunks:
        src = _get_source_key(c)
        source_groups.setdefault(src, []).append(c)

    source_names = list(source_groups.keys())
    selected: List[dict] = []
    counts: dict[str, int] = {s: 0 for s in source_names}

    pool = {s: [c for c in g if c.get("rerank_score", c.get("score", 0)) >= QUALITY_FLOOR]
            for s, g in source_groups.items()}
    exhausted = set(s for s, g in pool.items() if not g)

    while len(selected) < k:
        added = False
        for src in source_names:
            if src in exhausted:
                continue
            if counts[src] >= max_per_doc:
                continue
            if not pool.get(src):
                exhausted.add(src)
                continue
            chunk = pool[src].pop(0)
            selected.append(chunk)
            counts[src] += 1
            added = True
            if len(selected) >= k:
                break
        if not added:
            break

    sources_in_result = set(_get_source_key(c) for c in selected)
    if len(sources_in_result) < min_sources:
        logger.info(
            f"Source diversity: only {len(sources_in_result)} sources after "
            f"score-filtered promotion (need {min_sources}) — "
            f"returning best available"
        )
    return selected[:k]


# ── Section diversity enforcement ──────────────────────────────────────────

def _enforce_section_diversity(
    chunks: List[dict],
    k: int,
    min_sections: int,
    max_per_section: int,
    full_pool: Optional[List[dict]] = None,
) -> List[dict]:
    """Ensure the final set of chunks spans multiple sections.

    1. Always enforce max_per_section cap
    2. If minimum sections not met, round-robin promote underrepresented sections
       using `full_pool` if provided (to recover sections that source-balancing may
       have pruned).
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

    # Step 3 — Round-robin promote underrepresented sections.
    # Use full_pool when provided (pre-source-balancing) to access sections
    # that were pruned by source caps.
    # Only promote chunks whose score is at least 60% of the best score
    # to avoid filling the context with near-zero-relevance content.
    pool_source = full_pool if full_pool else chunks
    best_score = max((c.get("rerank_score", c.get("score", 0)) for c in pool_source if c), default=0.001)
    QUALITY_FLOOR = best_score * 0.6

    section_groups: dict[str, list] = {}
    for c in pool_source:
        sec = _get_section_key(c)
        if c.get("rerank_score", c.get("score", 0)) >= QUALITY_FLOOR:
            section_groups.setdefault(sec, []).append(c)

    section_names = list(section_groups.keys())
    already_selected = set(id(c) for c in capped)
    selected: List[dict] = list(capped)
    counts: dict[str, int] = {}
    for c in selected:
        sec = _get_section_key(c)
        counts[sec] = counts.get(sec, 0) + 1

    remaining_pool = {s: [c for c in g if id(c) not in already_selected]
                      for s, g in section_groups.items()}
    exhausted = set(s for s, g in remaining_pool.items() if not g)

    while len(selected) < k:
        added = False
        for sec in section_names:
            if sec in exhausted:
                continue
            if counts.get(sec, 0) >= max_per_section:
                continue
            if not remaining_pool.get(sec):
                exhausted.add(sec)
                continue
            chunk = remaining_pool[sec].pop(0)
            selected.append(chunk)
            counts[sec] = counts.get(sec, 0) + 1
            added = True
            if len(selected) >= k:
                break
        if not added:
            break

    return selected[:k]


# ── Advanced deduplication ──────────────────────────────────────────────────

def _tokenize(text: str) -> set:
    return set(re.findall(r'\w+', text.lower()))


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / max(len(union), 1)


def _deduplicate_jaccard(
    chunks: List[dict],
    threshold: float = None,
) -> List[dict]:
    """Remove near-duplicate chunks using Jaccard token-set similarity.

    More accurate than the simple prefix-based dedup because it catches
    chunks with high content overlap even when they start differently.

    Protects:
      - Very short chunks (<= 50 chars) — likely continuation fragments
      - Chunks from the same section with different section_chunk_index
        (continuation chunks from the same logical section)
    """
    threshold = threshold or settings.retrieval_jaccard_threshold
    if not chunks:
        return chunks

    result: List[dict] = []
    kept_entries: List[dict] = []  # each entry: {"tokens": set, "section_id": str, "indices": set}

    for c in chunks:
        text = c.get("text", "")
        tokens = _tokenize(text)
        if not tokens:
            result.append(c)
            continue

        meta = c.get("metadata", {})
        c_sec_id = meta.get("section_id", "")
        c_chunk_idx = meta.get("section_chunk_index", -1)

        # Short-chunk protection: never dedup very short chunks
        if len(text) <= 50:
            result.append(c)
            kept_entries.append({"tokens": tokens, "section_id": c_sec_id, "indices": set()})
            continue

        is_dup = False
        for ke in kept_entries:
            kt = ke["tokens"]
            if len(tokens) < 5 or len(kt) < 5:
                continue
            jaccard = len(tokens & kt) / max(len(tokens | kt), 1)
            if jaccard > threshold:
                # Protection: same-section continuation chunks are NOT duplicates
                if c_sec_id and c_sec_id == ke["section_id"] and c_chunk_idx >= 0:
                    if c_chunk_idx not in ke["indices"]:
                        continue  # not a duplicate — different chunk in same section
                is_dup = True
                break

        if not is_dup:
            result.append(c)
            idx_set = set()
            if c_sec_id:
                idx_set.add(c_chunk_idx)
            kept_entries.append({"tokens": tokens, "section_id": c_sec_id, "indices": idx_set})

    removed = len(chunks) - len(result)
    if removed:
        logger.info(f"Jaccard dedup: removed {removed} near-duplicates (threshold={threshold})")

    return result


# ── Novelty scoring ─────────────────────────────────────────────────────────

def _score_chunk_novelty(
    chunk: dict,
    already_selected: List[dict],
) -> float:
    """Score how much NEW information a chunk adds vs already-selected chunks.

    Returns 1.0 for fully novel, 0.0 for identical to an existing chunk.
    """
    if not already_selected:
        return 1.0

    text = chunk.get("text", "")
    max_overlap = max(
        _jaccard_similarity(text, selected.get("text", ""))
        for selected in already_selected
    )

    if max_overlap >= 0.85:
        return 0.0
    if max_overlap >= 0.7:
        return 0.3
    if max_overlap >= 0.5:
        return 0.6

    return 1.0


def _select_with_novelty(
    chunks: List[dict],
    k: int,
    novelty_lambda: float = None,
) -> List[dict]:
    """Select top-k chunks balancing relevance score with novelty.

    Uses a greedy approach: picks the highest-scoring chunk first,
    then iteratively selects chunks that maximize:
        score * (1 - lambda) + novelty * lambda

    where novelty is measured against already-selected chunks.
    """
    novelty_lambda = novelty_lambda or settings.retrieval_novelty_lambda
    if not settings.retrieval_novelty_scoring or not chunks:
        return chunks[:k]

    if len(chunks) <= k:
        return chunks

    selected: List[dict] = []
    pool = list(chunks)

    first = pool.pop(0)
    selected.append(first)

    while len(selected) < k and pool:
        best_idx = -1
        best_score = -float("inf")

        for i, c in enumerate(pool):
            relevance = c.get("rerank_score", c.get("score", 0))
            novelty = _score_chunk_novelty(c, selected)
            combined = relevance * (1 - novelty_lambda) + novelty * novelty_lambda

            if combined > best_score:
                best_score = combined
                best_idx = i

        if best_idx != -1:
            selected.append(pool.pop(best_idx))
        else:
            break

    logger.info(
        f"Novelty selection: {len(selected)} chunks (λ={novelty_lambda})"
    )
    return selected


# ── Structured retrieval tracing ────────────────────────────────────────────

@dataclass
class TraceEntry:
    stage: str
    duration_ms: float
    input_count: int
    output_count: int
    detail: str = ""


class RetrievalTrace:
    """Collects per-stage timing and counts for observability."""

    def __init__(self, query: str):
        self.query = query
        self.entries: List[TraceEntry] = []
        self._start_time = time.monotonic()

    def stage(self, name: str, input_count: int, output_count: int, detail: str = ""):
        elapsed = (time.monotonic() - self._start_time) * 1000
        self.entries.append(TraceEntry(
            stage=name,
            duration_ms=round(elapsed, 1),
            input_count=input_count,
            output_count=output_count,
            detail=detail,
        ))

    def to_dict(self) -> dict:
        return {
            "query": self.query[:120],
            "total_duration_ms": round(sum(e.duration_ms for e in self.entries), 1),
            "stages": [
                {
                    "stage": e.stage,
                    "duration_ms": e.duration_ms,
                    "input_count": e.input_count,
                    "output_count": e.output_count,
                    "detail": e.detail,
                }
                for e in self.entries
            ],
        }


# ── Adjacent chunk expansion ──────────────────────────────────────────────

def _expand_with_adjacent_chunks(
    chunks:  List[dict],
    window:  int = 1,
) -> List[dict]:
    """Expand retrieved chunks with adjacent siblings from the same section.

    For each retrieved chunk that has a section_id, fetches all sibling chunks
    from the same section (via Qdrant) and includes those whose
    section_chunk_index falls within `window` positions of any retrieved chunk
    in that section.

    This ensures multi-page sections, lists, tables, and framework definitions
    that were split across chunk boundaries are retrieved as a complete context.
    Expansion preserves section boundaries — no cross-section bleed.

    Returns the merged list with expanded chunks interleaved at the correct
    position (ordered by section_chunk_index within each section).
    """
    if not settings.retrieval_chunk_expansion_enabled or not chunks:
        return chunks

    store = get_vector_store()

    # Group chunks by section_id
    section_groups: dict[str, list[dict]] = {}
    for c in chunks:
        sec_id = c.get("metadata", {}).get("section_id", "")
        if sec_id:
            section_groups.setdefault(sec_id, []).append(c)

    if not section_groups:
        return chunks

    # Track already-included chunk IDs (Qdrant point IDs)
    included_ids = set(c.get("id") for c in chunks if c.get("id"))

    expanded: list[dict] = []

    for sec_id, group in section_groups.items():
        section_chunks = store.get_chunks_by_section_id(sec_id)
        if not section_chunks:
            continue

        # Collect section_chunk_index values from retrieved chunks
        retrieved_indices = set()
        total_in_section = 0
        for c in group:
            idx = c.get("metadata", {}).get("section_chunk_index", -1)
            if idx >= 0:
                retrieved_indices.add(idx)
            total = c.get("metadata", {}).get("section_total_chunks", 0)
            total_in_section = max(total_in_section, total)

        # Compute desired window around retrieved indices
        desired = set()
        for idx in retrieved_indices:
            for offset in range(-window, window + 1):
                desired.add(idx + offset)

        if total_in_section > 0:
            desired = {i for i in desired if 0 <= i < total_in_section}

        # Select missing sibling chunks within the window
        for sc in section_chunks:
            sc_id = sc.get("id")
            if sc_id in included_ids:
                continue
            sc_idx = sc.get("metadata", {}).get("section_chunk_index", -1)
            if sc_idx in desired:
                sc["_expanded"] = True
                expanded.append(sc)
                included_ids.add(sc_id)

    if not expanded:
        return chunks

    # Merge expanded chunks into original list, preserving section order
    all_by_section: dict[str, list[dict]] = {}
    for c in chunks:
        sec_id = c.get("metadata", {}).get("section_id", "")
        all_by_section.setdefault(sec_id, []).append(c)
    for c in expanded:
        sec_id = c.get("metadata", {}).get("section_id", "")
        all_by_section.setdefault(sec_id, []).append(c)

    for sec_id in all_by_section:
        all_by_section[sec_id].sort(
            key=lambda x: x.get("metadata", {}).get("section_chunk_index", -1),
        )

    result: List[dict] = []
    seen_ids = set()

    for c in chunks:
        sec_id = c.get("metadata", {}).get("section_id", "")
        if sec_id and sec_id in all_by_section:
            for sc in all_by_section.pop(sec_id):
                scid = sc.get("id")
                if scid not in seen_ids:
                    result.append(sc)
                    if scid:
                        seen_ids.add(scid)
        else:
            cid = c.get("id")
            if cid not in seen_ids:
                result.append(c)
                if cid:
                    seen_ids.add(cid)

    for sec_id, remaining in all_by_section.items():
        for sc in remaining:
            scid = sc.get("id")
            if scid not in seen_ids:
                result.append(sc)
                if scid:
                    seen_ids.add(scid)

    logger.info(
        "Chunk expansion: added %d adjacent chunks (window=%d) "
        "to %d retrieved chunks → %d total, across %d sections",
        len(expanded), window, len(chunks), len(result), len(section_groups),
    )
    return result


# ── Table-aware row expansion ──────────────────────────────────────────────

def _expand_table_rows(
    chunks: List[dict],
) -> List[dict]:
    """Expand table chunks by retrieving additional rows from the same table.

    When a question references a table, this ensures full row context is preserved
    rather than returning isolated cells. Uses section_id, file_name, page, and
    table_name metadata to find sibling rows.

    Fallback chain:
      1. Same section_id + table_name (primary, most precise)
      2. Same file_name + page + table_name (when section_id is missing)
      3. Same file_name + table_name (broadest, catches cross-page tables)
    """
    if not chunks:
        return chunks

    store = get_vector_store()

    table_groups: dict[str, list[dict]] = {}
    for c in chunks:
        meta = c.get("metadata", {})
        is_table = meta.get("content_type") == "table" or meta.get("table_preserved", False)
        if is_table:
            table_name = meta.get("table_name", meta.get("table_title", ""))
            section_id = meta.get("section_id", "")
            file_name = meta.get("file_name", meta.get("source", ""))
            page = meta.get("page", meta.get("table_page", -1))
            # Primary key: section_id + table_name
            if section_id:
                group_key = f"sec:{section_id}:{table_name}"
            else:
                # Fallback: file_name + page + table_name
                group_key = f"file:{file_name}:pg{page}:{table_name}"
            table_groups.setdefault(group_key, []).append(c)

    if not table_groups:
        return chunks

    included_ids = set(c.get("id") for c in chunks if c.get("id"))
    expanded_rows: List[dict] = []

    for group_key, group in table_groups.items():
        parts = group_key.split(":", 2)
        key_type = parts[0] if len(parts) > 0 else ""
        table_name = parts[2] if len(parts) > 2 else ""

        sibling_candidates: List[dict] = []

        if key_type == "sec":
            section_id = parts[1] if len(parts) > 1 else ""
            if section_id:
                sibling_candidates = store.get_chunks_by_section_id(section_id) or []

        elif key_type == "file":
            # Fallback: get all chunks from the same source and filter by table
            file_name = parts[1] if len(parts) > 1 else ""
            if file_name:
                all_source_chunks = store.get_chunks_by_source([file_name]) if file_name else []
                sibling_candidates = all_source_chunks

        if not sibling_candidates:
            continue

        # Collect all table chunks matching the group
        for sc in sibling_candidates:
            sc_meta = sc.get("metadata", {})
            sc_table = sc_meta.get("table_name", sc_meta.get("table_title", ""))
            sc_is_table = sc_meta.get("content_type") == "table" or sc_meta.get("table_preserved", False)

            if not sc_is_table:
                continue
            if table_name and sc_table != table_name:
                continue

            sc_id = sc.get("id")
            if sc_id and sc_id not in included_ids:
                sc["_expanded"] = True
                expanded_rows.append(sc)
                included_ids.add(sc_id)

    if not expanded_rows:
        return chunks

    result = list(chunks) + expanded_rows
    logger.info(
        f"Table row expansion: added {len(expanded_rows)} rows from {len(table_groups)} table(s)"
    )
    return result


# ── Core retriever ──────────────────────────────────────────────────────────

def _build_search_filter(
    document_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    search_mode: str = "conversation",
) -> Optional[qdrant_models.Filter]:
    """Build a Qdrant Filter for document/conversation scoping.

    Modes:
      - "document"     → scope to a single document_id
      - "conversation" → scope to a conversation_id (all docs in that conversation)
      - "global"       → no filter (search everything)

    Returns None (no filter) for global mode, or a Qdrant Filter with must
    conditions for scoped modes.
    """
    if search_mode == "global":
        return None

    must_conditions = []

    if search_mode == "document" and document_id:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="document_id",
                match=qdrant_models.MatchValue(value=document_id),
            ),
        )
    elif search_mode == "conversation" and conversation_id:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="conversation_id",
                match=qdrant_models.MatchValue(value=conversation_id),
            ),
        )

    if not must_conditions:
        return None

    return qdrant_models.Filter(must=must_conditions)


def retrieve(
    query:             str,
    k:                 int   = None,
    score_threshold:   float = None,
    max_context_chars: int   = None,
    mmr_lambda:        float = None,
    source_files:      Optional[List[str]] = None,
    summarization_mode: bool  = False,
    document_id:       Optional[str] = None,
    conversation_id:   Optional[str] = None,
    search_mode:       str   = "conversation",
) -> RetrievalResult:
    """
    Full retrieval pipeline:

     1. Build search filter (document / conversation scoping)
     2. Query classification (table question, multi-hop) → parameter adaptation
     3. Query expansion — domain-term injection + LLM rephrasing
     4. Hybrid search (BM25 + semantic) with MMR diversity
     5. Score threshold filter
     6. Jaccard-based near-duplicate deduplication
     7. Cross-encoder reranking
     8. Novelty-aware selection (avoids redundant content)
     9. Source diversity enforcement (balance across documents)
    10. Section diversity enforcement (balance across headings)
    11. Adjacent chunk expansion — retrieves sibling chunks from same section
        to complete multi-page sections, lists, tables, and framework definitions
    12. Table-aware row expansion — when question references tables
    13. Context assembly with section-grouped formatting
    14. Source tracking & structured trace

    When summarization_mode=True:
      - Uses higher MMR lambda for more diversity
      - Lowers score threshold to include more sections
      - Applies stronger section diversity enforcement
      - Uses novelty selection with higher diversity weight
      - Forces broader section coverage over depth
    """
    k                 = k                 or settings.retrieval_k
    fetch_k           = settings.retrieval_fetch_k
    score_threshold   = score_threshold   or settings.retrieval_score_threshold
    max_context_chars = max_context_chars or settings.retrieval_max_context_chars
    mmr_lambda        = mmr_lambda        or settings.retrieval_mmr_lambda

    # ── 0. Query classification ─────────────────────────────────────────────
    is_table_q = _is_table_question(query) if settings.table_qa_enabled else False
    is_multi_hop = _is_multi_hop_question(query) if settings.multihop_enabled else False

    if is_table_q and settings.eval_log_retrieved_chunks:
        logger.info(f"Table question detected: '{query[:80]}' — adapting retrieval")
    if is_multi_hop and settings.eval_log_retrieved_chunks:
        logger.info(f"Multi-hop question detected: '{query[:80]}' — adapting retrieval")

    # Adapt parameters for table questions — expand k, lower threshold
    if is_table_q:
        k = max(k, 15)
        score_threshold = max(score_threshold * 0.7, 0.05)
        mmr_lambda = max(mmr_lambda * 0.8, 0.2)

    # Adapt parameters for multi-hop questions — more sections, more diversity
    if is_multi_hop:
        k = max(k, settings.multihop_min_sections * settings.retrieval_max_chunks_per_section)
        score_threshold = max(score_threshold * 0.7, 0.05)
        mmr_lambda = max(mmr_lambda * 0.8, 0.2)

    # Summarization mode: override parameters for broader, more diverse retrieval
    if summarization_mode:
        mmr_lambda = min(mmr_lambda * 0.7, 0.5)
        score_threshold = max(score_threshold * 0.5, 0.05)
        k = max(k, settings.retrieval_min_sections * settings.retrieval_max_chunks_per_section + 2)
        if settings.eval_log_retrieved_chunks:
            logger.info(
                f"Summarization mode: mmr={mmr_lambda:.2f}, threshold={score_threshold:.2f}, k={k}"
            )

    # ── 0b. Build document/conversation filter ──────────────────────────────
    filter_condition = _build_search_filter(
        document_id=document_id,
        conversation_id=conversation_id,
        search_mode=search_mode,
    )
    if filter_condition:
        logger.info(
            f"Search filter active: mode={search_mode}, "
            f"document_id={document_id}, conversation_id={conversation_id}"
        )

    trace = RetrievalTrace(query) if settings.eval_trace_enabled else None

    # ── 1. Query expansion + hybrid search (with optional filter) ──────────
    effective_fetch_k = fetch_k * 3 if source_files else fetch_k
    raw_results, expanded_queries = _search_with_expansion(
        query, k, effective_fetch_k, mmr_lambda,
        filter_condition=filter_condition,
    )

    # ── 1b. Source filter ──────────────────────────────────────────────────
    if source_files:
        source_set = set(source_files)
        raw_results = [r for r in raw_results if _get_source_key(r) in source_set]
        logger.info(f"After source filter ({len(source_files)} source(s)): {len(raw_results)} chunks")

    if trace:
        trace.stage("hybrid_search", 0, len(raw_results), f"{len(expanded_queries)} query variants")

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Retrieved {len(raw_results)} raw candidates for query: '{query[:120]}'")
        for i, r in enumerate(raw_results[:10]):
            score_info = f"score={r.get('score', 0):.4f}"
            bm25_info = f" bm25={r.get('bm25_score', 0):.4f}" if "bm25_score" in r else ""
            if settings.eval_log_scores:
                section = r.get("metadata", {}).get("heading", "") or r.get("metadata", {}).get("section", "")
                source = r.get("metadata", {}).get("file_name", r.get("metadata", {}).get("source", ""))
                logger.info(f"  [{i+1}] {score_info}{bm25_info} | src='{source}' | section='{section}' | preview={r['text'][:80]}")

    # ── 2. Score filter ────────────────────────────────────────────────────
    filtered = [r for r in raw_results if r["score"] >= score_threshold]
    if settings.eval_log_scores:
        logger.info(f"After score filter (≥{score_threshold}): {len(filtered)} / {len(raw_results)} chunks")
    if trace:
        trace.stage("score_filter", len(raw_results), len(filtered), f"threshold={score_threshold}")

    # ── 3. Jaccard-based near-duplicate dedup ──────────────────────────────
    deduped = _deduplicate_jaccard(filtered, threshold=settings.retrieval_jaccard_threshold)
    if settings.eval_log_retrieved_chunks:
        logger.info(f"After Jaccard dedup: {len(deduped)} unique chunks")
    if trace:
        trace.stage("jaccard_dedup", len(filtered), len(deduped), f"threshold={settings.retrieval_jaccard_threshold}")

    # ── 4. Rerank ──────────────────────────────────────────────────────────
    reranker = _get_reranker()
    if reranker and deduped:
        if settings.reranker_provider == "cohere":
            reranked = _rerank_cohere(reranker, query, deduped, top_n=min(k * 2, len(deduped)))
        else:
            reranked = _rerank_bge(reranker, query, deduped, top_n=min(k * 2, len(deduped)))
    else:
        reranked = sorted(deduped, key=lambda x: x.get("rerank_score", x["score"]), reverse=True)
        reranked = reranked[:k * 2]

    if settings.eval_log_reranking:
        logger.info(f"After reranking: top {len(reranked)} chunks")
    if trace:
        trace.stage("rerank", len(deduped), len(reranked), f"provider={settings.reranker_provider}")

    # ── 5. Novelty-aware selection ─────────────────────────────────────────
    novelty_selected = _select_with_novelty(reranked, k * 2)
    if trace:
        trace.stage("novelty_selection", len(reranked), len(novelty_selected), f"lambda={settings.retrieval_novelty_lambda}")

    # ── 6. Source diversity ───────────────────────────────────────────────
    source_balanced = _enforce_source_diversity(
        novelty_selected, k,
        settings.retrieval_min_sources,
        settings.retrieval_max_chunks_per_doc,
    )
    if settings.eval_log_retrieved_chunks:
        sources_in_balanced = set(_get_source_key(c) for c in source_balanced)
        counts_in_balanced = {}
        for c in source_balanced:
            s = _get_source_key(c)
            counts_in_balanced[s] = counts_in_balanced.get(s, 0) + 1
        logger.info(
            f"After source balancing: {len(sources_in_balanced)} sources in "
            f"top {len(source_balanced)} chunks — per-source: {counts_in_balanced}"
        )
    if trace:
        trace.stage("source_balancing", len(novelty_selected), len(source_balanced))

    # ── 7. Section diversity ──────────────────────────────────────────────
    diversified = _enforce_section_diversity(
        source_balanced, k,
        settings.retrieval_min_sections,
        settings.retrieval_max_chunks_per_section,
        full_pool=novelty_selected,
    )

    if settings.eval_log_retrieved_chunks:
        logger.info(f"Final {len(diversified)} chunks sent to LLM:")
        for i, c in enumerate(diversified):
            section = _get_section_key(c)
            source = _get_source_key(c)
            score = c.get("rerank_score", c.get("score", 0))
            logger.info(f"  [{i+1}] score={score:.4f} | src='{source}' | section='{section}' | len={len(c['text'])} | preview={c['text'][:80]}")
    if trace:
        trace.stage("section_diversity", len(source_balanced), len(diversified))

    # ── 8. Adjacent chunk expansion ──────────────────────────────────────
    expanded = _expand_with_adjacent_chunks(
        diversified,
        window=settings.retrieval_expansion_window,
    )
    if trace:
        trace.stage("chunk_expansion", len(diversified), len(expanded))

    # ── 8b. Table-aware row expansion ─────────────────────────────────────
    if is_table_q and settings.table_qa_retrieve_full_row and expanded:
        table_row_expanded = _expand_table_rows(expanded)
        if len(table_row_expanded) > len(expanded):
            logger.info(
                f"Table row expansion: added {len(table_row_expanded) - len(expanded)} "
                f"table rows for table question"
            )
            expanded = table_row_expanded
            if trace:
                trace.stage("table_row_expansion", len(diversified), len(expanded))

    # ── 9. Build context with section-grouped formatting ─────────────────
    sources: List[SourceReference] = []
    source_number = 1

    for chunk in expanded:
        text = chunk["text"].strip()
        meta = chunk.get("metadata", {})
        src = _get_source_key(chunk)
        section_label = meta.get("heading", meta.get("section", ""))
        is_expanded = chunk.get("_expanded", False)
        is_table_chunk = meta.get("content_type") == "table" or meta.get("table_preserved", False)

        if is_table_chunk:
            table_name = meta.get("table_name", meta.get("table_title", ""))
            annotations = " — ".join(filter(None, [section_label, table_name]))
            if annotations:
                label = f"[{source_number}] ({src} — {annotations})"
            else:
                label = f"[{source_number}] ({src})" if src else f"[{source_number}]"
        else:
            if section_label:
                label = f"[{source_number}] ({src} — {section_label})" if src else f"[{source_number}]"
            else:
                label = f"[{source_number}] ({src})" if src else f"[{source_number}]"

        sources.append(SourceReference(
            file_name     = meta.get("file_name", meta.get("source", "unknown")),
            chunk_index   = meta.get("chunk_index", -1),
            page          = meta.get("page"),
            score         = round(chunk.get("rerank_score", chunk.get("score", 0)), 4),
            rerank_score  = round(chunk.get("rerank_score", 0), 4),
            preview       = text[:120],
            content_type  = meta.get("content_type", "prose"),
            section       = section_label,
            table_name    = meta.get("table_name", meta.get("table_title", "")),
            source_number = source_number,
            expanded      = is_expanded,
        ))
        source_number += 1

    # Build raw context string (section-grouped)
    section_groups = _group_chunks_by_section(expanded)
    context_parts: List[str] = []
    chars_used = 0
    for section_name, group in section_groups.items():
        if section_name and section_name != "__prose__":
            header = f"=== Section: {section_name} ==="
            if chars_used + len(header) <= max_context_chars:
                context_parts.append(header)
                chars_used += len(header)
        for chunk in group:
            text = chunk.get("text", "").strip()
            meta = chunk.get("metadata", {})
            is_table = meta.get("content_type") == "table" or meta.get("table_preserved", False)
            prefix = "[TABLE] " if is_table else ""
            entry = f"{prefix}{text}"
            if chars_used + len(entry) + 2 <= max_context_chars:
                context_parts.append(entry)
                chars_used += len(entry) + 2
            else:
                break
        if chars_used >= max_context_chars:
            break
    context = "\n\n".join(context_parts)

    # ── 9. Retrieval metrics & trace ──────────────────────────────────────
    sections_found = set()
    sources_found = set()
    content_type_counts = {}
    for c in expanded:
        sec = _get_section_key(c)
        if sec != "__prose__":
            sections_found.add(sec)
        src = _get_source_key(c)
        if src != "__unknown__":
            sources_found.add(src)
        ct = c.get("metadata", {}).get("content_type", "prose")
        content_type_counts[ct] = content_type_counts.get(ct, 0) + 1

    per_source_counts = {}
    per_section_counts = {}
    for c in expanded:
        s = _get_source_key(c)
        per_source_counts[s] = per_source_counts.get(s, 0) + 1
        sec = _get_section_key(c)
        per_section_counts[sec] = per_section_counts.get(sec, 0) + 1

    metrics = {
        "total_candidates":    len(raw_results),
        "after_filter":        len(filtered),
        "after_dedup":         len(deduped),
        "after_rerank":        len(reranked),
        "after_novelty":       len(novelty_selected),
        "final_chunks":        len(expanded),
        "expanded_chunks":     len(expanded) - len(diversified),
        "sections_covered":    len(sections_found),
        "sources_covered":     len(sources_found),
        "content_types":       content_type_counts,
        "expanded_queries":    len(expanded_queries),
        "hybrid_search":       settings.retrieval_hybrid_search,
        "source_balancing":    settings.retrieval_source_balancing,
        "per_source":          per_source_counts,
        "per_section":         per_section_counts,
    }

    if trace:
        trace.stage("context_assembly", len(expanded), len(sources))
        metrics["trace"] = trace.to_dict()

    return RetrievalResult(
        query            = query,
        context          = context,
        sources          = sources,
        chunks           = expanded,
        total_found      = len(expanded),
        expanded_queries = expanded_queries,
        retrieval_metrics = metrics,
    )


# ── Context formatter ───────────────────────────────────────────────────────

def retrieve_as_dict(
    query:           str,
    k:               int   = None,
    score_threshold: float = None,
    source_files:    Optional[List[str]] = None,
) -> dict:
    result = retrieve(query, k=k, score_threshold=score_threshold, source_files=source_files)
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
                "expanded":    s.expanded,
            }
            for s in result.sources
        ],
    }


def _is_table_question(query: str) -> bool:
    """Detect if a question references tabular data."""
    table_patterns = [
        r'\b(table|row|column|cell)\b',
        r'\b(compare|comparison|versus|vs\.)\b',
        r'\b(rank|ranking|sorted|sort)\b',
        r'\b(breakdown|distribution|overview|summary)\s+of\b',
        r'\b(list|show|give|what.are)\s+(all|every|each|the)\b',
        r'\b(how many|how much|count|total)\b',
        r'\b(percentage|percent|rate|ratio)\b',
        r'\b(revenue|profit|cost|margin|price)\b',
        r'\b(roi|mrr|arr|ebitda|growth)\b',
    ]
    q_lower = query.lower()
    for pat in table_patterns:
        if re.search(pat, q_lower):
            return True
    return False


def _is_multi_hop_question(query: str) -> bool:
    """Detect if a question requires multi-hop reasoning across sections."""
    multi_hop_patterns = [
        r'\b(and|or)\b.*\b(and|or)\b',
        r'\b(compare|contrast|difference|similar|versus|vs\.)\b',
        r'\b(relationship|connection|correlation|link|tie)\b',
        r'\b(both|all|across|throughout|overall)\b',
        r'\b(why|how\s+does|how\s+do|how\s+are)\b',
        r'\b(impact|effect|influence|affect)\b',
        r'\b(although|however|while|whereas)\b',
        r'\b(aggregate|combine|synthesize|integrate)\b',
        r'\b(implications?|significance|meaning)\b',
        r'\b(trend|pattern|change|shift|develop)\b',
    ]
    q_lower = query.lower()
    score = 0
    for pat in multi_hop_patterns:
        if re.search(pat, q_lower):
            score += 1
    return score >= 2


def _group_chunks_by_section(chunks: List[dict]) -> dict:
    """Group chunks by section heading for structured context assembly."""
    groups: dict = {}
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        section = meta.get("heading_path", "") or meta.get("heading", "") or meta.get("section", "__prose__")
        groups.setdefault(section, []).append(chunk)
    return groups


def format_context_for_llm(result: RetrievalResult) -> str:
    """
    Format retrieved context for the LLM prompt with cross-document and
    multi-section synthesis support, section-grouped layout, and table-aware
    context assembly.

    Improvements:
      - Chunks are grouped by section with clear section headers
      - Table chunks are flagged with [TABLE] markers
      - Adjacent/expanded chunks are noted
      - Synthesis hints are included when multiple sections/docs are present
      - Table question detection adds specialized instructions
    """
    if not result.context:
        return "No relevant context was found in the knowledge base."

    has_tables = any(s.content_type == "table" for s in result.sources)
    table_question = _is_table_question(result.query)
    multi_hop = _is_multi_hop_question(result.query)

    # ── Source reference map ──────────────────────────────────────────────
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
        if s.expanded:
            line += " [adjacent]"
        line += f" — score {s.score}"
        source_lines.append(line)
    source_str = "\n".join(source_lines)

    # ── Detect context characteristics ────────────────────────────────────
    has_expanded = any(getattr(s, "expanded", False) for s in result.sources)
    multi_section = len(set(s.section for s in result.sources if s.section)) > 1
    multi_source = len(set(s.file_name for s in result.sources if s.file_name)) > 1

    hints = []
    if has_tables or table_question:
        hints.append("Note: The context or question involves TABLE data. Extract ALL numeric values, percentages, and row-level data from every relevant row. Return complete row information, not just individual cells.")
    if has_expanded:
        hints.append("Note: Adjacent context chunks from the same section have been merged to provide complete coverage.")
    if multi_hop:
        hints.append("Note: This question may require information from multiple places. Only use what is directly relevant to the question.")
    if multi_hop and has_tables:
        hints.append("Note: This question may reference both tables and surrounding text. Use only the data that answers the question.")

    hint_str = "\n".join(hints)
    sep = "\n\n"
    hints_block = f"{hint_str}{sep}" if hints else ""

    # ── Pre-generation synthesis context ────────────────────────────────────
    synthesis_block = ""
    if settings.synthesis_enabled and len(result.chunks) >= 2:
        try:
            from app.services.synthesis import build_synthesis_context, extract_synthesis_hints
            synthesis_context = build_synthesis_context(result.chunks, result.query)
            if synthesis_context:
                synthesis_hints = extract_synthesis_hints(synthesis_context, result.chunks)
                if synthesis_hints:
                    synthesis_block = "SYNTHESIS ANALYSIS:\n" + "\n".join(synthesis_hints) + "\n\n"
                    if settings.eval_log_retrieved_chunks:
                        logger.info(f"Synthesis: {len(synthesis_hints)} hints generated")
        except Exception as e:
            logger.warning(f"Synthesis context generation failed: {e}")

    # ── Section-grouped context body with chunk merging ────────────────────
    section_groups = _group_chunks_by_section(result.chunks)
    context_parts: List[str] = []

    for section_name, group in section_groups.items():
        if section_name and section_name != "__prose__":
            context_parts.append(f"=== Section: {section_name} ===")

        merged_texts: List[str] = []
        current_merge: List[str] = []
        current_is_table = False

        for chunk in group:
            meta = chunk.get("metadata", {})
            text = chunk.get("text", "").strip()
            is_table_chunk = meta.get("content_type") == "table" or meta.get("table_preserved", False)
            is_expanded = chunk.get("_expanded", False)

            if is_table_chunk:
                # Flush current merge and add table separately
                if current_merge:
                    merged_texts.append("\n\n".join(current_merge))
                    current_merge = []
                merged_texts.append(f"[TABLE CONTENT] {text}")
                current_is_table = True
            else:
                if current_is_table and current_merge:
                    merged_texts.append("\n\n".join(current_merge))
                    current_merge = []
                prefix = ""
                if is_expanded:
                    prefix = "[ADJACENT CONTENT] "
                current_merge.append(f"{prefix}{text}")
                current_is_table = False

        if current_merge:
            merged_texts.append("\n\n".join(current_merge))

        context_parts.extend(merged_texts)
        context_parts.append("")

    section_context = "\n".join(context_parts).strip()

    return (
        f"SOURCES:\n{source_str}\n\n"
        f"{hints_block}"
        f"{synthesis_block}"
        f"RETRIEVED CONTEXT (grouped by section):\n"
        f"{section_context}"
    )
