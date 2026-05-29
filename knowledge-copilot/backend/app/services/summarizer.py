"""
summarizer.py — Multi-stage hierarchical summarization with section-diverse retrieval.

Pipeline:
  1. Section-diverse chunk retrieval (proportional sampling across all sections)
  2. Build global concept inventory (LLM + fallback seed-term coverage, merged)
  3. Map every concept to chunks that mention it (cross-chunk aggregation)
  4. Score chunks by concept coverage + rare-concept inverse-frequency boost
  5. Section-balanced selection (minimum per section + proportional fill)
  6. Coverage validation and supplementation (plugs missing technical terms)
  7. Per-section summarization (one LLM call per section, forces every section
     to be independently processed — prevents any single section from dominating)
  8. Merge section summaries and generate final 3-point global summary
"""

import json
import logging
import re
from collections import Counter
from typing import Dict, List, Optional, Set

from app.core.config import settings
from app.services.llm import get_llm
from app.services.vector_store import get_vector_store

logger = logging.getLogger("knowledge_copilot.summarizer")

# ── Stop words for entity extraction ───────────────────────────────────────────

_STOP_WORDS: set = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "this", "that",
    "these", "those", "it", "its", "they", "them", "their", "we", "our",
    "you", "your", "he", "she", "his", "her", "not", "no", "nor", "all",
    "each", "every", "both", "few", "more", "most", "other", "some", "such",
}


def _extract_meaningful_entities(text: str) -> set:
    entities: set = set()
    patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        r'\b[A-Z]{2,}\b',
        r'\b\d+(?:\.\d+)?%?\b',
        r'\$\d+(?:,\d{3})*(?:\.\d+)?',
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        for m in matches:
            m_clean = m.strip()
            if m_clean and m_clean.lower() not in _STOP_WORDS:
                entities.add(m_clean)
    return entities


SEED_CONCEPTS: Set[str] = {
    "embedding", "embeddings", "vector", "semantic search", "similarity",
    "cosine similarity", "vector store", "vector database",
    "retrieval", "retriever", "reranker", "reranking", "bm25", "hybrid search",
    "mmr", "maximal marginal relevance", "query expansion",
    "llm", "large language model", "inference", "generation", "prompt",
    "context window", "token", "completion", "chain", "agent",
    "pipeline", "workflow", "validation", "validation layer",
    "quality check", "automation", "orchestration",
    "pinecone", "voyage", "index", "indexing", "database", "storage", "api",
    "chunking", "chunker", "splitting", "text splitter", "document loader",
    "parsing", "extraction",
    "monitoring", "evaluation", "metric", "observability", "logging",
    "tracing", "dashboard",
    "clustering", "cluster", "classification", "categorization", "analysis",
    "insight", "pattern", "grouping",
    "architecture", "system design", "microservice", "service", "module",
    "component", "integration",
    "accuracy", "precision", "recall", "f1", "score", "quality",
    "hallucination", "grounding", "schema", "retry",
    "multi-turn", "conversation", "workspace", "copilot",
}


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


# ── Phase 1: Section-diverse retrieval ─────────────────────────────────────────
# Replaces the old flat retrieval with proportional section-aware sampling.
# Guarantees every section is represented, preventing any single section
# (e.g., "RAG Workspace") from dominating the candidate pool.

# ── Phase 0a: Global entity centrality ─────────────────────────────────────────
# Identifies entities that span the most sections — these are globally significant.

def _compute_global_entity_centrality(chunks: List[dict]) -> dict:
    """Compute how many distinct sections each entity appears in.
    
    Returns dict mapping entity -> {section_count, sections: set, frequency: int}.
    Entities appearing in MORE sections are more globally significant.
    """
    entity_sections: dict[str, set] = {}
    entity_frequency: dict[str, int] = {}

    for chunk in chunks:
        sec = _get_section_key(chunk)
        text = chunk.get("text", "")
        entities = _extract_meaningful_entities(text)
        for ent in entities:
            entity_sections.setdefault(ent, set()).add(sec)
            entity_frequency[ent] = entity_frequency.get(ent, 0) + 1

    centrality = {}
    for ent, sections in entity_sections.items():
        centrality[ent] = {
            "section_count": len(sections),
            "sections": sections,
            "frequency": entity_frequency.get(ent, 0),
        }

    logger.info(
        f"Global entity centrality: {len(centrality)} entities across sections, "
        f"top: {sorted(centrality, key=lambda e: centrality[e]['section_count'], reverse=True)[:5]}"
    )
    return centrality


def _get_top_global_entities(
    centrality: dict,
    min_sections: Optional[int] = None,
    top_n: int = 20,
) -> List[str]:
    """Return the entities that appear in the most sections (globally significant)."""
    min_sections = min_sections or settings.summarization_global_entity_min_sections
    scored = []
    for e, data in centrality.items():
        if isinstance(data, dict):
            sc = data.get("section_count", 0)
            if sc >= min_sections:
                scored.append((e, sc, data.get("frequency", 0)))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [e for e, _, _ in scored[:top_n]]


# ── Phase 0b: Section importance scoring ───────────────────────────────────────
# Scores each section by how many globally significant entities and concepts it
# covers.  Important sections get more representation in the final chunk selection.

def _compute_section_importance(
    chunks: List[dict],
    global_centrality: dict,
    concepts: Optional[List[dict]] = None,
    concept_chunk_map: Optional[dict] = None,
) -> dict:
    """Score each section by global entity density and concept coverage breadth.
    
    Returns dict mapping section_name -> importance_score (0.0 - 1.0).
    """
    section_groups: dict[str, list] = {}
    for c in chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    section_scores: dict[str, float] = {}
    for sec, group in section_groups.items():
        section_text = " ".join(c.get("text", "") for c in group).lower()

        # Entity richness: fraction of top global entities present in this section
        top_entities = _get_top_global_entities(global_centrality, top_n=30)
        entities_in_section = sum(1 for e in top_entities if e.lower() in section_text)
        entity_richness = entities_in_section / max(len(top_entities), 1)

        # Entity cross-section score: entities that span many sections AND appear here
        cross_section_entities = 0
        for ent, data in global_centrality.items():
            if data["section_count"] >= settings.summarization_global_entity_min_sections:
                if ent.lower() in section_text:
                    cross_section_entities += data["section_count"]
        max_possible = max(
            (sum(d["section_count"] for d in global_centrality.values())),
            1,
        )
        cross_section_score = min(cross_section_entities / max_possible, 1.0)

        # Concept coverage breadth (if available)
        concept_coverage = 0.0
        if concepts and concept_chunk_map:
            chunk_indices_in_section = set(
                i for i, c in enumerate(chunks) if _get_section_key(c) == sec
            )
            covered_concepts = 0
            for ci, chunk_indices in concept_chunk_map.items():
                if chunk_indices & chunk_indices_in_section:
                    covered_concepts += 1
            concept_coverage = covered_concepts / max(len(concepts), 1)

        # Section size factor (very small sections are less important)
        size_factor = min(len(group) / 5.0, 1.0)

        importance = (
            0.35 * entity_richness
            + 0.30 * cross_section_score
            + 0.25 * concept_coverage
            + 0.10 * size_factor
        )

        section_scores[sec] = round(importance, 4)

    logger.info(
        f"Section importance scores: {len(section_scores)} sections, "
        f"top: {sorted(section_scores, key=lambda s: section_scores.get(s, 0.0) or 0.0, reverse=True)[:5]}"
    )
    return section_scores


def _get_important_sections(
    section_scores: dict,
    threshold: float = 0.25,
    min_count: int = 1,
) -> set:
    """Return set of section names that are above the importance threshold."""
    sorted_scores = sorted(section_scores.items(), key=lambda x: x[1], reverse=True)
    important = set()
    for sec, score in sorted_scores:
        if score >= threshold:
            important.add(sec)
        if len(important) >= min_count:
            break
    return important


def _retrieve_all_chunks(source_files: Optional[List[str]] = None) -> List[dict]:
    store = get_vector_store()
    if source_files:
        chunks = store.get_chunks_by_source(source_files)
        logger.info(f"Retrieved {len(chunks)} chunks from {len(source_files)} source(s)")
    else:
        all_sources = list(store.list_sources().keys())
        chunks = store.get_chunks_by_source(all_sources) if all_sources else []
        logger.info(f"Retrieved {len(chunks)} chunks from all sources")
    return chunks


def _deduplicate_similar_chunks(
    chunks: List[dict],
    threshold: float = 0.85,
) -> List[dict]:
    """Remove near-duplicate chunks based on token-set Jaccard similarity."""
    if not chunks:
        return chunks

    def _token_set(text: str) -> set:
        return set(re.findall(r'\w+', text.lower()))

    result: List[dict] = []
    kept_sets: List[set] = []

    for c in chunks:
        tokens = _token_set(c.get("text", ""))
        if not tokens:
            result.append(c)
            continue

        is_dup = False
        for ks in kept_sets:
            if len(tokens) < 10 or len(ks) < 10:
                continue
            jaccard = len(tokens & ks) / max(len(tokens | ks), 1)
            if jaccard > threshold:
                is_dup = True
                break

        if not is_dup:
            result.append(c)
            kept_sets.append(tokens)

    removed = len(chunks) - len(result)
    if removed:
        logger.info(f"Dedup: removed {removed} similar chunks (threshold={threshold})")

    return result


def _retrieve_diverse_chunks(
    source_files: Optional[List[str]] = None,
    max_chunks: int = 50,
    min_per_section: int = 4,
) -> List[dict]:
    all_chunks = _retrieve_all_chunks(source_files)
    if not all_chunks:
        return all_chunks

    # Remove near-duplicates before section grouping
    all_chunks = _deduplicate_similar_chunks(all_chunks, threshold=0.88)

    section_groups: Dict[str, List[dict]] = {}
    for c in all_chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    section_names = sorted(section_groups.keys())
    num_sections = len(section_names)

    if num_sections <= 1:
        return all_chunks[:max_chunks]

    result: List[dict] = []
    selected_ids: Set[int] = set()

    # Evenly sample across each section by position, not just first N
    for sec in section_names:
        group = section_groups[sec]
        take = min(min_per_section, len(group), max_chunks - len(result))
        if take <= 0:
            continue
        # Evenly spaced indices across the section
        if take == 1:
            indices = [0]
        else:
            step = (len(group) - 1) / max(take - 1, 1)
            indices = [int(round(i * step)) for i in range(take)]
        for idx in indices:
            if idx < len(group):
                chunk = group[idx]
                if id(chunk) not in selected_ids:
                    selected_ids.add(id(chunk))
                    result.append(chunk)

    remaining_sections = [
        s for s in section_names
        if len([c for c in result if _get_section_key(c) == s]) < len(section_groups[s])
    ]
    while len(result) < max_chunks and remaining_sections:
        for sec in remaining_sections[:]:
            if len(result) >= max_chunks:
                break
            candidates = [c for c in section_groups[sec] if id(c) not in selected_ids]
            if not candidates:
                remaining_sections.remove(sec)
                continue
            # Pick from a position not yet represented
            existing_positions = set()
            for c in result:
                if _get_section_key(c) == sec:
                    pr = c.get("metadata", {}).get("position_ratio", 0.5)
                    existing_positions.add(round(pr * 10))
            chosen = None
            for c in candidates:
                pr = round(c.get("metadata", {}).get("position_ratio", 0.5) * 10)
                if pr not in existing_positions:
                    chosen = c
                    break
            if chosen is None:
                chosen = candidates[0]
            selected_ids.add(id(chosen))
            result.append(chosen)
            count_in_result = sum(1 for c in result if _get_section_key(c) == sec)
            if count_in_result >= len(section_groups[sec]):
                remaining_sections.remove(sec)

    logger.info(
        f"Diverse retrieval: {len(result)} chunks from {num_sections} sections "
        f"(evenly sampled + round-robin fill, dedup'd)"
    )
    return result


# ── Phase 2: Global concept inventory ─────────────────────────────────────────
# Uses the LLM to identify ALL major technical systems across the document
# (not just TF-popular terms). Samples one chunk per section for efficiency.

_CONCEPT_EXTRACTION_PROMPT = """You are analyzing a technical document. Below are representative excerpts from different sections of the document.

Document section headings:
{section_headings}

Identify the 15-20 most important technical contributions, systems, pipelines, and features described ACROSS ALL excerpts. Focus on engineering contributions: data pipelines, AI/ML integrations, search/indexing systems, validation layers, automation workflows, and infrastructure.

You MUST check for ALL of the following areas and include concepts from any that are discussed in the document:
- Embedding models (e.g., Voyage AI, BGE, OpenAI)
- Semantic search and retrieval pipelines
- LLM inference integrations (e.g., Groq, OpenAI, Ollama)
- Vector storage systems (e.g., Qdrant, Pinecone, Chroma)
- Document chunking and parsing strategies
- Clustering and insight analysis
- Schema validation and retry handling
- Cross-encoder reranking
- Query expansion techniques
- Conversation and multi-turn support

Rules:
- Each concept must be a discrete technical system or feature (not a general topic)
- Use specific technology names when available (e.g., "Voyage AI", "Pinecone", "Groq")
- Cover the FULL breadth of the document, not just the last excerpts
- Do NOT merge multiple systems into one concept
- Include concepts from BOTH early and late sections of the document
- If a technical area above is NOT discussed in the document, omit it

For each concept, provide:
1. "name": A concise, specific name (e.g., "Embedding Pipeline with Voyage AI")
2. "keywords": 3-5 technical search keywords that would appear in chunks discussing this system (be specific)
3. "importance": 1-10 based on how technically central this is to the document

Output ONLY valid JSON — a single list of objects:
[{"name": "...", "keywords": ["...", "...", "..."], "importance": 8}]"""


def _build_concept_inventory(chunks: List[dict]) -> List[dict]:
    """Identify major technical concepts by having the LLM scan representative chunks."""
    section_groups: Dict[str, list] = {}
    for c in chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    # Include section headings so LLM knows full document structure
    section_headings = []
    for sec in sorted(section_groups.keys()):
        section_headings.append(f"- {sec}")

    # Take early, middle, late samples from each section for balanced coverage
    samples = []
    for sec, group in section_groups.items():
        sorted_group = sorted(
            group,
            key=lambda c: c.get("metadata", {}).get("position_ratio", 0.5),
        )
        selected = []
        if sorted_group:
            selected.append(sorted_group[0])
        if len(sorted_group) > 2:
            selected.append(sorted_group[len(sorted_group) // 2])
        if len(sorted_group) > 3:
            selected.append(sorted_group[-1])
        for s in selected:
            text = s.get("text", "")
            if text.strip():
                samples.append(f"[{sec}]\n{text[:1000]}")

    samples_text = "\n\n".join(samples[:40])
    headings_text = "\n".join(section_headings)

    llm = get_llm()
    try:
        response = llm.invoke([
            {"role": "system", "content": "You output only valid JSON arrays."},
            {
                "role": "user",
                "content": _CONCEPT_EXTRACTION_PROMPT.format(
                    section_headings=headings_text
                ) + f"\n\nDOCUMENT EXCERPTS:\n{samples_text}",
            },
        ])
        raw = response.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        concepts = json.loads(raw)
        if not isinstance(concepts, list):
            raise ValueError("LLM did not return a list")
        for c in concepts:
            if not all(k in c for k in ("name", "keywords", "importance")):
                raise ValueError(f"Missing keys in concept: {c}")
            c["keywords"] = [kw.lower() for kw in c["keywords"]]
        logger.info(f"LLM identified {len(concepts)} concepts: {[c['name'] for c in concepts]}")
        return concepts[:25]
    except Exception as e:
        logger.warning(f"LLM concept extraction failed: {e}")
        return []


# ── Phase 3: Concept-to-chunk mapping ─────────────────────────────────────────
# For each LLM-identified concept, find ALL chunks that mention its keywords.
# Also supplement with SEED_CONCEPTS matches for completeness.

def _keywords_match_chunk(keywords: List[str], text_lower: str) -> bool:
    """Check if any keyword matches within a chunk's text."""
    for kw in keywords:
        if " " in kw:
            if kw in text_lower:
                return True
        else:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                return True
    return False


def _map_concepts_to_chunks(
    concepts: List[dict],
    chunks: List[dict],
) -> Dict[int, Set[int]]:
    """Build mapping: concept_index -> set of chunk_indices that discuss it."""
    concept_chunk_map: Dict[int, Set[int]] = {i: set() for i in range(len(concepts))}

    for ci, concept in enumerate(concepts):
        keywords = concept.get("keywords", [])
        for chi, chunk in enumerate(chunks):
            text = chunk.get("text", "")
            if _keywords_match_chunk(keywords, text.lower()):
                concept_chunk_map[ci].add(chi)

    for ci, chunk_indices in concept_chunk_map.items():
        logger.debug(
            f"  Concept '{concepts[ci]['name']}': {len(chunk_indices)} matching chunks"
        )

    return concept_chunk_map


# ── Phase 4: Chunk scoring by concept coverage breadth ───────────────────────
# Score each chunk by how many DISTINCT concepts it covers.
# This naturally favors chunks that discuss multiple systems,
# not chunks that repeat the same term many times.

def _score_by_concept_coverage(
    chunks: List[dict],
    concepts: List[dict],
    concept_chunk_map: Dict[int, Set[int]],
    section_scores: Optional[Dict[str, float]] = None,
    global_centrality: Optional[Dict] = None,
) -> List[dict]:
    """Score each chunk by distinct concept coverage + seed term density + importance.

    Weight config (controlled by settings):
      - concept_coverage: breadth of distinct concepts covered
      - entity_density:   density of globally significant entities
      - position bonus:   favor diverse positions within the document
      - cross_section_boost: how many global entities from cross-section concepts
      - section_importance: importance of this chunk's section
    """
    chunk_concept_count = Counter()
    for ci, chunk_indices in concept_chunk_map.items():
        for chi in chunk_indices:
            chunk_concept_count[chi] += 1

    max_count = max(chunk_concept_count.values()) if chunk_concept_count else 1

    concept_freq = Counter()
    for ci, chunk_indices in concept_chunk_map.items():
        concept_freq[ci] = len(chunk_indices)
    max_concept_freq = max(concept_freq.values()) if concept_freq else 1

    w_cov = settings.summarization_importance_coverage_weight
    w_ent = settings.summarization_importance_entity_weight
    w_pos = settings.summarization_importance_position_weight
    w_con = settings.summarization_importance_concept_weight

    for chi, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        raw_terms = _extract_technical_terms(text)

        concept_coverage = chunk_concept_count.get(chi, 0) / max_count

        rare_boost = 0.0
        chunk_rare_sum = 0.0
        num_concepts_for_chunk = 0
        for ci, chunk_indices in concept_chunk_map.items():
            if chi in chunk_indices:
                freq = concept_freq.get(ci, 1)
                icf = 1.0 - (freq / max_concept_freq)
                chunk_rare_sum += icf
                num_concepts_for_chunk += 1
        if num_concepts_for_chunk > 0:
            rare_boost = chunk_rare_sum / num_concepts_for_chunk

        seed_density = min(len(raw_terms) / 15.0, 1.0)

        # Position bonus: favor chunks from diverse positions
        position_ratio = chunk.get("metadata", {}).get("position_ratio", 0.5)
        position_bonus = 1.0 - abs(position_ratio - 0.5) * 2.0

        # Entity density: fraction of top global entities present in this chunk
        entity_density = 0.0
        if global_centrality:
            top_entities = _get_top_global_entities(global_centrality, top_n=30)
            text_lower = text.lower()
            entity_hits = sum(1 for e in top_entities if e.lower() in text_lower)
            entity_density = entity_hits / max(len(top_entities), 1)

        # Cross-section entity boost: entities that span multiple sections AND appear here
        cross_section_boost = 0.0
        if global_centrality:
            text_lower = text.lower()
            cross_count = 0
            for ent, data in global_centrality.items():
                if data["section_count"] >= settings.summarization_global_entity_min_sections:
                    if ent.lower() in text_lower:
                        cross_count += data["section_count"]
            cross_section_boost = min(cross_count / 20.0, 1.0) * settings.summarization_cross_section_boost

        # Section importance boost
        section_boost = 0.0
        if section_scores:
            sec = _get_section_key(chunk)
            section_boost = section_scores.get(sec, 0.0) * 0.2

        entity_score = (entity_density * 0.6 + cross_section_boost * 0.4)
        coverage_score = (concept_coverage * 0.6 + rare_boost * 0.4)

        chunk["_concept_coverage"] = concept_coverage
        chunk["_seed_density"] = seed_density
        chunk["_rare_boost"] = rare_boost
        chunk["_position_bonus"] = position_bonus
        chunk["_entity_density"] = entity_density
        chunk["_cross_section_boost"] = cross_section_boost
        chunk["_section_boost"] = section_boost
        chunk["_combined_score"] = (
            w_con * coverage_score
            + w_ent * entity_score
            + w_pos * position_bonus
            + w_cov * chunk.get("score", 0.5)
            + section_boost
        )

    return chunks


def _extract_technical_terms(text: str) -> List[str]:
    text_lower = text.lower()
    found: Set[str] = set()

    for term in SEED_CONCEPTS:
        if " " in term:
            if term in text_lower:
                found.add(term)
        else:
            if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
                found.add(term)

    tech_patterns = re.findall(
        r'\b[A-Z][a-z]*(?:Store|Indexer|Loader|Splitter|Pipeline|'
        r'Service|Engine|Model|System|Module|Agent|Chain|API)\b',
        text,
    )
    found.update(t.lower() for t in tech_patterns)

    return list(found)


# ── Phase 5: Section-balanced chunk selection ────────────────────────────────
# Selects a FIXED number of chunks per section, then fills remaining slots
# with highest-scoring chunks regardless of section.
# This guarantees earlier sections are never starved.

def _select_chunks_balanced(
    chunks: List[dict],
    max_chunks: int,
    min_per_section: int = 3,
    section_scores: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    Select chunks guaranteeing each section gets at least `min_per_section` slots
    (or dynamic allocation based on section_importance if available), with position
    diversity, then fill remaining with highest-scoring chunks across all sections.

    When section_scores are provided, important sections get more slots and minor
    sections get fewer, preventing over-focus on verbose but unimportant sections.
    """
    section_groups: Dict[str, List[dict]] = {}
    for c in chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    section_names = sorted(section_groups.keys())

    if len(section_names) <= 1:
        return chunks[:max_chunks]

    # Dynamic per-section allocation based on importance
    if section_scores and settings.summarization_section_importance:
        max_score = max(section_scores.values()) if section_scores else 1.0
        norm_scores = {
            s: max(section_scores.get(s, 0.0) / max(max_score, 0.01), 0.1)
            for s in section_names
        }
        # Important sections get more slots, minor sections fewer
        per_section_allocation = {}
        for s in section_names:
            if norm_scores[s] >= 0.7:
                per_section_allocation[s] = max(
                    settings.summarization_min_per_important_section,
                    min_per_section,
                )
            elif norm_scores[s] >= 0.4:
                per_section_allocation[s] = max(min_per_section - 1, 1)
            else:
                per_section_allocation[s] = settings.summarization_min_per_minor_section
    else:
        per_section_allocation = {s: min_per_section for s in section_names}

    selected: Set[int] = set()
    result: List[dict] = []

    # Per-section selection with position diversity (hard constraint now)
    for sec in section_names:
        group = sorted(
            section_groups[sec],
            key=lambda c: c.get("_combined_score", 0),
            reverse=True,
        )
        taken = 0
        allocation = per_section_allocation.get(sec, min_per_section)
        for chunk in group:
            if taken >= allocation or len(result) >= max_chunks:
                break
            chi = id(chunk)
            if chi in selected:
                continue
            # Position diversity: hard constraint — skip if too close
            if taken > 0:
                pos = chunk.get("metadata", {}).get("position_ratio", 0.5)
                too_close = False
                for ac in result:
                    if _get_section_key(ac) == sec:
                        existing_pos = ac.get("metadata", {}).get("position_ratio", 0.5)
                        if abs(pos - existing_pos) < 0.15:
                            too_close = True
                            break
                if too_close:
                    # Try next chunk in same section
                    continue
            selected.add(chi)
            result.append(chunk)
            taken += 1

    # Fill remaining slots with highest-scored from any section
    remaining = [
        c for c in chunks
        if id(c) not in selected
    ]
    remaining.sort(key=lambda c: c.get("_combined_score", 0), reverse=True)

    for chunk in remaining:
        if len(result) >= max_chunks:
            break
        chi = id(chunk)
        if chi not in selected:
            selected.add(chi)
            result.append(chunk)

    logger.info(
        f"Section-balanced selection: {len(result)} chunks from "
        f"{len(section_names)} sections "
        f"(dynamic allocation: {per_section_allocation} + position diversity + fill)"
    )
    return result


# ── Phase 6-7: Multi-stage summarization ─────────────────────────────────────
# Phase 6: Per-section summarization — forces the LLM to independently process
#   each section's chunks, preventing any single section from dominating.
# Phase 7: Merge section summaries and generate final 3-point global summary.

_SECTION_SUMMARY_PROMPT = """Below are technical excerpts from the "{section_name}" section of a document (section importance: {importance_label}). For EACH excerpt, write one sentence capturing its key technical contribution.

{section_chunks}

Guidelines:
- Focus on the SINGLE most important technical contribution per excerpt
- Use specific technology names when present (e.g., "Voyage AI", "Pinecone", "Groq")
- Be CONCISE — one sentence per excerpt is sufficient
- If excerpts are repetitive, MERGE them into fewer points
- Output a numbered list (or fewer if some excerpts are redundant)
- Prioritize globally significant systems over implementation details"""

_MERGE_AND_GLOBAL_PROMPT = """Below are summaries of different sections of a technical document. Synthesize them into a comprehensive {num_points}-point technical summary.

MANDATORY COVERAGE — Your summary MUST mention ALL of the following areas if ANY section summary discusses them:
- Embeddings pipeline (Voyage AI / BGE / OpenAI)
- Semantic search and retrieval pipeline
- LLM inference integration (Groq / OpenAI / Ollama)
- Vector storage (Qdrant / Pinecone / Chroma)
- Document chunking
- Clustering / insight analysis
- Schema validation and retry handling
- Cross-encoder reranking
- Query expansion
- Multi-turn conversation
- System architecture

CRITICAL — Avoid these common pitfalls:
1. **NO REPETITION**: Each of the {num_points} Key Contributions MUST describe a DIFFERENT system or concept. If two sections discuss similar concepts, MERGE them into ONE contribution, do NOT list them separately.
2. **COVER THE FULL BREADTH**: Ensure ALL major sections of the document are represented across the {num_points} contributions. If a section discusses a unique concept, it MUST appear in at least one contribution.
3. **GLOBAL SIGNIFICANCE FIRST**: When selecting what to highlight, prioritize concepts that are mentioned ACROSS MULTIPLE sections of the document (they are more central).
4. **CROSS-SECTION RELATIONSHIPS**: Explicitly note when concepts from different sections are related or complementary (e.g., "The X pipeline [section A] integrates with Y [section B] to...").
5. **CONCISE, NON-REPETITIVE LANGUAGE**: Use different sentence structures for each contribution. Avoid starting each point with the same phrase.
6. **BALANCED REPRESENTATION**: If some sections are more detailed, extract their essence rather than reproducing all their details at the expense of other sections.

Section Summaries:
{merged_summaries}

GLOBAL CONCEPT INDEX — concepts that appear across the most document sections:
{concept_index}

Output:
## Technical Summary
[2-3 sentence overview capturing the FULL document breadth — mention the most cross-cutting theme first]

### Key Contributions
1. **[Most globally significant system/concept]** — [what it does and why it matters across multiple sections, 1-2 sentences]
2. **[Second most significant, MUST be from a DIFFERENT area than #1]** — [1-2 sentences]
3. **[Third most significant, MUST be from a DIFFERENT area than #1 and #2]** — [1-2 sentences]

### Architecture & Integration
[How the systems connect and interact — 1-2 paragraphs covering how concepts from different sections relate. Start with the most cross-cutting integration point.]"""


def _generate_section_summaries(
    selected_chunks: List[dict],
    llm,
    section_scores: Optional[Dict[str, float]] = None,
) -> str:
    """Generate a summary for each section's chunks independently.

    When section_scores are provided, passes importance context to the LLM
    so it can weight per-section output accordingly.
    """
    section_groups: Dict[str, List[dict]] = {}
    for c in selected_chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    # Normalize section scores for prompt labels
    imp_labels = {}
    if section_scores and settings.summarization_section_importance:
        max_score = max(section_scores.values()) if section_scores else 1.0
        for sec, score in section_scores.items():
            norm = score / max(max_score, 0.01)
            if norm >= 0.7:
                imp_labels[sec] = "HIGH (core section)"
            elif norm >= 0.4:
                imp_labels[sec] = "MEDIUM (supporting section)"
            else:
                imp_labels[sec] = "LOW (peripheral section)"

    section_summaries: List[str] = []

    for sec in sorted(section_groups.keys()):
        group = section_groups[sec]
        chunk_texts = []
        for i, c in enumerate(group):
            text = c.get("text", "")
            chunk_texts.append(f"Excerpt {i+1}:\n{text[:800]}")

        section_content = "\n\n".join(chunk_texts)
        importance_label = imp_labels.get(sec, "MEDIUM")
        prompt = _SECTION_SUMMARY_PROMPT.format(
            section_name=sec,
            section_chunks=section_content,
            importance_label=importance_label,
        )

        try:
            response = llm.invoke([{"role": "user", "content": prompt}])
            summary = response.content.strip()
            section_summaries.append(f"=== {sec} ===\n{summary}")
            logger.info(f"Section summary for '{sec}' ({importance_label}): {len(summary)} chars")
        except Exception as e:
            logger.warning(f"Section summary for '{sec}' failed: {e}")
            section_summaries.append(f"=== {sec} ===\n[Summary unavailable]")

    return "\n\n".join(section_summaries)


def _identify_globally_significant_concepts(
    concepts: List[dict],
    concept_chunk_map: Dict[int, Set[int]],
    chunks: List[dict],
) -> List[dict]:
    """Rank concepts by how many DISTINCT sections they appear in.
    
    A concept that spans more sections is more globally significant.
    Returns concepts sorted by cross-section coverage (most global first).
    """
    if not concepts or not concept_chunk_map:
        return concepts

    section_of_chunk: dict = {}
    for i, c in enumerate(chunks):
        sec = _get_section_key(c)
        section_of_chunk[i] = sec

    concept_sections: List[tuple] = []
    for ci, chunk_indices in concept_chunk_map.items():
        sections_covered = set()
        for chi in chunk_indices:
            if chi in section_of_chunk:
                sections_covered.add(section_of_chunk[chi])
        concept_sections.append((ci, len(sections_covered), sections_covered))

    concept_sections.sort(key=lambda x: x[1], reverse=True)

    ranked = []
    for ci, num_sections, sections in concept_sections:
        c = dict(concepts[ci])
        c["_num_sections"] = num_sections
        c["_sections"] = sections
        ranked.append(c)

    top5 = [(r.get("name", "?"), r.get("_num_sections", 0)) for r in ranked[:5]]
    top5_str = " | ".join(f"{name} ({n} sections)" for name, n in top5)
    logger.info(f"Global concept significance: {top5_str}")
    return ranked


def _generate_global_from_merged(
    merged_summaries: str,
    concepts: List[dict],
    llm,
    global_concepts: Optional[List[dict]] = None,
    num_points: Optional[int] = None,
) -> str:
    """Generate the final global summary from merged section summaries."""
    num_points = num_points or settings.summarization_concise_max_points

    concept_lines = []
    for i, c in enumerate(concepts):
        concept_lines.append(f"{i+1}. {c['name']} (importance: {c['importance']}/10)")

    # Show cross-section coverage for globally significant concepts
    global_concept_lines = []
    if global_concepts:
        for i, gc in enumerate(global_concepts[:8]):
            sec_count = gc.get("_num_sections", 0)
            tag = f"[{sec_count} sections]" if sec_count > 1 else "[local]"
            global_concept_lines.append(f"  {i+1}. {gc['name']} {tag}")

    concept_index = "\n".join(concept_lines)
    gc_index = "\n".join(global_concept_lines) if global_concept_lines else concept_index

    prompt = _MERGE_AND_GLOBAL_PROMPT.format(
        merged_summaries=merged_summaries,
        concept_index=gc_index,
        num_points=num_points,
    )

    try:
        response = llm.invoke([{"role": "user", "content": prompt}])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Global summary generation failed: {e}")
        return _fallback_summary(concepts)


def _fallback_summary(concepts: List[dict]) -> str:
    lines = ["## Technical Summary"]
    lines.append("Summary generation encountered an error. Key concepts identified:")
    lines.append("")
    for c in sorted(concepts, key=lambda x: x.get("importance", 0), reverse=True):
        lines.append(f"- {c['name']}")
    return "\n".join(lines)


# ── Refinement pass ────────────────────────────────────────────────────────────
# After the initial summary is generated, check if key concepts are missing
# and regenerate with targeted supplementation.

_REFINEMENT_PROMPT = """Below is a technical summary of a document. Your task is to check it for issues and generate an improved version.

ISSUES FOUND:
{issues}

CURRENT SUMMARY:
{current_summary}

Generate an improved version of ## Technical Summary and ### Key Contributions sections that addresses the issues above. Keep the same format:
## Technical Summary
[2-3 sentence overview — start with the most cross-cutting theme]

### Key Contributions
1. **[concept]** — [description — MUST be distinct from other contributions]
2. **[concept]** — [description — MUST be distinct from #1]
3. **[concept]** — [description — MUST be distinct from #1 and #2]

### Architecture & Integration
[How systems connect — highlight cross-section relationships first]

CRITICAL: Each Key Contribution MUST describe a DIFFERENT core system or concept. If any two are similar, merge them."""


def _refine_summary(
    summary: str,
    concepts: List[dict],
    section_summaries: str,
    llm,
    global_concepts: Optional[List[dict]] = None,
) -> str:
    """Check summary coverage and repetition, refine if needed.

    Detects:
      1. Missing high-importance concepts
      2. Repetitive structure across Key Contributions
      3. Over-focus on one area at expense of cross-section concepts
    """
    summary_lower = summary.lower()

    # ── Issue detection ──────────────────────────────────────────────────────
    issues = []

    # Issue 1: Missing high-importance concepts
    missing = []
    for c in sorted(concepts, key=lambda x: x.get("importance", 0), reverse=True):
        if c.get("importance", 0) >= 7:
            name_lower = c["name"].lower()
            keywords_lower = [kw.lower() for kw in c.get("keywords", [])]
            name_found = any(term in summary_lower for term in name_lower.split())
            kw_found = any(kw in summary_lower for kw in keywords_lower)
            if not name_found and not kw_found:
                missing.append(c["name"])

    if len(missing) >= 2:
        issues.append(
            f"MISSING CONCEPTS — The following important concepts are not mentioned:\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    # Issue 2: Check if globally significant (cross-section) concepts are missing
    if global_concepts:
        missing_global = []
        for gc in global_concepts[:5]:
            if gc.get("_num_sections", 0) >= 2:
                name_lower = gc["name"].lower()
                if name_lower not in summary_lower:
                    missing_global.append(gc["name"])
        if len(missing_global) >= 2:
            issues.append(
                f"MISSING GLOBAL CONCEPTS — These span multiple sections but are not in the summary:\n"
                + "\n".join(f"- {m}" for m in missing_global)
            )

    # Issue 3: Check for repetitive contribution patterns
    contrib_section = summary[summary.find("### Key Contributions"):] if "### Key Contributions" in summary else ""
    if contrib_section:
        contrib_lines = [l.strip() for l in contrib_section.split("\n") if l.strip().startswith(("1.", "2.", "3.", "**"))]
        # Check for similar opening phrases
        openings = [l.split("—")[0].strip() if "—" in l else l for l in contrib_lines]
        if len(openings) >= 2:
            word_overlaps = sum(
                1 for i in range(len(openings))
                for j in range(i + 1, len(openings))
                if set(openings[i].lower().split()) & set(openings[j].lower().split())
            )
            if word_overlaps > 2:
                issues.append(
                    "REPETITIVE CONTRIBUTIONS — The Key Contributions have overlapping "
                    "concepts or similar wording. Each MUST describe a DIFFERENT system."
                )

    if len(issues) < 2:
        return summary

    logger.info(f"Refinement: {len(issues)} issues found: {[i.split(':')[0] for i in issues]}")

    try:
        response = llm.invoke([{
            "role": "user",
            "content": _REFINEMENT_PROMPT.format(
                issues="\n\n".join(issues),
                current_summary=summary,
            ),
        }])
        refined = response.content.strip()
        if len(refined) > len(summary) * 0.5:
            logger.info(f"Refinement: summary expanded from {len(summary)} to {len(refined)} chars")
            return refined
    except Exception as e:
        logger.warning(f"Refinement pass failed: {e}")

    return summary


# ── Concept inventory merging ──────────────────────────────────────────────────

def _merge_concept_inventories(
    llm_concepts: List[dict],
    fallback_concepts: List[dict],
) -> List[dict]:
    """Merge LLM-identified concepts with fallback concepts, deduplicating by keyword overlap."""
    if not llm_concepts:
        return fallback_concepts[:20]
    if not fallback_concepts:
        return llm_concepts[:20]

    merged = list(llm_concepts)
    existing_keywords: Set[str] = set()
    for c in llm_concepts:
        for kw in c.get("keywords", []):
            existing_keywords.add(kw.lower())

    added = 0
    for fc in fallback_concepts:
        fc_keywords = set(k.lower() for k in fc.get("keywords", []))
        if not fc_keywords.intersection(existing_keywords):
            merged.append(fc)
            existing_keywords.update(fc_keywords)
            added += 1

    logger.info(
        f"Merged concept inventory: {len(llm_concepts)} LLM + {added} fallback = {len(merged)} total"
    )
    return merged[:25]


# ── Coverage validation ────────────────────────────────────────────────────────

def _check_and_supplement_coverage(
    selected_chunks: List[dict],
    all_chunks: List[dict],
    concepts: Optional[List[dict]] = None,
) -> List[dict]:
    """Check if key technical entities and concepts are covered, supplement if not."""
    COVERAGE_TERMS = [
        "embedding", "semantic", "groq", "voyage", "clustering",
        "validation", "schema", "retry", "reranker", "reranking",
        "vector", "index", "similarity", "search", "retrieval",
        "llm", "pipeline", "chunk", "splitter", "hybrid",
        "bm25", "mmr", "query expansion", "cross-encoder",
        "parsing", "extraction", "architecture", "integration",
    ]

    selected_text = " ".join(c.get("text", "").lower() for c in selected_chunks)

    missing_terms = [t for t in COVERAGE_TERMS if t not in selected_text]

    supplements: List[dict] = []
    selected_ids = set(id(c) for c in selected_chunks)

    # Also check concept coverage if concepts provided
    missing_concepts: List[str] = []
    if concepts:
        concept_names = [c["name"].lower() for c in concepts]
        concept_keywords = []
        for c in concepts:
            concept_keywords.extend(c.get("keywords", []))
        for kw in concept_keywords:
            if kw not in selected_text:
                missing_concepts.append(kw)

    if not missing_terms and not missing_concepts:
        return selected_chunks

    logger.info(
        f"Coverage validation: missing {len(missing_terms)} terms and "
        f"{len(missing_concepts) if missing_concepts else 0} concept keywords"
    )

    for chunk in all_chunks:
        if id(chunk) in selected_ids:
            continue
        text_lower = chunk.get("text", "").lower()
        for term in missing_terms:
            if term in text_lower:
                supplements.append(chunk)
                break
        else:
            # Also check concept keywords
            for kw in missing_concepts[:20]:
                if kw in text_lower:
                    supplements.append(chunk)
                    break

    supplements.sort(key=lambda c: c.get("_combined_score", 0), reverse=True)

    # Deduplicate against already-selected chunks
    seen_ids = set(id(c) for c in selected_chunks)
    unique_supplements = []
    for s in supplements:
        sid = id(s)
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_supplements.append(s)

    result = list(selected_chunks)
    for s in unique_supplements[:8]:
        result.append(s)

    logger.info(
        f"Coverage validation: added {len(result) - len(selected_chunks)} supplement chunks"
    )
    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def hierarchical_summarize(
    source_files: Optional[List[str]] = None,
) -> str:
    """
    Multi-stage hierarchical summarization with enhanced global importance
    awareness, section-balanced coverage, and repetition reduction:

     0a. Global entity centrality — entities spanning multiple sections
     0b. Section importance scoring — importance-weighted section prioritization
     1.  Section-diverse chunk retrieval (evenly sampled across sections, dedup'd)
     2.  Global concept inventory (LLM + fallback seed-term coverage, merged)
     3.  Concept-to-chunk mapping (cross-chunk aggregation)
     4.  Chunk scoring by concept coverage + entity density + importance + position
     5.  Importance-aware section-balanced selection (dynamic per-section allocation)
     6.  Coverage validation and supplementation (expanded term list)
     7.  Per-section summarization with importance-labeled prompts
     8.  Merge section summaries into final global summary (with cross-section
         concept index and repetition avoidance)
     9.  Refinement pass — checks for missing concepts, repetitive patterns,
         and over-focus on narrow subset
    """
    llm = get_llm()

    # Phase 1: Section-diverse chunk retrieval with even sampling & dedup
    chunks = _retrieve_diverse_chunks(
        source_files=source_files,
        max_chunks=settings.summarization_max_chunks * 3,
        min_per_section=4,
    )
    if not chunks:
        return "No content found to summarize."

    logger.info(f"Phase 1: {len(chunks)} chunks from diverse sections (dedup'd)")

    # Phase 0a: Global entity centrality (cross-section analysis)
    global_centrality = _compute_global_entity_centrality(chunks)
    top_global_entities = _get_top_global_entities(global_centrality, top_n=10)
    logger.info(
        f"Phase 0a: Global entity centrality — {len(global_centrality)} entities, "
        f"top global: {top_global_entities[:5]}"
    )

    # Phase 2: Build global concept inventory via LLM + fallback merge
    llm_concepts = _build_concept_inventory(chunks)
    fallback_concepts = _fallback_concepts(chunks)
    concepts = _merge_concept_inventories(llm_concepts, fallback_concepts)
    logger.info(
        f"Phase 2: {len(concepts)} concepts "
        f"({len(llm_concepts)} LLM + {len(fallback_concepts)} fallback)"
    )

    # Phase 3: Map concepts to chunks (cross-chunk aggregation)
    concept_chunk_map = _map_concepts_to_chunks(concepts, chunks)
    total_mappings = sum(len(v) for v in concept_chunk_map.values())
    logger.info(f"Phase 3: {total_mappings} concept-chunk mappings")

    # Phase 0b: Section importance scoring (for importance-aware allocation)
    section_scores = _compute_section_importance(
        chunks, global_centrality, concepts, concept_chunk_map,
    ) if settings.summarization_section_importance else None
    if section_scores:
        important_secs = _get_important_sections(section_scores)
        logger.info(
            f"Phase 0b: {len(important_secs)} important sections, "
            f"{len(section_scores) - len(important_secs)} minor sections"
        )

    # Phase 4: Score chunks by concept coverage + entity density + importance
    chunks = _score_by_concept_coverage(
        chunks, concepts, concept_chunk_map,
        section_scores=section_scores,
        global_centrality=global_centrality,
    )
    if chunks:
        top = chunks[0]
        logger.info(
            f"Phase 4: Top chunk covers "
            f"{top.get('_concept_coverage', 0):.2%} of concepts, "
            f"entity_density={top.get('_entity_density', 0):.2f}, "
            f"position={top.get('_position_bonus', 0):.2f}"
        )

    # Phase 5: Importance-aware section-balanced selection
    selected = _select_chunks_balanced(
        chunks,
        max_chunks=settings.summarization_max_chunks,
        section_scores=section_scores,
    )
    logger.info(f"Phase 5: {len(selected)} chunks selected across sections")

    # Phase 6: Coverage validation and supplementation (with concept awareness)
    selected = _check_and_supplement_coverage(selected, chunks, concepts=concepts)
    logger.info(f"Phase 6: {len(selected)} chunks after coverage validation")

    # Phase 7: Per-section summarization with importance context
    section_summaries = _generate_section_summaries(selected, llm, section_scores=section_scores)
    logger.info(f"Phase 7: Section summaries generated ({len(section_summaries)} chars)")

    # Phase 8: Merge section summaries into final global summary
    # Identify globally significant concepts (spanning multiple sections)
    global_concepts = _identify_globally_significant_concepts(concepts, concept_chunk_map, chunks)
    summary = _generate_global_from_merged(
        section_summaries, concepts, llm,
        global_concepts=global_concepts,
    )
    logger.info(f"Phase 8: Summary generated ({len(summary)} chars)")

    # Phase 9: Refinement pass — checks for missing concepts, repetition, over-focus
    summary = _refine_summary(
        summary, concepts, section_summaries, llm,
        global_concepts=global_concepts,
    )
    logger.info(f"Phase 9: Refined summary ({len(summary)} chars)")

    return summary


# ── Fallback concept detection (when LLM fails) ──────────────────────────────
# Uses SEED_CONCEPTS term coverage across sections to build a concept list.

def _fallback_concepts(chunks: List[dict]) -> List[dict]:
    """Build concept list from seed term coverage across sections."""
    section_groups: Dict[str, list] = {}
    for c in chunks:
        sec = _get_section_key(c)
        section_groups.setdefault(sec, []).append(c)

    section_count = len(section_groups)

    # Count how many DIFFERENT sections each term appears in
    term_section_coverage: Dict[str, int] = {}
    for sec, group in section_groups.items():
        section_text = " ".join(c.get("text", "") for c in group).lower()
        for term in SEED_CONCEPTS:
            if " " in term:
                if term in section_text:
                    term_section_coverage[term] = term_section_coverage.get(term, 0) + 1
            else:
                if re.search(r'\b' + re.escape(term) + r'\b', section_text):
                    term_section_coverage[term] = term_section_coverage.get(term, 0) + 1

    # Build concepts from terms that span multiple sections
    CONCEPT_GROUPINGS = {
        "embedding": "Embedding Pipeline",
        "pinecone": "Vector Storage with Pinecone",
        "voyage": "Embedding Model (Voyage AI)",
        "semantic search": "Semantic Similarity Search",
        "llm": "LLM Inference Integration",
        "groq": "Groq LLM Integration",
        "clustering": "Automated Insight Clustering",
        "cluster": "Automated Insight Clustering",
        "validation": "Schema Validation & Retry Handling",
        "schema": "Schema Validation & Retry Handling",
        "retry": "Schema Validation & Retry Handling",
        "retrieval": "RAG Retrieval Pipeline",
        "reranker": "Cross-Encoder Reranking",
        "multi-turn": "Multi-Turn Conversation Support",
        "conversation": "Multi-Turn Conversation Support",
        "workspace": "RAG Workspace Understanding",
        "copilot": "Knowledge Copilot Architecture",
        "monitoring": "Observability & Monitoring",
        "evaluation": "RAG Quality Evaluation",
        "chunking": "Document Chunking Pipeline",
        "architecture": "System Architecture",
    }

    concept_map: Dict[str, dict] = {}
    for term, concept_name in CONCEPT_GROUPINGS.items():
        coverage = term_section_coverage.get(term, 0)
        if coverage > 0:
            importance = int(5 + 5 * (coverage / max(section_count, 1)))
            if concept_name not in concept_map:
                concept_map[concept_name] = {
                    "name": concept_name,
                    "keywords": [],
                    "importance": 0,
                }
            concept_map[concept_name]["keywords"].append(term)
            concept_map[concept_name]["importance"] = max(
                concept_map[concept_name]["importance"],
                min(importance, 10),
            )

    concepts = sorted(
        concept_map.values(),
        key=lambda c: c["importance"],
        reverse=True,
    )

    logger.info(
        f"Fallback: {len(concepts)} concepts from seed term section-coverage: "
        f"{[c['name'] for c in concepts]}"
    )
    return concepts[:20]
