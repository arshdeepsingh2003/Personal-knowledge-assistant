"""
synthesis.py — Pre-generation cross-chunk synthesis module.

Identifies relationships, complementarities, and contradictions across
retrieved chunks before passing context to the LLM. Generates a synthesis
context that helps the LLM produce more coherent cross-document answers.
"""

import logging
import re
from collections import Counter
from typing import List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.synthesis")


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r'\w+', text.lower()))


def _extract_named_entities(text: str) -> List[str]:
    entities: List[str] = []
    patterns = [
        r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b',
        r'\b\d+(?:\.\d+)?%?(?=\b|[^a-zA-Z0-9])',
        r'\b[A-Z]{2,}\b',
        r'\$\d+(?:,\d{3})*(?:\.\d+)?',
    ]
    for pat in patterns:
        entities.extend(re.findall(pat, text))
    return list(set(entities))


def _chunk_overlap_score(text_a: str, text_b: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / max(len(union), 1)


def _shared_entities(text_a: str, text_b: str) -> List[str]:
    entities_a = set(_extract_named_entities(text_a))
    entities_b = set(_extract_named_entities(text_b))
    shared = entities_a & entities_b
    return [e for e in shared if not e.isdigit()]


def _is_complementary(text_a: str, text_b: str) -> Tuple[bool, str]:
    entities_a = set(_extract_named_entities(text_a))
    entities_b = set(_extract_named_entities(text_b))
    shared = entities_a & entities_b

    if not shared:
        return False, ""

    unique_to_a = entities_a - entities_b
    unique_to_b = entities_b - entities_a

    if unique_to_a and unique_to_b:
        shared_str = ", ".join(sorted(shared)[:3])
        return True, f"complementary on [{shared_str}]"

    return False, ""


def _is_contradictory(text_a: str, text_b: str) -> Tuple[bool, str]:
    numbers_a = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text_a))
    numbers_b = set(re.findall(r'\b\d+(?:\.\d+)?%?\b', text_b))
    shared_numbers = numbers_a & numbers_b
    if shared_numbers:
        return True, f"shared numeric values: {', '.join(sorted(shared_numbers)[:3])}"
    return False, ""


def build_synthesis_context(
    chunks: List[dict],
    query: str,
    max_chars: Optional[int] = None,
) -> str:
    """
    Analyze relationships across retrieved chunks and build a synthesis context.
    Identifies complementary chunks (same entities, different details),
    contradictory chunks (conflicting numbers), and key themes.
    """
    if not chunks or len(chunks) < 2:
        return ""

    max_chars = max_chars or settings.synthesis_max_context_chars

    query_entities = set(_extract_named_entities(query))
    query_tokens = _tokenize(query)

    relationships: List[str] = []
    identified_pairs: Set[Tuple[int, int]] = set()

    for i in range(len(chunks)):
        for j in range(i + 1, len(chunks)):
            if (i, j) in identified_pairs or (j, i) in identified_pairs:
                continue
            text_i = chunks[i].get("text", "")
            text_j = chunks[j].get("text", "")
            if not text_i or not text_j:
                continue

            overlap = _chunk_overlap_score(text_i, text_j)
            if overlap > 0.6:
                relationships.append(
                    f"Chunks [{i+1}] and [{j+1}] overlap significantly ({overlap:.0%} token similarity)."
                )
                identified_pairs.add((i, j))

            compl, reason = _is_complementary(text_i, text_j)
            if compl:
                relationships.append(
                    f"Chunks [{i+1}] and [{j+1}] are {reason}."
                )
                identified_pairs.add((i, j))

            contra, reason_c = _is_contradictory(text_i, text_j)
            if contra:
                relationships.append(
                    f"Chunks [{i+1}] and [{j+1}] share {reason_c}."
                )
                identified_pairs.add((i, j))

    query_relevant_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        chunk_tokens = _tokenize(text)
        query_overlap = len(query_tokens & chunk_tokens) / max(len(query_tokens), 1)
        if query_overlap > 0.15:
            query_relevant_chunks.append(i + 1)

    synthesis_parts: List[str] = []

    if relationships:
        synthesis_parts.append("Cross-chunk relationships detected:\n" + "\n".join(relationships))

    if query_relevant_chunks:
        synthesis_parts.append(
            f"Chunks most relevant to query: [{', '.join(str(c) for c in query_relevant_chunks)}]"
        )

    synthesis = "\n\n".join(synthesis_parts)
    if len(synthesis) > max_chars:
        synthesis = synthesis[:max_chars] + "\n...[truncated]"

    if synthesis:
        logger.info(
            f"Synthesis: {len(relationships)} relationships, "
            f"{len(query_relevant_chunks)} query-relevant chunks"
        )

    return synthesis


def extract_synthesis_hints(
    synthesis_context: str,
    chunks: List[dict],
) -> List[str]:
    """
    Extract specific synthesis hints from the analysis for LLM prompt injection.
    """
    hints: List[str] = []

    if not chunks:
        return hints

    sources = set()
    sections = set()
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        src = meta.get("file_name", meta.get("source", ""))
        sec = meta.get("heading", meta.get("section", ""))
        if src:
            sources.add(src)
        if sec:
            sections.add(sec)

    if len(sources) > 1:
        hints.append(
            f"Note: The retrieved context includes content from {len(sources)} documents. "
            f"Only use information that is directly relevant to the question."
        )
    if len(sections) > 1:
        hints.append(
            f"Note: The retrieved context includes content from {len(sections)} sections. "
            f"Only use information that is directly relevant to the question."
        )
    if "contradict" in synthesis_context.lower():
        hints.append(
            "Note: Different chunks may present different data points. "
            "Present all perspectives with their respective source citations."
        )
    if "complementary" in synthesis_context.lower():
        hints.append(
            "Note: Some chunks provide complementary information on the same topic. "
            "Combine details from complementary chunks for a complete answer."
        )

    return hints
