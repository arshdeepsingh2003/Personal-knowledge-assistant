"""
memory_manager.py — Smarter conversation memory management for multi-turn queries.

Replaces the simple "last N turns" approach with:
1. Dynamic windowing based on conversation complexity
2. Automatic conversation summarization for long chats
3. Entity and topic extraction for cross-turn context injection
4. Relevance-based history pruning
"""

import logging
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.memory")


def get_relevant_history(
    messages: List[dict],
    current_query: str,
    max_turns: Optional[int] = None,
) -> List[dict]:
    """
    Select the most relevant conversation history for the current query.
    Instead of just taking the last N turns, this function:
    1. Extracts entities and terms from the current query
    2. Scores each history turn by relevance to the current query
    3. Returns the most relevant turns up to max_turns

    Args:
        messages: Full conversation history [{role, content}, ...]
        current_query: The user's current question
        max_turns: Maximum turns to include (default: config)

    Returns:
        List of relevant history messages [{role, content}, ...]
    """
    max_turns = max_turns or settings.memory_max_turns

    if not messages:
        return []

    query_terms = _extract_key_terms(current_query)
    query_entities = _extract_entities(current_query)

    scored: List[tuple] = []
    for i, msg in enumerate(messages):
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        msg_terms = _extract_key_terms(content)
        msg_entities = _extract_entities(content)

        term_overlap = len(query_terms & msg_terms)
        entity_overlap = len(query_entities & msg_entities)
        recency = i / max(len(messages) - 1, 1)

        relevance = (
            term_overlap * 0.3
            + entity_overlap * 0.4
            + recency * 0.3
        )
        scored.append((relevance, i, msg))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = scored[:max_turns]
    selected.sort(key=lambda x: x[1])

    result = [msg for _, _, msg in selected]

    logger.info(
        f"Memory: selected {len(result)}/{len(messages)} turns "
        f"(query terms: {len(query_terms)}, entities: {len(query_entities)})"
    )

    return result


def _extract_key_terms(text: str) -> Set[str]:
    text_lower = text.lower()
    words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "this", "that", "with", "from", "they", "what", "when", "where",
        "which", "their", "there", "about", "would", "could", "should",
        "your", "some", "such", "than", "then", "them", "these",
    }
    return set(w for w in words if w not in stop_words)


def _extract_entities(text: str) -> Set[str]:
    entities = re.findall(
        r'\b[A-Z][A-Za-z]*(?:\s+(?:and|&)?\s*[A-Z][A-Za-z]*)*\b',
        text,
    )
    return set(e for e in entities if len(e) > 2)


def needs_compression(messages: List[dict]) -> bool:
    """Check if the conversation history is long enough to need compression."""
    user_turns = sum(1 for m in messages if m.get("role") == "user")
    return user_turns >= settings.memory_compression_threshold


def compress_history(
    messages: List[dict],
    max_summary_tokens: Optional[int] = None,
) -> str:
    """
    Generate a compressed summary of the conversation history.
    Returns a text summary that can be injected as a system-level context.
    """
    max_summary_tokens = max_summary_tokens or settings.memory_summary_max_tokens

    if not messages:
        return ""

    user_messages = [m for m in messages if m.get("role") == "user"]
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]

    topics: Set[str] = set()
    entities: Set[str] = set()
    for msg in messages:
        content = msg.get("content", "")
        topics.update(_extract_key_terms(content))
        entities.update(_extract_entities(content))

    top_topics = sorted(topics, key=lambda t: sum(
        1 for m in messages if t in m.get("content", "").lower()
    ), reverse=True)[:10]

    top_entities = sorted(entities, key=lambda e: sum(
        1 for m in messages if e in m.get("content", "")
    ), reverse=True)[:10]

    summary_parts = [
        f"Conversation summary: {len(user_messages)} user turns, {len(assistant_messages)} assistant turns.",
    ]

    if top_topics:
        summary_parts.append(f"Key topics discussed: {', '.join(top_topics)}")
    if top_entities:
        summary_parts.append(f"Key entities mentioned: {', '.join(top_entities)}")

    compressed = " | ".join(summary_parts)
    if len(compressed) > max_summary_tokens * 4:
        compressed = compressed[:max_summary_tokens * 4] + "..."

    logger.info(
        f"Memory compression: {len(messages)} turns → {len(compressed)} chars "
        f"({len(top_topics)} topics, {len(top_entities)} entities)"
    )

    return compressed


def build_memory_context(
    messages: List[dict],
    current_query: str,
) -> Dict:
    """
    Build the full memory context for LLM injection.
    Returns:
        - history: relevant history turns for the prompt
        - summary: compressed summary (if history is long)
        - entities: tracked entities from past turns
    """
    history = get_relevant_history(messages, current_query)

    result = {
        "history": history,
        "summary": "",
        "entities": {},
    }

    if needs_compression(messages):
        result["summary"] = compress_history(messages)

    if settings.memory_entity_tracking:
        all_entities: Dict[str, int] = {}
        for msg in messages:
            for ent in _extract_entities(msg.get("content", "")):
                all_entities[ent] = all_entities.get(ent, 0) + 1

        result["entities"] = dict(
            sorted(all_entities.items(), key=lambda x: x[1], reverse=True)[:15]
        )

    return result
