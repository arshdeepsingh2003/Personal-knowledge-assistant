"""
query_analyzer.py — Ambiguity detection, adversarial filtering, and query analysis.

Pre-processes user queries before they enter the retrieval pipeline to:
1. Detect ambiguous or under-specified queries
2. Filter adversarial or malicious inputs
3. Generate disambiguation prompts when needed
4. Classify query intent for routing optimization
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.query_analyzer")

AMBIGUITY_PATTERNS: List[Tuple[str, str, float]] = [
    (r'\b(it|they|this|that|these|those)\b', "vague_pronoun", 0.3),
    (r'\b(something|anything|things|stuff|whatnot)\b', "vague_term", 0.4),
    (r'^.{0,15}$', "too_short", 0.5),
    (r'^\s*\?\s*$', "question_mark_only", 0.6),
    (r'\b(help|explain|tell|describe|elaborate)\s+(me|us)?\s*(about|on)?\s*$', "open_ended", 0.3),
    (r'\b(and|or)\s+(and|or)\b', "conjunction_chain", 0.2),
    (r'\b(\w+)\s+\1\b', "repeated_word", 0.2),
]


ADVERSARIAL_PATTERNS: List[Tuple[str, str, float]] = [
    (r'\b(ignore|disregard|bypass|override)\s+(the\s+)?(above|previous|context|instructions)\b', "context_override", 0.8),
    (r'\b(system|secret|hidden)\s*(prompt|instruction|command)\b', "prompt_leak", 0.9),
    (r'\b(who\s+created|who\s+made|who\s+built)\s+you\b', "identity_probe", 0.3),
    (r'\b(repeat|say|output)\s+(the\s+)?(above|previous|whole|entire)\b', "context_extraction", 0.7),
    (r'(?:role.play|roleplay|pretend|act\s+as|imagine\s+you.are)', "roleplay", 0.6),
    (r'(?:jailbreak|jail.break|prompt.injection)', "jailbreak_attempt", 1.0),
    (r'(?:hack|inject|exploit|vulnerability)\s+(?:the\s+)?(?:prompt|system|model)', "exploit_attempt", 1.0),
    (r'(?:DAN|do.anything.now|unfiltered|ungoverned)', "known_jailbreak", 1.0),
]


INTENT_PATTERNS: Dict[str, List[str]] = {
    "factual": [
        r'\b(what|who|when|where|which|how\s+(many|much|often|long))\b',
        r'\b(define|describe|explain|list|name|identify)\b',
    ],
    "comparison": [
        r'\b(compare|contrast|difference|similar|versus|vs\.|better|worse)\b',
        r'\b(pros?|cons?|advantages?|disadvantages?|trade.?offs?)\b',
    ],
    "procedural": [
        r'\b(how\s+(to|do|can|would|should))\b',
        r'\b(steps?|process|procedure|method|guide|tutorial)\b',
    ],
    "analytical": [
        r'\b(why|analyze|evaluate|assess|synthesize|interpret)\b',
        r'\b(implications?|significance|meaning|reason|cause)\b',
    ],
    "summarization": [
        r'\b(summarize|summary|overview|recap|gist|key.?points?|main.?ideas?)\b',
    ],
}


def analyze_query(query: str) -> Dict:
    """
    Analyze a user query for ambiguity, adversarial content, and intent.

    Returns:
        dict with:
            - is_ambiguous: bool
            - ambiguity_score: float 0-1
            - ambiguity_reasons: list[str]
            - is_adversarial: bool
            - adversarial_score: float 0-1
            - adversarial_reasons: list[str]
            - intent: str (factual|comparison|procedural|analytical|summarization|unknown)
            - disambiguation_prompt: str (if ambiguous)
            - needs_clarification: bool
            - query_entities: list[str]
    """
    result: Dict = {
        "is_ambiguous": False,
        "ambiguity_score": 0.0,
        "ambiguity_reasons": [],
        "is_adversarial": False,
        "adversarial_score": 0.0,
        "adversarial_reasons": [],
        "intent": "unknown",
        "disambiguation_prompt": "",
        "needs_clarification": False,
        "query_entities": [],
    }

    if not query or not query.strip():
        result["is_ambiguous"] = True
        result["ambiguity_score"] = 1.0
        result["ambiguity_reasons"].append("Empty query")
        result["needs_clarification"] = True
        return result

    query_lower = query.lower().strip()

    # ── Ambiguity detection ─────────────────────────────────────────────────
    if settings.query_ambiguity_detection:
        max_score = 0.0
        reasons: List[str] = []
        for pattern, reason, weight in AMBIGUITY_PATTERNS:
            if re.search(pattern, query_lower):
                max_score = max(max_score, weight)
                if weight > 0.3:
                    reasons.append(reason)

        result["ambiguity_score"] = round(max_score, 2)
        result["ambiguity_reasons"] = reasons
        result["is_ambiguous"] = max_score >= settings.query_ambiguity_threshold

        if result["is_ambiguous"]:
            result["needs_clarification"] = max_score >= 0.6
            if result["needs_clarification"]:
                result["disambiguation_prompt"] = (
                    f"Your query appears to be ambiguous. "
                    f"Could you please provide more specific details? "
                    f"For example, what specific aspect of '{query}' are you interested in?"
                )

    # ── Adversarial detection ───────────────────────────────────────────────
    if settings.query_adversarial_filtering:
        max_adv_score = 0.0
        adv_reasons: List[str] = []
        for pattern, reason, weight in ADVERSARIAL_PATTERNS:
            if re.search(pattern, query_lower):
                max_adv_score = max(max_adv_score, weight)
                adv_reasons.append(reason)

        result["adversarial_score"] = round(max_adv_score, 2)
        result["adversarial_reasons"] = adv_reasons
        result["is_adversarial"] = max_adv_score >= 0.7

    # ── Intent classification ──────────────────────────────────────────────
    best_intent = "unknown"
    best_score = 0
    for intent, patterns in INTENT_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, query_lower))
        if matches > best_score:
            best_score = matches
            best_intent = intent
    result["intent"] = best_intent

    # ── Entity extraction ──────────────────────────────────────────────────
    entities = re.findall(
        r'\b[A-Z][A-Za-z]*(?:\s+(?:and|&)?\s*[A-Z][A-Za-z]*)*\b',
        query,
    )
    result["query_entities"] = list(set(e for e in entities if len(e) > 2))

    logger.info(
        f"Query: ambiguous={result['is_ambiguous']}({result['ambiguity_score']}), "
        f"adversarial={result['is_adversarial']}({result['adversarial_score']}), "
        f"intent={result['intent']}"
    )

    return result


def clarify_query(query: str, analysis: Dict) -> str:
    """
    If the query is ambiguous, generate a clarified version by extracting
    entities and intent to create a more specific search query.
    """
    if not analysis.get("is_ambiguous"):
        return query

    entities = analysis.get("query_entities", [])
    intent = analysis.get("intent", "unknown")

    clarified = query

    if entities:
        entity_hint = " ".join(entities[:3])
        clarified = f"{query} related to {entity_hint}"

    if intent and intent != "unknown":
        intent_labels = {
            "factual": "details about",
            "comparison": "comparison of",
            "procedural": "process for",
            "analytical": "analysis of",
            "summarization": "overview of",
        }
        prefix = intent_labels.get(intent, "")
        if prefix and prefix not in clarified.lower():
            clarified = f"{prefix} {clarified}"

    logger.info(f"Clarified query: '{query}' → '{clarified}'")
    return clarified
