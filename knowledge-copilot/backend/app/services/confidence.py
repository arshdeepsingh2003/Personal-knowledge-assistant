"""
confidence.py — Hallucination prevention and confidence estimation layer.

After the LLM generates an answer, this module:
1. Extracts factual claims (entities, numbers, relationships) from the answer
2. Verifies each claim against the retrieved context chunks
3. Produces a per-claim confidence score and an overall confidence estimate
4. Flags unsupported claims as potential hallucinations
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.confidence")


Claim = Dict[str, object]


def _extract_claims(answer: str) -> List[Claim]:
    claims: List[Claim] = []

    numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\s*(?:percent|%)?\b', answer)
    for num in numbers[:20]:
        claims.append({
            "type": "numeric",
            "value": num.strip(),
            "text": _get_surrounding_text(answer, num, 60),
        })

    entities = re.findall(
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
        answer,
    )
    for ent in set(entities[:30]):
        if len(ent) > 2:
            claims.append({
                "type": "entity",
                "value": ent,
                "text": _get_surrounding_text(answer, ent, 60),
            })

    comparisons = re.findall(
        r'(?:higher|lower|greater|less|more|fewer|better|worse)\s+than\s+\w+',
        answer.lower(),
    )
    for comp in comparisons[:10]:
        claims.append({
            "type": "comparison",
            "value": comp,
            "text": _get_surrounding_text(answer, comp, 80),
        })

    return claims


def _get_surrounding_text(text: str, target: str, window: int = 60) -> str:
    idx = text.lower().find(target.lower())
    if idx == -1:
        return target
    start = max(0, idx - window)
    end = min(len(text), idx + len(target) + window)
    return text[start:end].strip()


def _verify_numeric_claim(
    claim_value: str,
    context_text: str,
) -> Tuple[bool, float]:
    clean_claim = claim_value.replace("%", "").replace(",", "").replace(" ", "").lower()
    clean_context = context_text.replace(",", "").lower()
    if clean_claim in clean_context:
        return True, 1.0
    context_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', context_text)
    claim_num = re.sub(r'[^0-9.]', '', claim_value)
    if claim_num and claim_num in context_numbers:
        return True, 0.9
    return False, 0.0


def _verify_entity_claim(
    claim_value: str,
    context_text: str,
) -> Tuple[bool, float]:
    context_lower = context_text.lower()
    if claim_value.lower() in context_lower:
        return True, 1.0
    words = claim_value.split()
    if len(words) > 1:
        matches = sum(1 for w in words if w.lower() in context_lower)
        ratio = matches / len(words)
        if ratio >= settings.citation_min_entity_overlap:
            return True, ratio
    return False, 0.0


def _verify_claim_against_chunks(
    claim: Claim,
    chunks: List[dict],
) -> Tuple[bool, float, str]:
    claim_type = claim.get("type", "")
    claim_value = claim.get("value", "")
    claim_text = claim.get("text", "")

    best_score = 0.0
    best_source = ""

    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        if claim_type == "numeric":
            supported, score = _verify_numeric_claim(claim_value, chunk_text)
        elif claim_type == "entity":
            supported, score = _verify_entity_claim(claim_value, chunk_text)
        else:
            supported, score = _verify_entity_claim(claim_value, chunk_text)

        if supported and score > best_score:
            best_score = score
            meta = chunk.get("metadata", {})
            src = meta.get("file_name", meta.get("source", "unknown"))
            sec = meta.get("heading", meta.get("section", ""))
            best_source = f"{src}{' [' + sec + ']' if sec else ''}"

    return best_score >= settings.citation_min_entity_overlap, best_score, best_source


def estimate_confidence(
    answer: str,
    chunks: List[dict],
) -> Dict:
    """
    Analyze the LLM-generated answer against retrieved context chunks.
    Returns per-claim verification results and overall confidence score.

    Args:
        answer: The LLM-generated answer text
        chunks: The retrieved context chunks (dicts with 'text' and 'metadata')

    Returns:
        dict with:
            - overall_confidence: float 0-1
            - claims_verified: int
            - claims_failed: int
            - unsupported_claims: list of dicts
            - verified_claims: list of dicts
            - warnings: list of strings
    """
    if not answer or not chunks:
        return {
            "overall_confidence": 0.0,
            "claims_verified": 0,
            "claims_failed": 0,
            "unsupported_claims": [],
            "verified_claims": [],
            "warnings": ["No answer or context to verify against"],
        }

    claims = _extract_claims(answer)
    if not claims:
        return {
            "overall_confidence": 1.0,
            "claims_verified": 0,
            "claims_failed": 0,
            "unsupported_claims": [],
            "verified_claims": [],
            "warnings": ["No extractable claims found in answer"],
        }

    verified: List[Claim] = []
    unsupported: List[Claim] = []

    for claim in claims:
        supported, score, source = _verify_claim_against_chunks(claim, chunks)
        claim["supported"] = supported
        claim["confidence"] = round(score, 3)
        claim["source"] = source
        if supported:
            verified.append(claim)
        else:
            unsupported.append(claim)

    total_claims = len(claims)
    verified_count = len(verified)
    overall = verified_count / max(total_claims, 1)

    warnings = []
    if unsupported:
        unsup_types = {}
        for uc in unsupported:
            t = uc.get("type", "unknown")
            unsup_types[t] = unsup_types.get(t, 0) + 1
        type_summary = ", ".join(f"{k}: {v}" for k, v in unsup_types.items())
        warnings.append(
            f"{len(unsupported)}/{total_claims} claims unsupported in context ({type_summary})"
        )

    if overall < settings.confidence_threshold:
        warnings.append(
            f"Low confidence ({overall:.0%}): answer may contain hallucinations"
        )

    result = {
        "overall_confidence": round(overall, 3),
        "claims_verified": verified_count,
        "claims_failed": len(unsupported),
        "total_claims": total_claims,
        "unsupported_claims": [
            {"type": c.get("type"), "value": c.get("value"), "text": c.get("text")}
            for c in unsupported
        ],
        "verified_claims": [
            {"type": c.get("type"), "value": c.get("value"), "confidence": c.get("confidence")}
            for c in verified
        ],
        "warnings": warnings,
    }

    logger.info(
        f"Confidence: {overall:.0%} ({verified_count}/{total_claims} claims verified)"
    )
    if warnings:
        for w in warnings:
            logger.warning(f"Confidence warning: {w}")

    return result


def check_citation_grounding(
    answer: str,
    sources: list,
) -> Dict:
    """
    Validate that source citations in the answer match actual provided sources.
    Checks that every [N] reference in the answer corresponds to a provided source.
    """
    if not answer or not sources:
        return {
            "citations_valid": True,
            "cited_indices": [],
            "valid_indices": [],
            "invalid_indices": [],
            "warnings": [],
        }

    cited = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
    valid_sources = set(range(1, len(sources) + 1))

    valid = cited & valid_sources
    invalid = cited - valid_sources

    warnings = []
    if invalid:
        warnings.append(
            f"Answer cites invalid source numbers: {sorted(invalid)}. "
            f"Valid range: 1-{len(sources)}"
        )

    return {
        "citations_valid": len(invalid) == 0,
        "cited_indices": sorted(cited),
        "valid_indices": sorted(valid),
        "invalid_indices": sorted(invalid),
        "warnings": warnings,
    }
