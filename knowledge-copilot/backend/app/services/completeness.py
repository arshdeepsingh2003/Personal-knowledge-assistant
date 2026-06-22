"""
completeness.py — Answer completeness verification and fact coverage analysis.

Before returning an answer, this module:
1. Extracts key facts, metrics, and entities from ALL retrieved chunks
2. Checks which facts are present in the generated answer
3. Identifies omitted facts (percentages, dates, projections, comparisons)
4. Generates expansion prompts to fill gaps
5. Returns completeness report with missing information
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.completeness")


class Fact:
    def __init__(self, category: str, value: str, chunk_ids: List[int], context: str = ""):
        self.category = category
        self.value = value
        self.chunk_ids = chunk_ids
        self.context = context

    def __repr__(self):
        return f"Fact({self.category}: {self.value})"


def extract_facts_from_chunks(chunks: List[dict]) -> List[Fact]:
    """Extract all factual claims from retrieved chunks."""
    facts: List[Fact] = []
    seen: Set[str] = set()

    for chunk_idx, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text:
            continue

        # Numeric values (percentages, dollar amounts, plain numbers)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\s*(?:percent|%)?\b', text)
        for num in numbers[:10]:
            key = f"num:{num.strip()}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("numeric", num.strip(), [chunk_idx + 1]))

        # Dollar amounts
        dollars = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d+)?[BMK]?\b', text)
        for dol in dollars[:5]:
            key = f"dollar:{dol}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("currency", dol, [chunk_idx + 1]))

        # Percentage patterns (with explicit % sign or "percent" word)
        pcts = re.findall(r'\b\d+(?:\.\d+)?\s*%\b', text)
        for pct in pcts[:5]:
            key = f"pct:{pct}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("percentage", pct.strip(), [chunk_idx + 1]))

        # Named entities (capitalized words)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
        for ent in list(set(entities))[:15]:
            if len(ent) > 4 and ent not in seen:
                key = f"entity:{ent}"
                if key not in seen:
                    seen.add(key)
                    facts.append(Fact("entity", ent, [chunk_idx + 1]))

        # Date-like patterns
        dates = re.findall(r'\b(?:19|20)\d{2}\b', text)
        for date in dates[:5]:
            key = f"date:{date}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("date", date, [chunk_idx + 1]))

        # Comparison patterns ("higher than", "greater than", etc.)
        comparisons = re.findall(
            r'\b(higher|lower|greater|less|more|fewer|better|worse|faster|slower)'
            r'\s+than\b',
            text.lower(),
        )
        for comp in comparisons[:5]:
            key = f"comp:{comp}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("comparison", comp, [chunk_idx + 1]))

        # Projection/forecast patterns
        projections = re.findall(
            r'\b(projected|forecast|expected|estimated|predicted|anticipated)'
            r'\s+.*?\d+',
            text.lower(),
        )
        for proj in projections[:5]:
            key = f"proj:{proj[:60]}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("projection", proj[:80], [chunk_idx + 1]))

        # Ratio/rate patterns
        ratios = re.findall(r'\b\d+(?:\.\d+)?[xX]\b', text)
        for ratio in ratios[:5]:
            key = f"ratio:{ratio}"
            if key not in seen:
                seen.add(key)
                facts.append(Fact("ratio", ratio, [chunk_idx + 1]))

    return facts


def check_answer_completeness(
    answer: str,
    chunks: List[dict],
    query: str,
) -> Dict:
    """Check if the generated answer covers key facts from retrieved chunks.

    Returns:
        dict with:
            - is_complete: bool — whether answer covers sufficient facts
            - coverage_ratio: float — fraction of key facts covered
            - total_facts: int
            - covered_facts: int
            - missing_facts: list of Fact dicts
            - missing_categories: dict of category -> count
            - expansion_suggestions: list of prompt suggestions
    """
    if not answer or not chunks:
        return {
            "is_complete": True,
            "coverage_ratio": 1.0,
            "total_facts": 0,
            "covered_facts": 0,
            "missing_facts": [],
            "missing_categories": {},
            "expansion_suggestions": [],
        }

    facts = extract_facts_from_chunks(chunks)
    if not facts:
        return {
            "is_complete": True,
            "coverage_ratio": 1.0,
            "total_facts": 0,
            "covered_facts": 0,
            "missing_facts": [],
            "missing_categories": {},
            "expansion_suggestions": [],
        }

    answer_lower = answer.lower()
    covered: List[Fact] = []
    missing: List[Fact] = []

    for fact in facts:
        val_lower = fact.value.lower().strip()
        val_clean = val_lower.replace("%", "").replace("$", "").replace(",", "").strip()

        is_covered = (
            val_lower in answer_lower
            or val_clean in answer_lower
            or any(
                str(cid) in re.findall(r'\[(\d+)\]', answer)
                for cid in fact.chunk_ids
            )
        )

        if is_covered:
            covered.append(fact)
        else:
            missing.append(fact)

    total = len(facts)
    covered_count = len(covered)
    coverage_ratio = covered_count / max(total, 1)

    missing_categories: Dict[str, int] = {}
    for f in missing:
        missing_categories[f.category] = missing_categories.get(f.category, 0) + 1

    expansion_suggestions = []
    if missing_categories.get("percentage", 0) > 1:
        expansion_suggestions.append(
            "The context contains percentages not included in the answer. "
            "Check if any percentage values are relevant."
        )
    if missing_categories.get("currency", 0) > 1:
        expansion_suggestions.append(
            "The context contains financial figures not included in the answer. "
            "Consider adding relevant monetary values."
        )
    if missing_categories.get("comparison", 0) > 0:
        expansion_suggestions.append(
            "The context contains comparisons not reflected in the answer. "
            "Add relevant comparative statements."
        )
    if missing_categories.get("projection", 0) > 0:
        expansion_suggestions.append(
            "The context contains projections/forecasts not included. "
            "Include relevant forward-looking statements."
        )
    if missing_categories.get("date", 0) > 1:
        expansion_suggestions.append(
            "The context contains date references not in the answer. "
            "Add relevant temporal context."
        )
    if missing_categories.get("numeric", 0) > 2:
        expansion_suggestions.append(
            "Multiple numeric values from context are missing from the answer. "
            "Verify all relevant statistics are included."
        )

    is_complete = (
        coverage_ratio >= 0.6
        or (total <= 3 and coverage_ratio >= 0.5)
    )

    if missing:
        top_missing = sorted(missing, key=lambda f: f.category)[:10]
        missing_summary = [
            {
                "category": f.category,
                "value": f.value,
                "source_chunks": f.chunk_ids,
            }
            for f in top_missing
        ]
    else:
        missing_summary = []

    result = {
        "is_complete": is_complete,
        "coverage_ratio": round(coverage_ratio, 3),
        "total_facts": total,
        "covered_facts": covered_count,
        "missing_facts": missing_summary,
        "missing_categories": dict(
            sorted(missing_categories.items(), key=lambda x: x[1], reverse=True)
        ),
        "expansion_suggestions": expansion_suggestions,
    }

    log_level = logger.info if is_complete else logger.warning
    log_level(
        f"Completeness: {'PASS' if is_complete else 'NEEDS EXPANSION'} "
        f"({covered_count}/{total} facts, {coverage_ratio:.0%})"
    )
    if not is_complete and missing_categories:
        cat_summary = ", ".join(f"{k}: {v}" for k, v in missing_categories.items())
        logger.warning(f"Missing facts by category: {cat_summary}")

    return result


def generate_expansion_prompt(
    completeness_result: Dict,
    original_context: str,
) -> str:
    """Generate a prompt to expand the answer with missing information.

    Returns a string that can be appended to the original prompt to
    request the LLM to incorporate missing facts.
    """
    if completeness_result.get("is_complete", True):
        return ""

    missing = completeness_result.get("missing_facts", [])
    if not missing:
        return ""

    # Group missing facts by category
    by_category: Dict[str, List[str]] = {}
    for m in missing:
        cat = m.get("category", "other")
        by_category.setdefault(cat, []).append(m["value"])

    parts = []
    parts.append("IMPORTANT: Your previous answer is MISSING some key facts present in the context.")

    for cat, values in by_category.items():
        cat_label = {
            "numeric": "numerical values",
            "currency": "monetary figures",
            "percentage": "percentages",
            "entity": "named entities",
            "date": "dates/time periods",
            "comparison": "comparative statements",
            "projection": "projections/forecasts",
            "ratio": "ratios",
        }.get(cat, cat)
        values_str = ", ".join(v for v in values[:5])
        parts.append(f"- Missing {cat_label}: {values_str}")

    parts.append("Please expand your answer to include ALL relevant data from the context above.")
    parts.append("Do NOT remove any information already present. Only add missing facts.")

    return "\n\n".join(parts)


def expand_answer_with_missing_facts(
    answer: str,
    chunks: List[dict],
    query: str,
    llm_generate_fn=None,
) -> str:
    """Check completeness and optionally expand the answer with missing facts.

    If llm_generate_fn is provided, uses it to regenerate an expanded answer.
    Otherwise, returns the original answer with a note about missing facts.
    """
    if not settings.completeness_check_enabled:
        return answer

    result = check_answer_completeness(answer, chunks, query)

    if result.get("is_complete", True):
        return answer

    if llm_generate_fn is not None:
        expansion_prompt = generate_expansion_prompt(result, "")
        if expansion_prompt:
            try:
                expanded = llm_generate_fn(expansion_prompt)
                if expanded and len(expanded) > len(answer) * 0.5:
                    logger.info(
                        f"Answer expanded with missing facts "
                        f"({result.get('coverage_ratio', 0):.0%} coverage before)"
                    )
                    return expanded
            except Exception as e:
                logger.warning(f"Answer expansion failed: {e}")

    return answer
