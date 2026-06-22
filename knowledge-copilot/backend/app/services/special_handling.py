"""
special_handling.py — Document-type-aware retrieval and answer generation.

Provides specialized handling for:
  - Financial reports: numbers, projections, comparisons, tables
  - Research papers: abstracts, methods, results, citations
  - Technical documentation: APIs, code, configuration, parameters
  - Structured PDFs: sections, tables, appendices
  - Multi-page references: cross-page context preservation
"""

import logging
import re
from typing import Dict, List, Optional

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.special_handling")


DOCUMENT_TYPE_PATTERNS = {
    "financial_report": [
        r'\b(revenue|profit|loss|income|balance sheet|p&l|cash flow)\b',
        r'\b(ebitda|ebit|net income|gross margin|operating margin)\b',
        r'\b(annual report|quarterly|10-k|10-q|filing|sec)\b',
        r'\b(arr|mrr|subscription|recurring revenue|churn)\b',
        r'\b(guidance|forecast|outlook|projection|estimates)\b',
    ],
    "research_paper": [
        r'\b(abstract|introduction|methodology|results|discussion|conclusion)\b',
        r'\b(equation|algorithm|proposed|experiment|evaluation)\b',
        r'\b(state-of-the-art|baseline|ablation|benchmark)\b',
        r'\b(paper|study|research|contribution|novel)\b',
        r'\b(dataset|metric|accuracy|precision|recall|f1)\b',
    ],
    "technical_documentation": [
        r'\b(api|endpoint|request|response|parameter|configuration)\b',
        r'\b(installation|setup|deployment|integration|tutorial)\b',
        r'\b(function|method|class|interface|syntax|example)\b',
        r'\b(command|flag|option|argument|return value|error code)\b',
    ],
    "structured_pdf": [
        r'\b(section|chapter|appendix|exhibit|figure|table)\b',
        r'\b(page\s+\d+|p\.\s*\d+|pp\.\s*\d+)\b',
        r'\b(reference|see also|footnote|endnote)\b',
    ],
}


def detect_document_types(chunks: List[dict]) -> List[str]:
    """Detect document type(s) from chunk content.

    Returns list of detected types (financial_report, research_paper, etc.)
    """
    if not chunks:
        return []

    combined_text = " ".join(
        c.get("text", "") for c in chunks[:20]
    ).lower()

    detected = []
    for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        matches = sum(1 for p in patterns if re.search(p, combined_text))
        threshold = 3 if doc_type != "structured_pdf" else 2
        if matches >= threshold:
            detected.append(doc_type)

    return detected


def get_special_handling_hints(
    chunks: List[dict],
    query: str,
) -> List[str]:
    """Generate type-specific hints for the LLM prompt.

    Based on detected document type(s), returns instructions to help
    the LLM better handle the specific document format.
    """
    doc_types = detect_document_types(chunks)
    hints: List[str] = []

    q_lower = query.lower()
    has_numbers = bool(re.search(r'\b\d+', q_lower))
    has_table_ref = bool(re.search(r'\b(table|row|column|cell|spreadsheet)\b', q_lower))

    for doc_type in doc_types:
        if doc_type == "financial_report":
            hints.append(
                "This content appears to be from a FINANCIAL REPORT. "
                "Pay close attention to monetary values, percentages, "
                "and year-over-year comparisons. Extract all relevant "
                "financial metrics with their exact values and time periods."
            )
            if has_numbers:
                hints.append(
                    "Financial data may appear in different formats "
                    "(tables, bullet points, sentences). Check ALL formats "
                    "for the requested numbers before concluding they are absent."
                )

        elif doc_type == "research_paper":
            hints.append(
                "This content appears to be from a RESEARCH PAPER. "
                "Distinguish between proposed methods, baseline comparisons, "
                "and experimental results. Note which metrics are from "
                "the proposed approach vs. baselines."
            )
            if "compare" in q_lower or "versus" in q_lower or "vs" in q_lower:
                hints.append(
                    "For comparison questions, clearly separate the proposed "
                    "method's results from baseline/ablation results. "
                    "Mention the specific dataset and metric for each result."
                )

        elif doc_type == "technical_documentation":
            hints.append(
                "This content appears to be from TECHNICAL DOCUMENTATION. "
                "Include specific parameter names, values, and configuration "
                "details when answering. Distinguish between required and "
                "optional parameters."
            )

        elif doc_type == "structured_pdf":
            hints.append(
                "This content is from a STRUCTURED DOCUMENT. "
                "Section hierarchy and cross-references are important. "
                "When citing information, include the section context."
            )

    if has_table_ref or has_numbers:
        hints.append(
            "Extract COMPLETE ROWS from tables — include all columns "
            "for any matched row. Do not return isolated cell values."
        )

    return hints


def get_type_specific_retrieval_params(
    chunks: List[dict],
    query: str,
) -> Dict:
    """Get retrieval parameter adjustments based on document type.

    Returns dict with optional overrides for k, score_threshold, etc.
    """
    doc_types = detect_document_types(chunks)
    params: Dict = {}

    for doc_type in doc_types:
        if doc_type == "financial_report":
            params.setdefault("k", 12)
            params.setdefault("score_threshold", 0.08)
            params.setdefault("expansion_window", 2)
        elif doc_type == "research_paper":
            params.setdefault("k", 10)
            params.setdefault("min_sections", 4)
        elif doc_type == "technical_documentation":
            params.setdefault("k", 8)
            params.setdefault("expansion_window", 1)
        elif doc_type == "structured_pdf":
            params.setdefault("expansion_window", 2)

    return params


def extract_section_hierarchy(chunks: List[dict]) -> Dict[str, List[str]]:
    """Build a section hierarchy from chunks' heading_path metadata.

    Returns dict mapping section paths to their child sections.
    """
    hierarchy: Dict[str, List[str]] = {}
    for c in chunks:
        heading_path = c.get("metadata", {}).get("heading_path", "")
        if heading_path:
            parts = [p.strip() for p in heading_path.split("/")]
            for i in range(len(parts)-1):
                parent = " / ".join(parts[:i+1])
                child = " / ".join(parts[:i+2])
                hierarchy.setdefault(parent, [])
                if child not in hierarchy[parent]:
                    hierarchy[parent].append(child)
    return hierarchy


def format_with_document_context(
    query: str,
    chunks: List[dict],
    context_str: str,
) -> str:
    """Enhance the context string with document-type awareness.

    Adds type-specific hints and section context to help the LLM
    better understand and synthesize the retrieved information.
    """
    hints = get_special_handling_hints(chunks, query)
    if not hints:
        return context_str

    hint_block = "DOCUMENT CONTEXT:\n" + "\n".join(f"- {h}" for h in hints)

    hierarchy = extract_section_hierarchy(chunks)
    if hierarchy:
        hierarchy_lines = ["Section Structure:"]
        for parent, children in hierarchy.items():
            hierarchy_lines.append(f"  {parent}")
            for child in children:
                hierarchy_lines.append(f"    └─ {child.split('/')[-1].strip()}")
        hint_block += "\n\n" + "\n".join(hierarchy_lines)

    return f"{hint_block}\n\n{context_str}"
