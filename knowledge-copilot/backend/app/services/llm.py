"""
llm.py — LLM with table-optimised system prompt

Key change from Phase 7 / Groq migration:
  The SYSTEM_PROMPT is rewritten to explicitly instruct the model
  to extract data from tables and never claim "not enough information"
  when tabular or numeric data is present in the context.

  Root cause of the original failure: the old prompt said
  "if the context does not contain the answer, say so."
  Llama 3.1 and GPT-4 interpret pipe-delimited table rows or
  NL sentences like "For Retail, the ROI is 312%" as ambiguous
  and sometimes respond with "I don't have enough information"
  even when the answer is clearly present.

  The new prompt forces the model to:
    1. Actively look for numeric data and table entries
    2. Quote specific numbers when available
    3. Only give the "no information" response if the data is
       genuinely absent — not when it's just in tabular form
"""

import json
import logging
from functools import lru_cache
from typing import Generator, List, Optional

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.llm")


def _inject_memory_context(
    history: List[dict],
    current_query: str,
) -> dict:
    """Inject conversation memory context using the memory manager."""
    from app.services.memory_manager import build_memory_context, needs_compression
    if not history:
        return {"memory_block": "", "history": history}
    mem = build_memory_context(history, current_query)
    parts = []
    if mem.get("summary"):
        parts.append(f"[Conversation Context: {mem['summary']}]")
    if mem.get("entities"):
        ents = list(mem["entities"].keys())[:8]
        parts.append(f"[Previously Mentioned Entities: {', '.join(ents)}]")
    return {
        "memory_block": "\n".join(parts) if parts else "",
        "history": mem["history"],
        "entities": mem.get("entities", {}),
    }


@lru_cache(maxsize=1)
def get_llm():
    """Return a LangChain chat model. Cached after first load."""

    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env.\n"
                "Get a free key at console.groq.com"
            )
        from langchain_groq import ChatGroq
        print(f"✓ LLM: Groq — {settings.groq_model}")
        return ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")
        from langchain_openai import ChatOpenAI
        print(f"✓ LLM: OpenAI — {settings.llm_model}")
        return ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

    if settings.llm_provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        print(f"✓ LLM: Ollama — {settings.ollama_model}")
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: '{settings.llm_provider}'. "
        "Valid: groq | openai | ollama"
    )


# ── System prompt — enhanced with synthesis, confidence, and citation grounding ──

SYSTEM_PROMPT = """You are a precise research assistant for a personal knowledge base.
Your job is to answer questions using ONLY the provided CONTEXT section.

CRITICAL RULES — read these carefully:

1. GROUNDING IN CONTEXT:
   - Your answer MUST be based EXCLUSIVELY on the CONTEXT section below.
   - If the CONTEXT contains information relevant to the question, you MUST use it.
   - Do NOT add information from your training data, even if you are confident it is correct.
   - If the CONTEXT does not contain the answer at all, say "The provided context does not contain information about [topic]."
   - However, if the answer CAN be found by combining information from MULTIPLE chunks, DO SO.

2. TABLES AND NUMERIC DATA:
   - The context may contain TABLE data with rows like:
     "For Retail & E-commerce, the Year 1 ROI is 312% and the Payback Period is 3.8 months."
   - You MUST read ALL such sentences carefully and extract numeric values from them.
   - When a question asks about ROI, market share, performance, cost, speed, or any
     quantitative metric — scan every context chunk for matching numbers.
   - NEVER say "I don't have enough information" if numeric data is present in the
     context that is relevant to the question — even if it's in table form.
   - If you find partial numeric data, use what is available and note what is missing.

3. ACCURACY:
   - Quote specific numbers and percentages EXACTLY as they appear in the context.
   - If multiple rows match (e.g., ROI for multiple industries), list ALL of them.
   - Do not round or approximate numbers unless the source does.
   - Include units, time periods, and qualifiers exactly as written.

4. MULTI-SECTION SYNTHESIS:
   - The context may span MULTIPLE SECTIONS of a document.
   - Before answering, check if information from DIFFERENT chunks can be combined to give a complete answer.
   - If one chunk has a concept definition and another has specific data about it, combine both.
   - If information is spread across chunks [1], [2], [3] etc., reference ALL relevant chunks.

5. WHEN TO SAY "NOT ENOUGH INFORMATION":
   - ONLY use this response when the specific data point is genuinely absent
     from the context — not when it's present in a different format.
   - If you see partial information, give what you have and note what's missing.
   - Check ALL chunks before concluding information is missing.

6. CITATION FORMAT:
   - ALWAYS cite the source number [1], [2] etc. after each factual statement.
   - Place citations IMMEDIATELY after the claim they support, not at the end of a paragraph.
   - When combining multiple sources, cite each one separately like [1][2].
   - For numeric claims, the citation MUST be adjacent to the number.
   - Example correct: "The Retail ROI is 312% [1], while Healthcare ROI is 189% [2]."
   - Example WRONG: "The Retail ROI is 312% and Healthcare ROI is 189% [1][2]."
   - For tabular questions (comparisons, rankings, benchmarks), use bullet points or a small table.

7. HALLUCINATION PREVENTION:
   - Never fabricate numbers, names, or relationships.
   - If a number or statistic appears in the context, cite it with the specific source number.
   - If you are unsure about a relationship between concepts, say so explicitly.
   - Do not invent acronym expansions, definitions, or formulas.
   - If you are extrapolating or inferring, state that explicitly (e.g., "Based on [1] and [3], it appears that...").

8. CROSS-DOCUMENT & CROSS-SECTION SYNTHESIS:
   - When the question asks to COMPARE, CONTRAST, or discuss DIFFERENCES:
     * Aggregate information from ALL retrieved chunks, not just the first one.
     * Structure your answer to highlight similarities AND differences explicitly.
     * Look for data points on the same topic across different chunks and compare them.
   - When MULTIPLE chunks cover the same topic with different data:
     * Present ALL viewpoints with their respective source citations.
     * Note discrepancies explicitly (e.g., "Source [1] states X, while source [3] states Y").
   - For ANY question, before concluding "the context does not contain this information":
     * Check if the answer can be synthesized from MULTIPLE chunks.
     * Look for partial information that, when combined, provides a complete answer.

9. CONFIDENCE SIGNALING:
   - If you are highly confident about a claim (data directly stated in context), state it confidently.
   - If you are moderately confident (information is partially present or requires inference), use hedging language like "suggests", "indicates", "appears to be".
   - If you are uncertain (information is absent or ambiguous), say so clearly.
   - Never present speculation as fact.

10. SUMMARIZATION RULES (when asked to summarize or give an overview):
    - When summarizing, your goal is BALANCED COVERAGE of ALL major topics present in the context, not just the first or most detailed ones.
    - IDENTIFY the most globally significant concepts: those mentioned across MULTIPLE sections or chunks are more important than those confined to one section.
    - AVOID repetitive patterns: if multiple chunks describe similar concepts, MERGE them into a single point rather than repeating.
    - For SHORT/concise summaries, prioritize breadth over depth: mention each major area once rather than going deep on one area.
    - When the context covers DISTINCT topics, group your summary by topic area, NOT by chunk order. Look for complementary information across chunks.
    - For bullet-point summaries, ensure each bullet covers a DIFFERENT aspect. If two bullets overlap, merge them.
    - SIGNAL when you are combining information from different sections: e.g., "Across all sections, the key themes are..." or "The document covers three major areas..."
    - DO NOT artificially inflate the number of points. If the document truly covers fewer topics than requested, use fewer points."""


def build_prompt(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> List[dict]:
    """Assemble the full message list for the chat model."""
    history = history or []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Uses memory_manager for smarter history selection
    mem = _inject_memory_context(history, query)
    for turn in mem["history"]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    memory_block = mem.get("memory_block", "")
    user_content_parts = []
    if memory_block:
        user_content_parts.append(memory_block)
    if context:
        user_content_parts.append(f"CONTEXT:\n{context}")
    user_content_parts.append(f"QUESTION:\n{query}")
    user_content = "\n\n".join(user_content_parts)

    messages.append({"role": "user", "content": user_content})
    return messages


def _to_lc_messages(messages: List[dict]):
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            out.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            out.append(AIMessage(content=m["content"]))
    return out


def _run_confidence_check(
    answer: str,
    chunks: List[dict],
    sources: list,
) -> dict:
    """Run confidence estimation and citation grounding checks."""
    result = {"confidence": {}, "citation_check": {}}
    if not settings.confidence_enabled:
        return result
    try:
        from app.services.confidence import estimate_confidence, check_citation_grounding
        conf = estimate_confidence(answer, chunks)
        result["confidence"] = conf
        cit = check_citation_grounding(answer, sources)
        result["citation_check"] = cit
        if conf.get("warnings"):
            for w in conf["warnings"]:
                logger.warning(f"Confidence: {w}")
        if cit.get("warnings"):
            for w in cit["warnings"]:
                logger.warning(f"Citation: {w}")
    except Exception as e:
        logger.warning(f"Confidence check failed: {e}")
    return result


def generate_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
    chunks:  List[dict] = None,
    sources: list = None,
) -> str:
    llm  = get_llm()
    msgs = build_prompt(query, context, history)
    resp = llm.invoke(_to_lc_messages(msgs))
    answer = resp.content

    if chunks or sources:
        checks = _run_confidence_check(answer, chunks or [], sources or [])
        if checks.get("confidence", {}).get("warnings"):
            logger.warning(
                f"Answer confidence: {checks['confidence'].get('overall_confidence', 'N/A')}"
            )

    return answer


def generate_answer_with_meta(
    query:   str,
    context: str,
    history: List[dict] = None,
    chunks:  List[dict] = None,
    sources: list = None,
) -> dict:
    """Generate answer and return it with confidence/citation metadata."""
    answer = generate_answer(query, context, history, chunks, sources)
    meta = {"answer": answer}
    if chunks or sources:
        meta["confidence"] = _run_confidence_check(answer, chunks or [], sources or [])
    return meta


def stream_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
    chunks:  List[dict] = None,
    sources: list = None,
) -> Generator[str, None, None]:
    llm  = get_llm()
    msgs = build_prompt(query, context, history)
    for chunk in llm.stream(_to_lc_messages(msgs)):
        if chunk.content:
            yield chunk.content