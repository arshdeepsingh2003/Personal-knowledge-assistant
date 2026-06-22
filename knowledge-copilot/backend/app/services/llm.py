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


SYSTEM_PROMPT = """You are a precise research assistant for a personal knowledge base.
Answer the user's question using ONLY the RETRIEVED CONTEXT provided below.

RULES:
1. ANSWER ONLY FROM CONTEXT — If the answer is not found in the RETRIEVED CONTEXT, say: "I could not find this information in the provided context." Do not use your own knowledge.

2. BE CONCISE — Answer directly and briefly. Do not show reasoning steps, do not list what you checked, do not generate intermediate analysis.

3. CITE SOURCES — Use source tags [1], [2] adjacent to each factual statement. Place the tag right after the claim it supports.

4. TABLES — When the answer is in a table, extract the relevant row with all its columns. Return the complete row.

5. MULTIPLE CHUNKS — If relevant info is in multiple chunks, combine it. If chunks are unrelated to the question, ignore them.

6. NO chain-of-thought — Do not say "I checked", "I found", "Based on my analysis", "Let me verify", or any reasoning steps. Just give the answer.

7. NO fabricated answers — Never invent numbers, names, or facts. If the context partially answers, give what is available and state what is missing."""


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
    # ── Diagnostic logging: full prompt ────────────────────────────────────
    prompt_text = "\n".join(m.get("content", "") for m in msgs)
    logger.info("=== FULL PROMPT SENT TO LLM ===")
    for i, m in enumerate(msgs):
        logger.info(f"  [{i}] role={m['role']} | content={m['content'][:500]}")
    logger.info(f"  [prompt total length] {len(prompt_text)} chars")
    # ────────────────────────────────────────────────────────────────────────

    resp = llm.invoke(_to_lc_messages(msgs))
    answer = resp.content

    # ── Diagnostic logging: raw LLM response ───────────────────────────────
    logger.info(f"=== RAW LLM RESPONSE === {answer[:1000]}")
    logger.info(f"  [response length] {len(answer)} chars")
    # ────────────────────────────────────────────────────────────────────────

    if chunks or sources:
        checks = _run_confidence_check(answer, chunks or [], sources or [])
        if checks.get("confidence", {}).get("warnings"):
            logger.warning(
                f"Answer confidence: {checks['confidence'].get('overall_confidence', 'N/A')}"
            )

    # Completeness check: expand answer if key facts are missing
    # SAFETY: Never expand an answer that says "not found" — doing so
    # forces the LLM to hallucinate from prior knowledge.
    if settings.completeness_check_enabled and chunks:
        not_found_phrases = [
            "could not find this information",
            "cannot find this information",
            "not found in the provided context",
            "do not have enough information",
            "no information",
            "not mentioned",
            "does not contain the answer",
        ]
        answer_lower = answer.lower()
        is_not_found = any(phrase in answer_lower for phrase in not_found_phrases)

        if is_not_found:
            logger.info("Completeness check skipped: answer indicates information not found in context")
        else:
            try:
                from app.services.completeness import (
                    check_answer_completeness,
                    generate_expansion_prompt,
                )
                completeness = check_answer_completeness(answer, chunks, query)
                if not completeness.get("is_complete", True) and completeness.get("expansion_suggestions"):
                    expansion_prompt = generate_expansion_prompt(completeness, context)
                    if expansion_prompt:
                        expand_msgs = list(msgs)
                        expand_msgs.append({
                            "role": "assistant",
                            "content": answer,
                        })
                        expand_msgs.append({
                            "role": "user",
                            "content": expansion_prompt,
                        })
                        expanded_resp = llm.invoke(_to_lc_messages(expand_msgs))
                        expanded = expanded_resp.content
                        if expanded and len(expanded) > len(answer) * 0.5:
                            logger.info(
                                f"Answer expanded via completeness check "
                                f"(coverage: {completeness.get('coverage_ratio', 0):.0%})"
                            )
                            answer = expanded
            except Exception as e:
                logger.warning(f"Completeness check/expansion failed: {e}")

    return answer


def generate_answer_with_meta(
    query:   str,
    context: str,
    history: List[dict] = None,
    chunks:  List[dict] = None,
    sources: list = None,
) -> dict:
    """Generate answer and return it with confidence/citation/metadata."""
    answer = generate_answer(query, context, history, chunks, sources)
    meta = {"answer": answer}
    if chunks or sources:
        meta["confidence"] = _run_confidence_check(answer, chunks or [], sources or [])

    # Full evaluation metrics
    if chunks and sources:
        try:
            from app.services.retriever import RetrievalResult
            rr = RetrievalResult(
                query=query,
                context=context,
                sources=sources,
                chunks=chunks,
                total_found=len(chunks),
            )
            from app.services.metrics import compute_all_evaluation_metrics
            meta["evaluation"] = compute_all_evaluation_metrics(answer, query, rr)
        except Exception as e:
            logger.warning(f"Evaluation metrics failed: {e}")

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
    # ── Diagnostic logging: full prompt (streaming) ────────────────────────
    logger.info("=== FULL PROMPT SENT TO LLM (streaming) ===")
    for i, m in enumerate(msgs):
        logger.info(f"  [{i}] role={m['role']} | content={m['content'][:500]}")
    prompt_text = "\n".join(m.get("content", "") for m in msgs)
    logger.info(f"  [prompt total length] {len(prompt_text)} chars")
    # ────────────────────────────────────────────────────────────────────────

    full_response = []
    for chunk in llm.stream(_to_lc_messages(msgs)):
        if chunk.content:
            full_response.append(chunk.content)
            yield chunk.content

    # ── Diagnostic logging: raw LLM response ───────────────────────────────
    raw = "".join(full_response)
    logger.info(f"=== RAW LLM RESPONSE (streaming) === {raw[:1000]}")
    logger.info(f"  [response length] {len(raw)} chars")
    # ────────────────────────────────────────────────────────────────────────