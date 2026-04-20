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

from functools import lru_cache
from typing import Generator, List

from app.core.config import settings


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


# ── System prompt — table-aware version ──────────────────────────────────────

SYSTEM_PROMPT = """You are a precise research assistant for a personal knowledge base.
Your job is to answer questions using ONLY the provided CONTEXT section.

CRITICAL RULES — read these carefully:

1. TABLES AND NUMERIC DATA:
   - The context may contain TABLE data with rows like:
     "For Retail & E-commerce, the Year 1 ROI is 312% and the Payback Period is 3.8 months."
   - You MUST read ALL such sentences carefully and extract numeric values from them.
   - When a question asks about ROI, market share, performance, cost, speed, or any
     quantitative metric — scan every context chunk for matching numbers.
   - NEVER say "I don't have enough information" if numeric data is present in the
     context that is relevant to the question — even if it's in table form.

2. ACCURACY:
   - Quote specific numbers and percentages exactly as they appear in the context.
   - If multiple rows match (e.g., ROI for multiple industries), list ALL of them.
   - Do not round or approximate numbers unless the source does.

3. WHEN TO SAY "NOT ENOUGH INFORMATION":
   - ONLY use this response when the specific data point is genuinely absent
     from the context — not when it's present in a different format.
   - If you see partial information, give what you have and note what's missing.

4. FORMAT:
   - For tabular questions (comparisons, rankings, benchmarks), use a structured
     format in your answer: bullet points or a small table.
   - For prose questions, answer in clear paragraphs.
   - Always cite the source number [1], [2] etc. from the context.

5. SCOPE:
   - Only use information from the CONTEXT section.
   - Do not add information from your training data, even if you are confident it is correct."""


def build_prompt(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> List[dict]:
    """Assemble the full message list for the chat model."""
    history = history or []

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context if context else 'No context available.'}"
    )

    messages = [{"role": "system", "content": system_content}]

    MAX_HISTORY_TURNS = 6
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
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


def generate_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> str:
    llm  = get_llm()
    msgs = build_prompt(query, context, history)
    resp = llm.invoke(_to_lc_messages(msgs))
    return resp.content


def stream_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> Generator[str, None, None]:
    llm  = get_llm()
    msgs = build_prompt(query, context, history)
    for chunk in llm.stream(_to_lc_messages(msgs)):
        if chunk.content:
            yield chunk.content