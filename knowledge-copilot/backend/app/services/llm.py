from functools import lru_cache
from typing import Generator, List

from app.core.config import settings


# ── Model factory ─────────────────────────────────────────────────────────────
# @lru_cache means the model client is created once and reused across requests.
# Changing LLM_PROVIDER in .env requires a server restart to take effect.

@lru_cache(maxsize=1)
def get_llm():
    """
    Return a LangChain chat model based on LLM_PROVIDER in .env.

    Supported providers:
      groq   — Llama 3.1 70B via Groq LPU (fastest, recommended)
      openai — GPT-3.5-turbo / GPT-4o (highest accuracy)
      ollama — Local models (no API key, slowest)
    """
    if settings.llm_provider == "groq":
        if not settings.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set in .env. "
                "Get a free key at console.groq.com then add:\n"
                "  GROQ_API_KEY=gsk_your_key_here"
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
            raise ValueError(
                "OPENAI_API_KEY is not set in .env. "
                "Either add it or switch LLM_PROVIDER=groq"
            )
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
        "Valid options: groq | openai | ollama"
    )


# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for a personal knowledge base.
You answer questions strictly based on the provided context documents.

Rules you must follow:
- Only use information from the CONTEXT section below
- If the context does not contain the answer, say clearly: \
"I don't have enough information in the uploaded documents to answer this."
- Never make up facts or use knowledge outside the provided context
- Always be concise and direct
- When relevant, mention which source the information came from"""


def build_prompt(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> List[dict]:
    """
    Assemble the full message list for the chat model.

    Structure:
      [system]  ← instructions + retrieved context
      [history] ← last N conversation turns (optional)
      [user]    ← current question

    Returns a list of {role, content} dicts — the format every
    LangChain chat model accepts regardless of provider.
    """
    history = history or []

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context if context else 'No context available.'}"
    )

    messages = [{"role": "system", "content": system_content}]

    # Inject last 6 turns of history (3 user + 3 assistant)
    # Groq and Llama 3.1 handle multi-turn context well
    MAX_HISTORY_TURNS = 6
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
    return messages


def _to_langchain_messages(messages: List[dict]):
    """Convert {role, content} dicts to LangChain message objects."""
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    lc = []
    for m in messages:
        if m["role"] == "system":
            lc.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc.append(AIMessage(content=m["content"]))
    return lc


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> str:
    """
    Generate a complete answer (blocking / non-streaming).

    Used by: POST /chat/message, POST /api/v1/ask (stream=false)

    Groq Llama 3.1 70B typically responds in 0.8-2.0 seconds.
    """
    llm      = get_llm()
    messages = build_prompt(query, context, history)
    lc_msgs  = _to_langchain_messages(messages)
    response = llm.invoke(lc_msgs)
    return response.content


def stream_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> Generator[str, None, None]:
    """
    Stream the answer token by token.

    Used by: POST /chat/stream, POST /api/v1/ask (stream=true)

    Groq streams at 300-800 tokens/sec — the first token arrives
    in under 100ms, giving a very fast "typing" effect in the UI.
    """
    llm      = get_llm()
    messages = build_prompt(query, context, history)
    lc_msgs  = _to_langchain_messages(messages)

    for chunk in llm.stream(lc_msgs):
        if chunk.content:
            yield chunk.content


# ── Available Groq models (for reference) ─────────────────────────────────────
#
# Model name                      | Context  | Speed      | Best for
# --------------------------------|----------|------------|------------------
# llama-3.1-70b-versatile         | 128K     | ~300 t/s   | Best overall RAG ✓
# llama-3.1-8b-instant            | 128K     | ~750 t/s   | Speed-first RAG
# llama-3.2-90b-text-preview      | 128K     | ~200 t/s   | Highest accuracy
# llama-3.2-11b-text-preview      | 128K     | ~500 t/s   | Balanced
# mixtral-8x7b-32768              | 32K      | ~400 t/s   | Long context
# gemma2-9b-it                    | 8K       | ~500 t/s   | Lightweight
#
# Set GROQ_MODEL in .env to switch. No code changes needed.