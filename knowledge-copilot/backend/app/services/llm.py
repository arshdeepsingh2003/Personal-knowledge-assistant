from functools import lru_cache
from typing import Generator, List

from app.core.config import settings


# ── Model factory ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm():
    """
    Return a LangChain chat model.
    Default: Ollama (local, free)
    Optional: OpenAI (if explicitly configured)
    """

    # 👉 Use OpenAI ONLY if explicitly set
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model          = settings.llm_model,
            openai_api_key = settings.openai_api_key,
            temperature    = settings.llm_temperature,
            max_tokens     = settings.llm_max_tokens,
        )

    # 👉 Default: Ollama
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(
        model       = settings.ollama_model,
        base_url    = settings.ollama_base_url,
        temperature = settings.llm_temperature,
    )
    


# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant for a personal knowledge base.
You answer questions strictly based on the provided context documents.

Rules you must follow:
- Only use information from the CONTEXT section below
- If the context does not contain the answer, say clearly: "I don't have enough information in the uploaded documents to answer this."
- Never make up facts or use knowledge outside the provided context
- Always be concise and direct
- When relevant, mention which source the information came from"""


def build_prompt(
    query:       str,
    context:     str,
    history:     List[dict] = None,
) -> List[dict]:
    """
    Assemble the full message list for the chat model.

    Structure:
      [system]              ← instructions + context
      [assistant, user...]  ← conversation history (optional)
      [user]                ← current question

    Returns a list of {role, content} dicts — the format every
    LangChain chat model accepts.
    """
    history = history or []

    system_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context if context else 'No context available.'}"
    )

    messages = [{"role": "system", "content": system_content}]

    # Inject last N turns of history (capped to avoid token overflow)
    MAX_HISTORY_TURNS = 6   # 3 user + 3 assistant messages
    for turn in history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": query})
    return messages


# ── Answer generation ─────────────────────────────────────────────────────────

def generate_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> str:
    """
    Generate a complete answer (blocking).
    Returns the full answer string.
    Used when streaming is not needed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    llm      = get_llm()
    messages = build_prompt(query, context, history)

    # Convert dicts → LangChain message objects
    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    response = llm.invoke(lc_messages)
    return response.content


def stream_answer(
    query:   str,
    context: str,
    history: List[dict] = None,
) -> Generator[str, None, None]:
    """
    Stream the answer token by token.
    Yields string chunks as they arrive from the LLM.
    Used with SSE (Server-Sent Events) for real-time UI.
    """
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

    llm      = get_llm()
    messages = build_prompt(query, context, history)

    lc_messages = []
    for m in messages:
        if m["role"] == "system":
            lc_messages.append(SystemMessage(content=m["content"]))
        elif m["role"] == "user":
            lc_messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            lc_messages.append(AIMessage(content=m["content"]))

    for chunk in llm.stream(lc_messages):
        if chunk.content:
            yield chunk.content