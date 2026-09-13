"""
token_counter.py — Token estimation and budget calculation for LLM prompts and context.

Provides fast, accurate token counting (using tiktoken with conservative fallback)
and provider/model-aware token budget calculation.
"""

import math
import logging
from functools import lru_cache
from typing import List, Optional, Dict, Any

from app.core.config import settings

logger = logging.getLogger("knowledge_copilot.token_counter")

# Default request token limits by provider/model when not explicitly specified
DEFAULT_PROVIDER_BUDGETS: Dict[str, int] = {
    "groq": 6000,      # Leaves headroom for 8k TPM limit on developer tier + completion tokens
    "openai": 16000,   # Generous default for modern OpenAI models
    "ollama": 8000,    # Standard local context window
}

# Known model-specific limits for Groq
GROQ_MODEL_BUDGETS: Dict[str, int] = {
    "openai/gpt-oss-120b": 6000,
    "llama-3.3-70b-versatile": 6000,
    "llama-3.1-70b-versatile": 6000,
    "llama-3.1-8b-instant": 8000,
    "mixtral-8x7b-32768": 8000,
    "gemma2-9b-it": 6000,
}


@lru_cache(maxsize=4)
def _get_encoding(encoding_name: str = "cl100k_base"):
    """Get and cache tiktoken encoding."""
    try:
        import tiktoken
        return tiktoken.get_encoding(encoding_name)
    except Exception as e:
        logger.debug(f"Could not load tiktoken encoding '{encoding_name}': {e}")
        return None


def estimate_tokens(text: str, model: str = "") -> int:
    """
    Estimate the token count for a text string.
    Uses tiktoken cl100k_base if available; otherwise uses a conservative
    character-based heuristic (~3.7 chars per token).
    """
    if not text:
        return 0

    encoding = _get_encoding("cl100k_base")
    if encoding is not None:
        try:
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            pass

    # Conservative fallback for English/code/markdown text
    return max(1, math.ceil(len(text) / 3.7))


def estimate_messages_tokens(messages: List[Dict[str, Any]], model: str = "") -> int:
    """
    Estimate the total token count of a list of message dicts (e.g. [{"role": ..., "content": ...}]).
    Includes per-message framing overhead (~4 tokens per message).
    """
    if not messages:
        return 0

    total_tokens = 3  # every reply is primed with <|start|>assistant<|message|>
    for msg in messages:
        content = msg.get("content", "")
        role = msg.get("role", "")
        # ~4 tokens per message for formatting tags
        total_tokens += 4
        total_tokens += estimate_tokens(role, model=model)
        if isinstance(content, str):
            total_tokens += estimate_tokens(content, model=model)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    total_tokens += estimate_tokens(part["text"], model=model)
                elif isinstance(part, str):
                    total_tokens += estimate_tokens(part, model=model)

    return total_tokens


def get_model_token_budget() -> Dict[str, int]:
    """
    Calculate dynamic token budgets for the active LLM provider and model.

    Returns:
        {
            "total_request_budget": int,  # Max tokens allowed for the entire prompt
            "safety_margin": int,         # Headroom buffer reserved
            "history_budget": int,        # Max tokens allocated for conversation history
            "context_budget": int,        # Max tokens allocated for retrieved document context
            "max_completion_tokens": int, # Reserved for LLM output generation
        }
    """
    provider = (settings.llm_provider or "groq").lower()
    
    # 1. Base total request budget
    if settings.llm_token_budget is not None and settings.llm_token_budget > 0:
        total_budget = settings.llm_token_budget
    elif provider == "groq":
        model_name = (settings.groq_model or "").lower()
        total_budget = GROQ_MODEL_BUDGETS.get(model_name, DEFAULT_PROVIDER_BUDGETS.get("groq", 6000))
    else:
        total_budget = DEFAULT_PROVIDER_BUDGETS.get(provider, 6000)

    safety_margin = getattr(settings, "llm_token_safety_margin", 500)
    max_completion = getattr(settings, "llm_max_tokens", 1024)
    history_budget = getattr(settings, "history_max_tokens", 1200)

    # Effective prompt budget after safety margin
    effective_prompt_budget = max(total_budget - safety_margin, 1000)

    # Reserve budget for system prompt (~300 tokens) and user query (~150 tokens)
    system_and_query_reserve = 450

    # Remaining budget is split between history and context
    # Context gets the lion's share of available room
    context_budget = max(effective_prompt_budget - system_and_query_reserve - history_budget, 1000)

    return {
        "total_request_budget": effective_prompt_budget,
        "raw_budget": total_budget,
        "safety_margin": safety_margin,
        "history_budget": history_budget,
        "context_budget": context_budget,
        "max_completion_tokens": max_completion,
    }
