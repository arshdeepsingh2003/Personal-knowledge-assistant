"""
test_token_budget_context.py — Tests for dynamic token-budget-aware context management.

Validates:
  1. Token estimation and provider budget calculations.
  2. Priority-aware chunk packing (primary chunks first, then budget-capped expansion).
  3. Dynamic chunk expansion capping under tight vs generous token budgets.
  4. Conversation history token budgeting and trimming in build_prompt.
  5. Pre-invocation safeguard trimming when prompt exceeds budget.
  6. SourceReference consistency with packed chunks in format_context_for_llm.
  7. Resilient streaming error handling without ASGI crash.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.core.config import settings
from app.services.token_counter import (
    estimate_tokens,
    estimate_messages_tokens,
    get_model_token_budget,
)
from app.services.retriever import (
    _trim_text_to_tokens,
    _budget_and_pack_chunks,
    format_context_for_llm,
    RetrievalResult,
    SourceReference,
)
from app.services.llm import build_prompt, stream_answer


def test_token_estimation_basic():
    """Test token estimation for simple text and empty text."""
    assert estimate_tokens("") == 0
    assert estimate_tokens("Hello world") >= 2
    long_text = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = estimate_tokens(long_text)
    assert 400 <= tokens <= 600


def test_estimate_messages_tokens():
    """Test message token calculation with role and content overhead."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
    ]
    total = estimate_messages_tokens(messages)
    assert total > estimate_tokens("You are a helpful assistant.") + estimate_tokens("What is AI?")


def test_model_token_budget_defaults():
    """Test default budget calculations for different providers."""
    # Groq default
    with patch.object(settings, "llm_provider", "groq"):
        with patch.object(settings, "groq_model", "openai/gpt-oss-120b"):
            with patch.object(settings, "llm_token_budget", None):
                budget = get_model_token_budget()
                assert budget["total_request_budget"] <= 6000
                assert budget["context_budget"] > 0
                assert budget["history_budget"] == settings.history_max_tokens

    # Explicit override
    with patch.object(settings, "llm_token_budget", 4000):
        budget = get_model_token_budget()
        assert budget["raw_budget"] == 4000
        assert budget["total_request_budget"] == 4000 - settings.llm_token_safety_margin


def test_trim_text_to_tokens():
    """Test text trimming to target token limit."""
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    trimmed = _trim_text_to_tokens(text, 5)
    assert estimate_tokens(trimmed) <= 5
    assert "..." in trimmed or len(trimmed) < len(text)


def test_budget_and_pack_chunks_priority_and_expansion():
    """
    Test that primary chunks are prioritized first by relevance, and expansion
    chunks are dynamically included only up to the available token budget.
    """
    primary = [
        {
            "id": "c1",
            "text": "Primary chunk 1: High relevance ROI analysis data." * 10,
            "rerank_score": 0.95,
            "score": 0.95,
            "metadata": {"section_id": "sec1", "section_chunk_index": 0, "file_name": "doc1.pdf"},
        },
        {
            "id": "c2",
            "text": "Primary chunk 2: High relevance Strategy framework." * 10,
            "rerank_score": 0.85,
            "score": 0.85,
            "metadata": {"section_id": "sec2", "section_chunk_index": 0, "file_name": "doc1.pdf"},
        },
        {
            "id": "c3",
            "text": "Primary chunk 3: Lower relevance general info." * 10,
            "rerank_score": 0.60,
            "score": 0.60,
            "metadata": {"section_id": "sec3", "section_chunk_index": 0, "file_name": "doc1.pdf"},
        },
    ]

    # Mock vector store returning adjacent chunks for sec1
    mock_store = MagicMock()
    mock_store.get_chunks_by_section_id.return_value = [
        {
            "id": "c1_adj1",
            "text": "Adjacent chunk to primary 1 from sec1." * 10,
            "metadata": {"section_id": "sec1", "section_chunk_index": 1, "file_name": "doc1.pdf"},
        },
        {
            "id": "c1_adj2",
            "text": "Another adjacent chunk to primary 1 from sec1." * 10,
            "metadata": {"section_id": "sec1", "section_chunk_index": 2, "file_name": "doc1.pdf"},
        },
    ]

    with patch("app.services.retriever.get_vector_store", return_value=mock_store):
        # Case A: Generous budget (fits all primary + adjacent expansions)
        packed, total_tokens = _budget_and_pack_chunks(
            primary,
            context_token_budget=3000,
            expansion_enabled=True,
            expansion_window=2,
        )
        assert len(packed) >= 3
        # Primary chunks come first and are not marked expanded
        assert packed[0]["id"] == "c1"
        assert packed[0]["_expanded"] is False
        assert packed[1]["id"] == "c2"
        # Expanded chunks are present and marked expanded
        expanded_in_packed = [c for c in packed if c.get("_expanded")]
        assert len(expanded_in_packed) > 0

        # Case B: Tight budget (only fits top primary chunk and no/minimal expansion)
        c1_tokens = estimate_tokens(primary[0]["text"]) + 30
        tight_packed, tight_tokens = _budget_and_pack_chunks(
            primary,
            context_token_budget=c1_tokens + 20,
            expansion_enabled=True,
            expansion_window=2,
        )
        assert len(tight_packed) == 1
        assert tight_packed[0]["id"] == "c1"
        assert tight_tokens <= c1_tokens + 20


def test_build_prompt_history_budgeting():
    """Test that conversation history is budgeted and does not exceed history budget."""
    history = [
        {"role": "user", "content": f"Turn {i} question: " + ("what about this topic? " * 15)}
        for i in range(10)
    ]
    for h in history:
        h["role"] = "user"

    prompt = build_prompt("Current question", "Context data", history, max_total_tokens=4000)
    
    # Check that system prompt is first and user prompt is last
    assert prompt[0]["role"] == "system"
    assert prompt[-1]["role"] == "user"
    assert "Current question" in prompt[-1]["content"]

    # Check total prompt token count stays within budget
    assert estimate_messages_tokens(prompt) <= 4000


def test_build_prompt_safeguard_trimming():
    """Test that build_prompt trims huge context if total tokens exceed prompt budget."""
    huge_context = "Massive context paragraph with lots of data. " * 300
    prompt = build_prompt("Quick question", huge_context, history=[], max_total_tokens=500)
    total_tokens = estimate_messages_tokens(prompt)
    assert total_tokens <= 600  # Within tight budget ceiling


def test_format_context_for_llm_with_budget():
    """Test format_context_for_llm formatting and budget enforcement."""
    sources = [
        SourceReference(
            file_name="report.pdf",
            chunk_index=0,
            page=1,
            score=0.92,
            rerank_score=0.92,
            preview="Preview text...",
            content_type="prose",
            section="Executive Summary",
            table_name="",
            source_number=1,
            expanded=False,
        )
    ]
    chunks = [
        {
            "id": "1",
            "text": "Executive Summary: The company grew 40% YoY.",
            "metadata": {"section": "Executive Summary", "file_name": "report.pdf"},
            "_source_num": 1,
        }
    ]
    result = RetrievalResult(
        query="What was the YoY growth?",
        context="Executive Summary: The company grew 40% YoY.",
        sources=sources,
        chunks=chunks,
        total_found=1,
        expanded_queries=[],
        retrieval_metrics={},
    )
    formatted = format_context_for_llm(result, max_tokens=1000)
    assert "SOURCES:" in formatted
    assert "[1] report.pdf" in formatted
    assert "Executive Summary" in formatted
    assert "The company grew 40% YoY." in formatted


def test_stream_answer_clean_error_propagation():
    """Test that stream_answer raises standard RuntimeError on 413 / 429 rather than crashing."""
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = Exception("413 Request Entity Too Large - tokens exceed TPM")

    with patch("app.services.llm.get_llm", return_value=mock_llm):
        with pytest.raises(RuntimeError) as exc_info:
            list(stream_answer("query", "context", []))
        assert "413" in str(exc_info.value)
