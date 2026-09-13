"""
Direct test runner for token-budget-aware context management.
"""
import sys
import os

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_1_token_estimation():
    print("Testing token estimation...")
    assert estimate_tokens("") == 0
    assert estimate_tokens("Hello world") >= 2
    long_text = "The quick brown fox jumps over the lazy dog. " * 50
    tokens = estimate_tokens(long_text)
    assert 400 <= tokens <= 600
    print("  [PASS] test_1_token_estimation")


def test_2_estimate_messages_tokens():
    print("Testing message tokens...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is AI?"},
    ]
    total = estimate_messages_tokens(messages)
    assert total > estimate_tokens("You are a helpful assistant.") + estimate_tokens("What is AI?")
    print("  [PASS] test_2_estimate_messages_tokens")


def test_3_model_token_budget_defaults():
    print("Testing model token budgets...")
    with patch.object(settings, "llm_provider", "groq"):
        with patch.object(settings, "groq_model", "openai/gpt-oss-120b"):
            with patch.object(settings, "llm_token_budget", None):
                budget = get_model_token_budget()
                assert budget["total_request_budget"] <= 6000
                assert budget["context_budget"] > 0
                assert budget["history_budget"] == settings.history_max_tokens

    with patch.object(settings, "llm_token_budget", 4000):
        budget = get_model_token_budget()
        assert budget["raw_budget"] == 4000
        assert budget["total_request_budget"] == 4000 - settings.llm_token_safety_margin
    print("  [PASS] test_3_model_token_budget_defaults")


def test_4_trim_text():
    print("Testing text trimming...")
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    trimmed = _trim_text_to_tokens(text, 5)
    assert estimate_tokens(trimmed) <= 5
    print("  [PASS] test_4_trim_text")


def test_5_budget_and_pack_chunks():
    print("Testing priority-based chunk packing and dynamic expansion...")
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
        # Generous budget: packs primary + expansion
        packed, total_tokens = _budget_and_pack_chunks(
            primary,
            context_token_budget=3000,
            expansion_enabled=True,
            expansion_window=2,
        )
        assert len(packed) >= 3
        assert packed[0]["id"] == "c1"
        assert packed[0]["_expanded"] is False
        assert packed[1]["id"] == "c2"
        expanded_in_packed = [c for c in packed if c.get("_expanded")]
        assert len(expanded_in_packed) > 0

        # Tight budget: fits only top chunk
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
    print("  [PASS] test_5_budget_and_pack_chunks")


def test_6_build_prompt_budgeting():
    print("Testing prompt building and history budgeting...")
    history = [
        {"role": "user", "content": f"Turn {i} question: " + ("what about this topic? " * 15)}
        for i in range(10)
    ]
    for h in history:
        h["role"] = "user"

    prompt = build_prompt("Current question", "Context data", history, max_total_tokens=4000)
    assert prompt[0]["role"] == "system"
    assert prompt[-1]["role"] == "user"
    assert "Current question" in prompt[-1]["content"]
    assert estimate_messages_tokens(prompt) <= 4000
    print("  [PASS] test_6_build_prompt_budgeting")


def test_7_safeguard_trimming():
    print("Testing pre-invocation safeguard trimming...")
    huge_context = "Massive context paragraph with lots of data. " * 300
    prompt = build_prompt("Quick question", huge_context, history=[], max_total_tokens=500)
    total_tokens = estimate_messages_tokens(prompt)
    assert total_tokens <= 600
    print("  [PASS] test_7_safeguard_trimming")


def test_8_format_context_llm():
    print("Testing format_context_for_llm...")
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
    print("  [PASS] test_8_format_context_llm")


def test_9_streaming_error_handling():
    print("Testing streaming error handling...")
    mock_llm = MagicMock()
    mock_llm.stream.side_effect = Exception("413 Request Entity Too Large - tokens exceed TPM")

    with patch("app.services.llm.get_llm", return_value=mock_llm):
        try:
            list(stream_answer("query", "context", []))
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "413" in str(e)
    print("  [PASS] test_9_streaming_error_handling")


if __name__ == "__main__":
    print("=== RUNNING TOKEN BUDGET CONTEXT MANAGEMENT TESTS ===")
    test_1_token_estimation()
    test_2_estimate_messages_tokens()
    test_3_model_token_budget_defaults()
    test_4_trim_text()
    test_5_budget_and_pack_chunks()
    test_6_build_prompt_budgeting()
    test_7_safeguard_trimming()
    test_8_format_context_llm()
    test_9_streaming_error_handling()
    print("=== ALL TESTS PASSED SUCCESSFULLY! ===")
