"""
embedder.py — Upgraded embedding model for better table retrieval

Key changes from Phase 4:
  1. Default model changed from all-MiniLM-L6-v2 (384d) to
     BAAI/bge-large-en-v1.5 (1024d).

     Why bge-large is better for tables:
       - Trained on BEIR benchmark which includes structured data tasks
       - 1024 dimensions vs 384 — more capacity to encode numeric/tabular content
       - MTEB leaderboard score: 64.2 vs 56.3 for all-MiniLM

  2. BGE models require a query prefix for asymmetric retrieval:
       Query embeddings:   prepend "Represent this sentence for searching relevant passages: "
       Document embeddings: no prefix
     This asymmetry improves retrieval accuracy by ~15% on average.
     all-MiniLM did NOT use this — adding it to the wrong model breaks things.

  3. OpenAI option upgraded from text-embedding-3-small to text-embedding-3-large
     (3072d, better on structured content).
"""
from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from app.core.config import settings


# ── Model factory ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Return a LangChain-compatible embeddings object.
    Cached after first load — model stays in memory.
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Add it to .env or switch EMBEDDING_PROVIDER=local"
            )
        from langchain_openai import OpenAIEmbeddings
        print(f"✓ Embeddings: OpenAI — {settings.embedding_model_openai}")
        return OpenAIEmbeddings(
            model=settings.embedding_model_openai,
            openai_api_key=settings.openai_api_key,
        )

    # Local: use HuggingFaceEmbeddings which wraps sentence-transformers
    # BGE models need encode_kwargs to normalise vectors (required for cosine sim)
    from langchain_community.embeddings import HuggingFaceEmbeddings

    model_name = settings.embedding_model_local
    print(f"✓ Embeddings: Local — {model_name}")

    # BGE models use a query instruction prefix for better asymmetric retrieval
    # This MUST only be applied to query embeddings, not document embeddings
    is_bge = "bge" in model_name.lower()

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity
        # query_instruction is applied automatically by HuggingFaceEmbeddings
    
    )


# ── Core helpers ──────────────────────────────────────────────────────────────

def embed_query(text: str) -> List[float]:
    model = get_embedding_model()

    if "bge" in settings.embedding_model_local.lower():
        text = "Represent this sentence for searching relevant passages: " + text

    return model.embed_query(text)

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of document texts.
    No instruction prefix for documents — asymmetric by design.
    """
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_chunks(chunks: List[Document]) -> List[dict]:
    """
    Embed a list of LangChain Document chunks.
    Returns a list of dicts with text, embedding, and metadata.

    Table chunks are embedded using their natural language representation
    (which was already written into page_content by the chunker).
    """
    texts   = [c.page_content for c in chunks]
    vectors = embed_documents(texts)

    return [
        {
            "text":      texts[i],
            "embedding": vectors[i],
            "metadata":  chunks[i].metadata,
        }
        for i in range(len(chunks))
    ]


def get_embedding_dimension() -> int:
    """Return the vector dimension for the active model."""
    test_vector = embed_query("dimension probe")
    return len(test_vector)

    # Expected dimensions by model:
    #   BAAI/bge-large-en-v1.5  → 1024  (new default)
    #   BAAI/bge-small-en-v1.5  → 384
    #   all-MiniLM-L6-v2        → 384   (old default)
    #   text-embedding-3-small  → 1536
    #   text-embedding-3-large  → 3072