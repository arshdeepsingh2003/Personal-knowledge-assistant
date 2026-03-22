from functools import lru_cache
from typing import List

from langchain_core.documents import Document
from app.core.config import settings

# This file converts text → numbers (vectors)

''' 
PDF / Notes
   ↓
Chunking
   ↓
embed_chunks()   ← THIS FILE 💥
   ↓
Vector DB (FAISS)
   ↓
Search using embed_query()
   ↓
LLM response (RAG)
'''

# ── Model factory ─────────────────────────────────────────────────────────────
# @lru_cache means this function only runs ONCE no matter how many times it's
# called. The model loads into memory on first call and is reused after that.

@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Return a LangChain-compatible embeddings object.
    Cached after first load — model stays in memory.
    """
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set in .env. "
                "Either add it or switch EMBEDDING_PROVIDER=local"
            )
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.embedding_model_openai,
            openai_api_key=settings.openai_api_key,
        )

    # Default: local SentenceTransformers model
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    return SentenceTransformerEmbeddings(
        model_name=settings.embedding_model_local
    )


# ── Core helpers ──────────────────────────────────────────────────────────────

# Converts one sentence → vector
# Used during: Search time (user asks question)
def embed_query(text: str) -> List[float]:
    """Embed a single query string. Used at search time."""
    model = get_embedding_model()
    return model.embed_query(text)

# Converts multiple texts → vectors
def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of text strings. Used when indexing chunks."""
    model = get_embedding_model()
    return model.embed_documents(texts)


def embed_chunks(chunks: List[Document]) -> List[dict]:
    """
    Embed a list of LangChain Document chunks.
    Returns a list of dicts with text, embedding vector, and metadata.
    This is the format the vector DB will consume in Phase 5.
    """
    texts  = [c.page_content for c in chunks]
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
    """
    Return the vector dimension for the active model.
    Needed when initialising FAISS in Phase 5.
    """
    test_vector = embed_query("dimension probe")
    return len(test_vector)