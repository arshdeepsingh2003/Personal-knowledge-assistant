from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    app_name:    str  = "Knowledge Copilot"
    app_version: str  = "0.1.0"
    debug:       bool = True

    # Paths
    upload_dir: str = "data/uploads"

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Switched from all-MiniLM-L6-v2 (384d) to bge-large-en-v1.5 (1024d).
    # bge-large scores ~8 points higher on BEIR retrieval benchmarks and
    # handles numeric/tabular content significantly better.
    #
    # Other strong options:
    #   BAAI/bge-large-en-v1.5        — best local, 1024d  ← default
    #   text-embedding-3-large        — best OpenAI, 3072d
    #   BAAI/bge-m3                   — multilingual, 1024d
    embedding_provider:     Literal["local", "openai"] = "local"
    embedding_model_local:  str = "BAAI/bge-large-en-v1.5"
    embedding_model_openai: str = "text-embedding-3-large"
    openai_api_key:         str = ""

    # Vector store
    vector_store_provider: Literal["faiss", "chroma"] = "faiss"
    vector_store_path:     str = "data/vector_store"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider:    Literal["groq", "openai", "ollama"] = "groq"
    llm_temperature: float = 0.1   # lowered from 0.2 — more deterministic for tables
    llm_max_tokens:  int   = 1500  # raised from 1024 — tables need more output space

    groq_api_key: str = ""
    groq_model:   str = "llama-3.1-70b-versatile"

    llm_model:       str = "gpt-3.5-turbo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "llama3.2"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    # Increased k: tables often score lower than prose (0.35-0.45 range)
    # so we cast a wider net and let the reranker sort it out.
    retrieval_k:               int   = 8      # was 5
    retrieval_score_threshold: float = 0.25   # was 0.30 — tables score lower
    retrieval_max_context_chars: int = 4000   # was 3000 — tables need more space

    # ── Reranker ──────────────────────────────────────────────────────────────
    # Cross-encoder reranking after initial vector retrieval.
    # Dramatically improves table chunk selection.
    # Options:
    #   "bge"    — BAAI/bge-reranker-large (local, free, recommended)
    #   "cohere" — Cohere rerank API (paid, best accuracy)
    #   "none"   — disabled
    reranker_provider: str = "bge"
    reranker_model:    str = "BAAI/bge-reranker-large"
    cohere_api_key:    str = ""

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key:     str = "CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING"
    jwt_algorithm:      str = "HS256"
    jwt_expire_minutes: int = 10080

    mongodb_url:     str = "mongodb://localhost:27017"
    mongodb_db_name: str = "knowledge_copilot"

    clerk_secret_key:      str = ""
    clerk_publishable_key: str = ""

    class Config:
        env_file = ".env"


settings = Settings()