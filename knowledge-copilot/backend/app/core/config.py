from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
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
    llm_temperature: float = 0.1
    llm_max_tokens:  int   = 1500

    groq_api_key: str = ""
    groq_model:   str = "llama-3.1-70b-versatile"

    llm_model:       str = "gpt-3.5-turbo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "llama3.2"

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunking_default_strategy: str = "structure_aware"
    chunking_default_size:     int = 700
    chunking_default_overlap:  int = 150

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_k:               int   = 8
    retrieval_fetch_k:         int   = 30
    retrieval_score_threshold: float = 0.20
    retrieval_max_context_chars: int = 8000
    retrieval_mmr_lambda:      float = 0.3
    retrieval_hybrid_alpha:    float = 0.3
    retrieval_section_diversity: bool = True
    retrieval_min_sections:    int   = 2

    # ── Query Expansion ────────────────────────────────────────────────────────
    query_expansion_enabled:   bool = True
    query_expansion_max_terms: int  = 6

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_provider: str = "bge"
    reranker_model:    str = "BAAI/bge-reranker-large"
    cohere_api_key:    str = ""

    # ── Evaluation / Debug ────────────────────────────────────────────────────
    eval_log_retrieved_chunks: bool = True
    eval_log_scores:           bool = True
    eval_log_reranking:        bool = True

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key:     str = "CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING"
    jwt_algorithm:      str = "HS256"
    jwt_expire_minutes: int = 10080

    mongodb_url:     str = "mongodb://localhost:27017"
    mongodb_db_name: str = "knowledge_copilot"

    clerk_secret_key:      str = ""
    clerk_publishable_key: str = ""


settings = Settings()