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
    retrieval_k:                  int   = 15
    retrieval_fetch_k:            int   = 100
    retrieval_score_threshold:    float = 0.15
    retrieval_rerank_threshold:   float = 0.05
    retrieval_max_context_chars:  int   = 12000
    retrieval_mmr_lambda:         float = 0.3
    retrieval_hybrid_alpha:       float = 0.5
    retrieval_hybrid_search:      bool  = True
    retrieval_section_diversity:  bool  = True
    retrieval_min_sections:       int   = 3
    retrieval_max_chunks_per_section: int = 3
    retrieval_source_balancing:   bool  = True
    retrieval_min_sources:        int   = 3
    retrieval_max_chunks_per_doc: int   = 2

    # ── Query Expansion ────────────────────────────────────────────────────────
    query_expansion_enabled:        bool = True
    query_expansion_max_terms:      int  = 6
    query_expansion_domain_terms:   bool = True

    # ── Reranker ──────────────────────────────────────────────────────────────
    reranker_provider: str = "bge"
    reranker_model:    str = "BAAI/bge-reranker-large"
    cohere_api_key:    str = ""

    # ── Summarization ──────────────────────────────────────────────────────────
    summarization_enabled:            bool  = True
    summarization_max_chunks:         int   = 30
    summarization_top_chunks:         int   = 20
    summarization_max_context_chars:  int   = 30000
    summarization_chunk_summary_max_tokens: int = 400
    summarization_global_summary_max_tokens: int = 1500
    summarization_concept_weight:     float = 0.5
    summarization_semantic_weight:    float = 0.3
    summarization_min_concept_freq:   int   = 2

    # ── Summarization Enhancement ──────────────────────────────────────────────
    summarization_section_importance:       bool  = True
    summarization_importance_concept_weight: float = 0.35
    summarization_importance_entity_weight:  float = 0.25
    summarization_importance_position_weight: float = 0.15
    summarization_importance_coverage_weight: float = 0.25
    summarization_cross_section_boost:       float = 0.3
    summarization_min_per_important_section: int   = 4
    summarization_min_per_minor_section:    int   = 2
    summarization_deduplicate_concepts:      bool  = True
    summarization_max_concepts_per_section: int   = 3
    summarization_global_entity_min_sections: int = 2
    summarization_concise_max_points:       int   = 3
    summarization_repetition_penalty:       float = 0.15

    # ── Deduplication & Novelty ────────────────────────────────────────────────
    retrieval_jaccard_threshold:  float = 0.82
    retrieval_novelty_scoring:    bool  = True
    retrieval_novelty_lambda:     float = 0.4

    # ── Pre-generation Synthesis ───────────────────────────────────────────────
    synthesis_enabled:            bool  = True
    synthesis_max_context_chars:  int   = 4000
    synthesis_min_relation_score: float = 0.3

    # ── Confidence & Citation ──────────────────────────────────────────────────
    confidence_enabled:           bool  = True
    confidence_threshold:         float = 0.5
    citation_grounding_check:     bool  = True
    citation_min_entity_overlap:  float = 0.3

    # ── Conversation Memory ────────────────────────────────────────────────────
    memory_max_turns:             int   = 10
    memory_summary_max_tokens:    int   = 500
    memory_entity_tracking:       bool  = True
    memory_compression_threshold: int   = 20

    # ── Query Analysis ─────────────────────────────────────────────────────────
    query_ambiguity_detection:    bool  = True
    query_adversarial_filtering:  bool  = True
    query_ambiguity_threshold:    float = 0.4

    # ── Performance ────────────────────────────────────────────────────────────
    performance_use_ivf_index:    bool  = False
    performance_ivf_nlist:        int   = 100
    performance_cache_ttl:        int   = 300
    performance_async_retrieval:  bool  = False

    # ── Evaluation / Debug ────────────────────────────────────────────────────
    eval_log_retrieved_chunks: bool = True
    eval_log_scores:           bool = True
    eval_log_reranking:        bool = True
    eval_trace_output_path:    str  = ""
    eval_trace_enabled:        bool = True

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key:     str = "CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING"
    jwt_algorithm:      str = "HS256"
    jwt_expire_minutes: int = 10080

    mongodb_url:     str = "mongodb://localhost:27017"
    mongodb_db_name: str = "knowledge_copilot"

    clerk_secret_key:      str = ""
    clerk_publishable_key: str = ""


settings = Settings()