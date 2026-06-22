from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    # ── App ────────────────────────────────────────────────────────────────────
    app_name:    str  = "Knowledge Copilot"
    app_version: str  = "0.1.0"
    debug:       bool = True
    cors_origins: str = "http://localhost:3000"

    # Supabase Storage
    supabase_url:    str = ""
    supabase_key:    str = ""
    supabase_bucket: str = "documents"

    # ── Embeddings ────────────────────────────────────────────────────────────
    embedding_provider:     Literal["local", "openai"] = "local"
    embedding_model_local:  str  = "BAAI/bge-large-en-v1.5"
    embedding_model_openai: str  = "text-embedding-3-large"
    openai_api_key:         str  = ""

    # Vector store
    vector_store_provider: Literal["qdrant"] = "qdrant"

    # ── Qdrant Cloud ─────────────────────────────────────────────────────────
    qdrant_url:        str = ""
    qdrant_api_key:    str = ""
    qdrant_collection: str = "knowledge_copilot"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider:    Literal["groq", "openai", "ollama"] = "groq"
    llm_temperature: float = 0.1
    llm_max_tokens:  int   = 3000

    groq_api_key: str = ""
    groq_model:   str = "llama-3.1-70b-versatile"

    llm_model:       str = "gpt-3.5-turbo"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model:    str = "llama3.2"

    # ── Chunking ───────────────────────────────────────────────────────────────
    chunking_default_strategy: str = "semantic"
    chunking_default_size:     int = 700
    chunking_default_overlap:  int = 150
    chunking_table_preserve_rows: bool = True
    chunking_semantic_break_threshold: float = 0.45

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_k:                  int   = 8
    retrieval_fetch_k:            int   = 100
    retrieval_score_threshold:    float = 0.25
    retrieval_rerank_threshold:   float = 0.05
    retrieval_max_context_chars:  int   = 16000
    retrieval_mmr_lambda:         float = 0.3
    retrieval_hybrid_alpha:       float = 0.5
    retrieval_hybrid_search:      bool  = True
    retrieval_section_diversity:  bool  = True
    retrieval_min_sections:       int   = 1
    retrieval_max_chunks_per_section: int = 6
    retrieval_source_balancing:   bool  = True
    retrieval_min_sources:        int   = 1
    retrieval_max_chunks_per_doc: int   = 5
    retrieval_force_section_context: bool = True
    retrieval_min_chunks_for_synthesis: int = 3

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

    # ── Adjacent Chunk Expansion ───────────────────────────────────────────────
    retrieval_chunk_expansion_enabled: bool = True
    retrieval_expansion_window:        int  = 2

    # ── Pre-generation Synthesis ───────────────────────────────────────────────
    synthesis_enabled:            bool  = True
    synthesis_max_context_chars:  int   = 6000
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
    performance_cache_ttl:        int   = 300
    performance_async_retrieval:  bool  = False

    # ── Evaluation / Debug ────────────────────────────────────────────────────
    eval_log_retrieved_chunks: bool = True
    eval_log_scores:           bool = True
    eval_log_reranking:        bool = True
    eval_trace_output_path:    str  = ""
    eval_trace_enabled:        bool = True
    retrieval_debug_mode:      bool = False

    # ── Table QA ──────────────────────────────────────────────────────────────
    table_qa_enabled:            bool = True
    table_qa_retrieve_full_row:  bool = True
    table_qa_preserve_columns:   bool = True
    table_qa_detect_references:  bool = True

    # ── Multi-Hop Reasoning ───────────────────────────────────────────────────
    multihop_enabled:            bool = True
    multihop_max_rounds:         int  = 2
    multihop_aggregate_evidence: bool = True
    multihop_query_decomposition: bool = True
    multihop_min_sections:       int  = 2

    # ── Completeness Check ────────────────────────────────────────────────────
    completeness_check_enabled:  bool = True
    completeness_require_metrics: bool = True
    completeness_require_comparisons: bool = True
    completeness_require_projections: bool = True
    completeness_max_expansion_chars: int = 2000

    # ── Answer Generation ─────────────────────────────────────────────────────
    answer_require_all_evidence: bool = True
    answer_include_statistics:   bool = True
    answer_include_comparisons:  bool = True
    answer_min_sources_for_synthesis: int = 2
    answer_require_citations:    bool = True

    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key:     str = "CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING"
    jwt_algorithm:      str = "HS256"
    jwt_expire_minutes: int = 10080

    mongodb_url:     str = "mongodb://localhost:27017"
    mongodb_db_name: str = "knowledge_copilot"

    clerk_secret_key:      str = ""
    clerk_publishable_key: str = ""


settings = Settings()