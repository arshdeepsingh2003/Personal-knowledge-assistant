from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # App
    app_name:    str  = "Knowledge Copilot"
    app_version: str  = "0.1.0"
    debug:       bool = True

    # Paths
    upload_dir: str = "data/uploads"

    # Embeddings
    embedding_provider:     Literal["local", "openai"] = "local"
    embedding_model_local:  str = "all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"
    openai_api_key:         str = ""

    # Vector store
    vector_store_provider: Literal["faiss", "chroma"] = "faiss"
    vector_store_path:     str = "data/vector_store"

    # LLM
    llm_provider:    Literal["openai", "ollama"] = "openai"
    llm_model:       str   = "gpt-3.5-turbo"
    llm_temperature: float = 0.2
    llm_max_tokens:  int   = 1024
    ollama_base_url: str   = "http://localhost:11434"
    ollama_model:    str   = "llama3.2"

    # ── Auth ─────────────────────────────────────────────────────────────────
    # JWT
    jwt_secret_key:      str = "CHANGE_THIS_TO_A_RANDOM_64_CHAR_STRING"
    jwt_algorithm:       str = "HS256"
    jwt_expire_minutes:  int = 60 * 24 * 7   # 7 days

    # MongoDB
    mongodb_url:     str = "mongodb://localhost:27017"
    mongodb_db_name: str = "knowledge_copilot"

    # Clerk (for Google OAuth)
    clerk_secret_key:      str = ""   # sk_live_... from Clerk dashboard
    clerk_publishable_key: str = ""   # pk_live_... exposed to frontend

    class Config:
        env_file = ".env"


settings = Settings()