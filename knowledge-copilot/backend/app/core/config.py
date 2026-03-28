from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Knowledge Copilot"
    app_version: str = "0.1.0"
    debug: bool = True
    upload_dir: str = "data/uploads"

    # Embeddings
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model_local: str = "all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"
    openai_api_key: str = ""

    # Vector store
    vector_store_provider: Literal["faiss", "chroma"] = "faiss"
    vector_store_path: str = "data/vector_store"

    # 🔥 FIX HERE (IMPORTANT)
    llm_provider: Literal["openai", "ollama"] = "ollama"

    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3:latest"


settings = Settings()