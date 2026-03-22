from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    app_name:    str = "Knowledge Copilot"
    app_version: str = "0.1.0"
    debug:       bool = True
    upload_dir:  str = "data/uploads"

    # Embeddings
    embedding_provider:     Literal["local", "openai"] = "local"
    embedding_model_local:  str = "all-MiniLM-L6-v2"
    embedding_model_openai: str = "text-embedding-3-small"
    openai_api_key:         str = ""

    class Config:
        env_file = ".env"

settings = Settings()