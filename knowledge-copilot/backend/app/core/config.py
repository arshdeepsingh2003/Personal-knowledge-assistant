from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Knowledge Copilot"
    app_version: str = "0.1.0"
    debug: bool = True
    upload_dir: str = "data/uploads"

    class Config:
        env_file = ".env"

settings = Settings()