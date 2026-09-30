"""Application settings, loaded from environment variables / .env file."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[2]  # repo root when running locally


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(_ROOT / ".env"), extra="ignore")

    # LLM / embeddings (Google Gemini via Google AI Studio free API key)
    google_api_key: str = ""
    llm_model: str = "gemini-3.5-flash"
    available_models: str = (
        "gemini-3.8-flash,gemini-3.7-flash,gemini-3.5-flash,"
        "gemini-3.1-flash-lite,gemini-3.5-flash-lite"
    )
    embedding_model: str = "models/gemini-embedding-001"

    # PostgreSQL (orders)
    database_url: str = "postgresql+psycopg2://agent:agentpass@localhost:5432/agentdb"

    # ChromaDB server
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_collection: str = "company_policies_gemini"
    rag_top_k: int = 4

    # Misc
    data_dir: str = str(_ROOT / "data")
    log_level: str = "INFO"

    @property
    def model_list(self) -> list[str]:
        models = [m.strip() for m in self.available_models.split(",") if m.strip()]
        if self.llm_model not in models:
            models.insert(0, self.llm_model)
        return models


@lru_cache
def get_settings() -> Settings:
    return Settings()
