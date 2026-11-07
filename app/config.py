"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="HRA_", extra="ignore")

    # "openai" uses any OpenAI-compatible API (OpenAI, Azure OpenAI, vLLM, Ollama).
    # "offline" uses deterministic local components so tests and CI need no API key.
    provider: Literal["openai", "offline"] = "offline"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant: ":memory:" for local dev, a URL such as http://qdrant:6333 in Docker.
    qdrant_location: str = ":memory:"
    collection: str = "hr_policies"

    docs_dir: Path = ROOT / "data" / "policies"
    chunk_size: int = 600
    chunk_overlap: int = 100

    top_k: int = 4
    # Minimum fused relevance score a chunk needs to be used as evidence.
    min_relevance: float = 0.5


@lru_cache
def get_settings() -> Settings:
    return Settings()
