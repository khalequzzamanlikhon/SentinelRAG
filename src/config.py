"""
Centralized configuration for the SentinelRAG pipeline.

Uses Groq as the primary LLM (via OpenAI-compatible API) and local
sentence-transformers for embeddings — no paid API key required for
the embedding step.
"""

import os
import logging
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Centralized configuration — reads from environment variables / .env file."""

    # --- PRIMARY LLM: Groq (OpenAI-compatible API) ---
    groq_api_key: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str = "llama-3.3-70b-versatile"

    # --- FALLBACK LLM: Gemini ---
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    # Gemini uses LLM_MODEL (same as the old xAI setting, repurposed)
    llm_model: str = "gemini-2.5-flash"

    # --- EMBEDDINGS: Local sentence-transformers ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # Keep OPENAI_API_KEY alias for backward compat but it's no longer required
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")

    # --- LLM Call Resilience ---
    llm_timeout_seconds: int = Field(default=60, ge=5, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # --- Operational Thresholds ---
    max_loop_count: int = Field(default=3, ge=1, le=10)
    temperature_deterministic: float = Field(default=0.0, ge=0.0, le=1.0)
    temperature_creative: float = Field(default=0.3, ge=0.0, le=1.0)
    top_k_documents: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = Field(default=500, ge=10, le=10000)

    # --- Qdrant Configuration ---
    qdrant_mode: Literal["embedded", "server"] = "embedded"
    qdrant_persist_path: str = "./data/qdrant_local_db"
    qdrant_collection_name: str = "sentinel_rag_docs"

    # Qdrant server mode (Docker / standalone)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # --- Logging ---
    log_level: str = "INFO"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v}")
        return upper

    def validate_api_keys(self) -> None:
        """Validate that required API keys are configured."""
        if not self.groq_api_key or self.groq_api_key.startswith("your_"):
            raise ValueError(
                "GROQ_API_KEY is missing or still a placeholder. "
                "Get a free key at https://console.groq.com"
            )
        # Gemini and OpenAI keys are optional — Groq alone is sufficient

    def configure_logging(self) -> None:
        """Configure root logger with the configured log level."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(name)-24s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


# Module-level singleton — import `settings` anywhere to get the configured instance.
settings = Settings()
