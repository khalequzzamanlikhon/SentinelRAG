"""
Centralized configuration for the SentinelRAG pipeline.

Uses Groq as the primary LLM (via OpenAI-compatible API) and local
sentence-transformers for embeddings — no paid API key required for
the embedding step.

NEW in v2.1:
  - Tavily web search integration
  - Cross-encoder reranker (BGE-reranker-v2-m3)
  - BM25 hybrid search support
  - Rate limiting & input guardrails
  - LangFuse tracing (optional)
"""

import logging
import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator
from src.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Centralized configuration — reads from environment variables / .env file."""

    # --- PRIMARY LLM: Groq (OpenAI-compatible API) ---
    groq_api_key: Optional[str] = Field(default=None, validation_alias="GROQ_API_KEY")
    groq_model: str = "llama-3.3-70b-versatile"

    # --- FALLBACK LLM: Gemini ---
    gemini_api_key: Optional[str] = Field(default=None, validation_alias="GEMINI_API_KEY")
    gemini_model: str = "gemini-2.5-flash"

    # Legacy alias — kept for backward compatibility
    llm_model: str = "gemini-2.5-flash"

    # --- EMBEDDINGS: Local sentence-transformers ---
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    # --- CROSS-ENCODER RERANKER: Local (NEW) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = Field(default=10, ge=1, le=50)
    enable_cross_encoder: bool = True

    # --- WEB SEARCH: Tavily (NEW) ---
    tavily_api_key: Optional[str] = Field(default=None, validation_alias="TAVILY_API_KEY")
    tavily_search_depth: Literal["basic", "advanced"] = "advanced"
    tavily_max_results: int = Field(default=5, ge=1, le=20)
    enable_web_search: bool = True

    # --- HYBRID SEARCH: BM25 + Dense (NEW) ---
    enable_hybrid_search: bool = True
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

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

    # --- Guardrails (NEW) ---
    max_query_length: int = Field(default=2000, ge=100, le=10000)
    enable_guardrails: bool = True
    blocked_patterns: list[str] = Field(
        default_factory=lambda: [
            "ignore all previous instructions",
            "disregard",
            "system prompt",
            "DAN ",
        ]
    )

    # --- LangFuse Tracing (NEW, optional) ---
    langfuse_public_key: Optional[str] = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = "https://cloud.langfuse.com"
    enable_tracing: bool = False

    # --- Rate Limiting (NEW) ---
    rate_limit_requests: int = Field(default=30, ge=1, le=1000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)

    # --- Logging ---
    log_level: str = "INFO"

    # Keep OPENAI_API_KEY alias for backward compat
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")

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
        """Validate that required API keys are configured.

        Raises:
            ConfigurationError: If required keys are missing or still placeholders.
        """
        errors: list[str] = []

        if not self.groq_api_key or self.groq_api_key.startswith("your_"):
            errors.append(
                "GROQ_API_KEY is missing or still a placeholder. "
                "Get a free key at https://console.groq.com"
            )

        if self.enable_web_search:
            if not self.tavily_api_key or self.tavily_api_key.startswith("your_"):
                logger.warning(
                    "TAVILY_API_KEY not set — web search will be disabled. "
                    "Get a free key at https://tavily.com"
                )
                # Auto-disable web search to prevent runtime errors
                object.__setattr__(self, "enable_web_search", False)

        if errors:
            raise ConfigurationError(
                message="; ".join(errors),
                details={"missing_keys": errors},
            )

    def configure_logging(self) -> None:
        """Configure root logger with the configured log level."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format="%(asctime)s | %(name)-24s | %(levelname)-5s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def setup_tracing(self) -> None:
        """Initialise LangFuse tracing if configured.

        Sets up the LangFuse callback handler for automatic trace
        collection across all LangChain/LangGraph operations.
        """
        if not (self.enable_tracing and self.langfuse_public_key and self.langfuse_secret_key):
            return

        try:
            import langfuse  # noqa: F401
        except ImportError:
            logger.warning("langfuse package not installed — tracing disabled.")
            return

        try:
            os.environ.setdefault("LANGFUSE_PUBLIC_KEY", self.langfuse_public_key)
            os.environ.setdefault("LANGFUSE_SECRET_KEY", self.langfuse_secret_key)
            os.environ.setdefault("LANGFUSE_HOST", self.langfuse_host)

            from langfuse.callback import CallbackHandler
            handler = CallbackHandler()
            logger.info("LangFuse tracing enabled at %s", self.langfuse_host)

            # Store handler for potential use by the application
            self._langfuse_handler = handler  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning("Failed to initialise LangFuse tracing: %s", e)


# Module-level singleton — import `settings` anywhere to get the configured instance.
settings = Settings()
