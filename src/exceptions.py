"""
Custom exception hierarchy for the SentinelRAG agentic RAG pipeline.

All exceptions inherit from ``AgenticRAGError`` so callers can catch
pipeline-level errors uniformly.

NEW in v2.1: added WebSearchError, GuardrailViolationError, RerankerError,
BM25IndexError, StreamingError, ConfigurationError.
"""


class AgenticRAGError(Exception):
    """Base exception class for Agentic RAG system."""

    def __init__(self, message: str = "", details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class RetrievalError(AgenticRAGError):
    """Exception raised when document retrieval fails."""


class GradingError(AgenticRAGError):
    """Exception raised when document grading fails."""


class QueryRewriteError(AgenticRAGError):
    """Exception raised when query rewriting fails."""


class GenerationError(AgenticRAGError):
    """Exception raised when response generation fails."""


class HallucinationDetectionError(AgenticRAGError):
    """Exception raised when hallucination detection fails."""


class MaxLoopCountExceededError(AgenticRAGError):
    """Exception raised when the maximum loop count is exceeded."""


class WebSearchError(AgenticRAGError):
    """Exception raised when web search (Tavily) fails."""


class GuardrailViolationError(AgenticRAGError):
    """Exception raised when input fails content safety guardrails."""


class RerankerError(AgenticRAGError):
    """Exception raised when cross-encoder reranking fails."""


class BM25IndexError(AgenticRAGError):
    """Exception raised when BM25 sparse index operations fail."""


class StreamingError(AgenticRAGError):
    """Exception raised when token streaming encounters an error."""


class ConfigurationError(AgenticRAGError):
    """Exception raised when the system is misconfigured (e.g., missing API keys)."""
