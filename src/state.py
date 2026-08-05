"""
Pydantic schemas and TypedDict for the SentinelRAG agent state.

NEW in v2.1:
  - Web search results & metadata
  - Streaming token accumulator
  - Confidence calibration fields
  - Guardrail pass/fail status
  - BM25 sparse results
"""

from typing import List, TypedDict, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from enum import Enum


class QueryStrategy(str, Enum):
    """Strategies for query rewriting."""

    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    EXPANSION = "expansion"


class SearchSource(str, Enum):
    """Where the retrieved context came from."""
    VECTOR = "vector"
    BM25 = "bm25"
    WEB = "web"
    HYBRID = "hybrid"


class AgentState(TypedDict):
    """Shared operational state that flows through the LangGraph pipeline.

    Every node reads from and/or writes to this dict.
    """

    # --- Query ---
    question: str                     # Original user query
    current_query: str                # Active search query (may be rewritten)

    # --- Retrieval ---
    documents: List[Document]         # Vector-store retrieval results
    reranked_documents: List[Document]  # Post-reranking candidates
    bm25_documents: List[Document]     # BM25 sparse retrieval results (NEW)
    web_documents: List[Document]      # Web search results (NEW)
    web_search: bool                  # True = insufficient context found
    search_source: str                # Which source provided the context (NEW)

    # --- Grading & Rewriting ---
    query_strategy: QueryStrategy     # Current rewrite strategy
    loop_count: int                   # Safeguard iteration counter

    # --- Generation ---
    generation: Optional[str]         # Generated response text
    streaming_tokens: List[str]       # Accumulated streaming tokens (NEW)
    citations: List[Dict[str, Any]]   # Source-attributed citations

    # --- Conversation ---
    conversation_history: List[Dict[str, str]]  # Previous Q&A turns

    # --- Metrics & Audit ---
    retrieval_metrics: Dict[str, Any]
    generation_metrics: Dict[str, Any]
    confidence_calibration: Dict[str, Any]  # Calibrated confidence scores (NEW)
    guardrail_passed: bool            # Whether input passed content safety (NEW)

    # --- Error Handling ---
    error: Optional[str]

    # --- General Metadata ---
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Structured LLM output schemas
# ---------------------------------------------------------------------------


class GradeDocument(BaseModel):
    """Assessment score for document relevance checking."""

    relevance_score: float = Field(
        description="Relevance score between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).",
        ge=0.0,
        le=1.0,
    )
    binary_score: str = Field(
        description="Relevance grade. Use 'yes' if document matches query intent, otherwise 'no'.",
        pattern="^(yes|no)$",
    )
    reasoning: str = Field(description="One-sentence rationalization for the grading decision.")


class GradeHallucination(BaseModel):
    """Assessment score for generation truthfulness against context."""

    grounded_score: float = Field(
        description="Grounded score between 0.0 (completely hallucinated) and 1.0 (fully grounded).",
        ge=0.0,
        le=1.0,
    )
    binary_score: str = Field(
        description="Grounding grade. Use 'yes' if generation is completely grounded in context, otherwise 'no'.",
        pattern="^(yes|no)$",
    )
    reasoning: str = Field(description="Explanation of alignment or identified hallucination points.")
    hallucinated_claims: List[str] = Field(
        description="List of specific claims in the generation that are not supported by the context.",
        default_factory=list,
    )


class Citation(BaseModel):
    """A single source-attributed citation extracted from generation."""

    claim: str = Field(description="The factual claim made in the generation.")
    source_document: str = Field(description="Filename or identifier of the source document.")
    source_excerpt: str = Field(description="The exact excerpt from the source supporting the claim.")
    confidence: float = Field(
        description="Confidence that this claim is supported by the source (0.0 – 1.0).",
        ge=0.0,
        le=1.0,
    )


class CitationList(BaseModel):
    """Wrapper model for extracting a list of citations via structured output.

    LangChain's ``with_structured_output()`` cannot accept ``list[Citation]``
    directly because ``inspect.signature()`` fails on generic type aliases.
    This wrapper provides a concrete Pydantic model instead.
    """

    citations: List[Citation] = Field(
        description="List of extracted source citations.",
        default_factory=list,
    )


class GuardrailResult(BaseModel):
    """Result of input guardrail validation."""

    passed: bool = Field(description="Whether the input passed all guardrail checks.")
    reason: str = Field(description="Explanation if the input was blocked.")
    blocked_pattern: Optional[str] = Field(
        default=None,
        description="The specific pattern that triggered the block, if any."
    )
