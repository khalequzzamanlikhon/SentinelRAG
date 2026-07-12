"""
Pydantic schemas and TypedDict for the SentinelRAG agent state.
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


class AgentState(TypedDict):
    """Shared operational state that flows through the LangGraph pipeline.

    Every node reads from and/or writes to this dict. New fields added:
      - reranked_documents  — post-reranking candidates
      - citations           — source-attributed citation list
      - conversation_history — sliding window of previous Q&A turns
    """

    # --- Query ---
    question: str                     # Original user query
    current_query: str                # Active search query (may be rewritten)

    # --- Retrieval ---
    documents: List[Document]         # Vector-store retrieval results
    reranked_documents: List[Document]  # Post-reranking candidates (NEW)
    web_search: bool                  # True = insufficient context found

    # --- Grading & Rewriting ---
    query_strategy: QueryStrategy     # Current rewrite strategy
    loop_count: int                   # Safeguard iteration counter

    # --- Generation ---
    generation: Optional[str]         # Generated response text
    citations: List[Dict[str, Any]]   # Source-attributed citations (NEW)

    # --- Conversation ---
    conversation_history: List[Dict[str, str]]  # Previous Q&A turns (NEW)

    # --- Metrics & Audit ---
    retrieval_metrics: Dict[str, Any]
    generation_metrics: Dict[str, Any]

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
