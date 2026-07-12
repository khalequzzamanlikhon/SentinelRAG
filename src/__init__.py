"""
SentinelRAG — Self-Correcting Agentic RAG Pipeline.

Core pipeline logic: config, state, nodes, edges, pipeline, evaluators, utils.
"""

from src.config import settings, Settings
from src.state import (
    AgentState,
    QueryStrategy,
    GradeDocument,
    GradeHallucination,
    Citation,
)
from src.nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    rerank_documents_node,
    extract_citations_node,
    detect_hallucinations,
)
from src.edges import route_post_grading, route_post_reranking, route_post_generation
from src.pipeline import assemble_agentic_rag_workflow
from src.evaluators import compute_retrieval_metrics, compute_generation_metrics
from src.utils import (
    cache_llm_call,
    clear_cache,
    compute_document_similarity,
    chunk_document,
    generate_doc_id,
    deduplicate_documents,
    deterministic_hash,
    get_embeddings_client,
    retry_on_failure,
)
from src.exceptions import (
    AgenticRAGError,
    RetrievalError,
    GradingError,
    QueryRewriteError,
    GenerationError,
    HallucinationDetectionError,
    MaxLoopCountExceededError,
)

__all__ = [
    # config
    "settings",
    "Settings",
    # state
    "AgentState",
    "QueryStrategy",
    "GradeDocument",
    "GradeHallucination",
    "Citation",
    # nodes
    "retrieve_node",
    "grade_documents_node",
    "rewrite_query_node",
    "generate_node",
    "rerank_documents_node",
    "extract_citations_node",
    "detect_hallucinations",
    # edges
    "route_post_grading",
    "route_post_reranking",
    "route_post_generation",
    # pipeline
    "assemble_agentic_rag_workflow",
    # evaluators
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    # utils
    "cache_llm_call",
    "clear_cache",
    "compute_document_similarity",
    "chunk_document",
    "generate_doc_id",
    "deduplicate_documents",
    "deterministic_hash",
    "get_embeddings_client",
    "retry_on_failure",
    # exceptions
    "AgenticRAGError",
    "RetrievalError",
    "GradingError",
    "QueryRewriteError",
    "GenerationError",
    "HallucinationDetectionError",
    "MaxLoopCountExceededError",
]
