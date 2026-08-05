"""
SentinelRAG — Self-Correcting Agentic RAG Pipeline.

Core pipeline logic: config, state, nodes, edges, pipeline, evaluators, utils,
web_search, guardrails.

v2.1: Added hybrid BM25 search, cross-encoder reranking, Tavily web search
fallback, input guardrails, RAGAS evaluation, streaming generation.
"""

from src.config import settings, Settings
from src.state import (
    AgentState,
    QueryStrategy,
    SearchSource,
    GradeDocument,
    GradeHallucination,
    Citation,
    CitationList,
    GuardrailResult,
)
from src.nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    generate_node_stream,
    rerank_documents_node,
    web_search_node,
    extract_citations_node,
    detect_hallucinations,
)
from src.edges import route_post_grading, route_post_reranking, route_post_generation
from src.pipeline import (
    assemble_agentic_rag_workflow,
    assemble_agentic_rag_workflow_async,
)
from src.evaluators import (
    compute_retrieval_metrics,
    compute_generation_metrics,
    compute_faithfulness,
    compute_context_precision,
    compute_context_recall,
    compute_answer_relevancy,
    compute_composite_ragas_score,
)
from src.utils import (
    cache_llm_call,
    clear_cache,
    compute_document_similarity,
    chunk_document,
    generate_doc_id,
    deduplicate_documents,
    deterministic_hash,
    get_embeddings_client,
    get_reranker,
    rerank_with_cross_encoder,
    get_bm25_retriever,
    BM25Retriever,
    merge_hybrid_results,
    check_rate_limit,
    retry_on_failure,
)
from src.web_search import search_web, search_web_safe
from src.guardrails import validate_query, sanitize_query
from src.exceptions import (
    AgenticRAGError,
    RetrievalError,
    GradingError,
    QueryRewriteError,
    GenerationError,
    HallucinationDetectionError,
    MaxLoopCountExceededError,
    WebSearchError,
    GuardrailViolationError,
    RerankerError,
    BM25IndexError,
    StreamingError,
    ConfigurationError,
)

__all__ = [
    # config
    "settings",
    "Settings",
    # state
    "AgentState",
    "QueryStrategy",
    "SearchSource",
    "GradeDocument",
    "GradeHallucination",
    "Citation",
    "CitationList",
    "GuardrailResult",
    # nodes
    "retrieve_node",
    "grade_documents_node",
    "rewrite_query_node",
    "generate_node",
    "generate_node_stream",
    "rerank_documents_node",
    "web_search_node",
    "extract_citations_node",
    "detect_hallucinations",
    # edges
    "route_post_grading",
    "route_post_reranking",
    "route_post_generation",
    # pipeline
    "assemble_agentic_rag_workflow",
    "assemble_agentic_rag_workflow_async",
    # evaluators
    "compute_retrieval_metrics",
    "compute_generation_metrics",
    "compute_faithfulness",
    "compute_context_precision",
    "compute_context_recall",
    "compute_answer_relevancy",
    "compute_composite_ragas_score",
    # utils
    "cache_llm_call",
    "clear_cache",
    "compute_document_similarity",
    "chunk_document",
    "generate_doc_id",
    "deduplicate_documents",
    "deterministic_hash",
    "get_embeddings_client",
    "get_reranker",
    "rerank_with_cross_encoder",
    "get_bm25_retriever",
    "BM25Retriever",
    "merge_hybrid_results",
    "check_rate_limit",
    "retry_on_failure",
    # web_search
    "search_web",
    "search_web_safe",
    # guardrails
    "validate_query",
    "sanitize_query",
    # exceptions
    "AgenticRAGError",
    "RetrievalError",
    "GradingError",
    "QueryRewriteError",
    "GenerationError",
    "HallucinationDetectionError",
    "MaxLoopCountExceededError",
    "WebSearchError",
    "GuardrailViolationError",
    "RerankerError",
    "BM25IndexError",
    "StreamingError",
    "ConfigurationError",
]
