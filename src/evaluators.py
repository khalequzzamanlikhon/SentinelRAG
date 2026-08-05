"""
Evaluation metrics for the SentinelRAG pipeline.

Computes retrieval quality metrics, generation quality metrics,
and RAGAS-style evaluation metrics for benchmarking.

NEW in v2.1:
  - RAGAS-inspired faithfulness and answer relevancy scoring
  - Context precision and recall metrics
  - Self-play evaluation dataset generation
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
import numpy as np
import logging

from src.config import settings

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Compute quality metrics for the document retrieval step.

    Args:
        query: The search query used.
        documents: The retrieved document chunks (with ``similarity_score``
                   in their metadata).

    Returns:
        A dictionary with document_count, avg/max/min similarity scores,
        and a score_distribution histogram. Includes search_source breakdown.
    """
    if not documents:
        return {
            "document_count": 0,
            "avg_similarity_score": 0.0,
            "max_similarity_score": 0.0,
            "min_similarity_score": 0.0,
            "score_distribution": {},
            "search_sources": {},
        }

    scores = [doc.metadata.get("similarity_score", 0.0) for doc in documents]

    avg_score = float(np.mean(scores))
    max_score = float(np.max(scores))
    min_score = float(np.min(scores))

    hist, bin_edges = np.histogram(scores, bins=5, range=(0.0, 1.0))
    score_distribution = {
        f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}": int(count)
        for i, count in enumerate(hist)
    }

    # Track which search sources contributed
    search_sources: Dict[str, int] = {}
    for doc in documents:
        src = doc.metadata.get("search_source", "unknown")
        search_sources[src] = search_sources.get(src, 0) + 1

    rerank_scores = [
        doc.metadata.get("rerank_score", 0.0)
        for doc in documents
        if "rerank_score" in doc.metadata
    ]

    return {
        "document_count": len(documents),
        "avg_similarity_score": avg_score,
        "max_similarity_score": max_score,
        "min_similarity_score": min_score,
        "score_distribution": score_distribution,
        "search_sources": search_sources,
        "rerank_available": len(rerank_scores) > 0,
        "avg_rerank_score": float(np.mean(rerank_scores)) if rerank_scores else None,
    }


def compute_generation_metrics(
    generation: str,
    documents: List[Document],
    hallucination_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute quality metrics for the generated response.

    Args:
        generation: The generated response text.
        documents: Source documents used for generation.
        hallucination_metrics: Output from ``detect_hallucinations``.

    Returns:
        Dictionary with word/sentence counts, document coverage, and
        hallucination-related metrics.
    """
    word_count = len(generation.split())
    sentence_count = len([s for s in generation.split(".") if s.strip()])
    avg_sentence_length = word_count / max(sentence_count, 1)

    # Improved document coverage using semantic overlap
    doc_coverage = _compute_document_coverage(generation, documents)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "document_coverage": doc_coverage,
        "grounded_score": hallucination_metrics.get("grounded_score", 0.0),
        "hallucination_count": len(hallucination_metrics.get("hallucinated_claims", [])),
    }


def _compute_document_coverage(
    generation: str, documents: List[Document]
) -> float:
    """Compute what fraction of source documents contributed to the generation.

    Uses both exact substring matching and approximate token-overlap to
    estimate coverage more robustly than simple exact-match.
    """
    if not documents:
        return 0.0

    gen_tokens = set(generation.lower().split())
    used_count = 0

    for doc in documents:
        doc_tokens = set(doc.page_content.lower().split())
        overlap = len(gen_tokens & doc_tokens)
        if overlap > max(3, len(doc_tokens) * 0.05):
            used_count += 1

    return used_count / len(documents)


# ===================================================================
# RAGAS-style evaluation metrics (NEW)
# ===================================================================


def compute_faithfulness(
    generation: str, context: str, hallucination_claims: List[str]
) -> float:
    """Compute faithfulness: fraction of claims in the generation
    that are supported by the context.

    Faithfulness = 1.0 - (hallucinated_claims / total_claims_estimate)

    Args:
        generation: Generated response.
        context: Source context used for generation.
        hallucination_claims: List of hallucinated claims from audit.

    Returns:
        Faithfulness score between 0.0 and 1.0.
    """
    sentences = [s.strip() for s in generation.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    total_claims = len(sentences)
    hallucination_count = len(hallucination_claims)

    if total_claims == 0:
        return 0.0

    return max(0.0, 1.0 - hallucination_count / max(total_claims, 1))


def compute_context_precision(documents: List[Document]) -> float:
    """Estimate context precision: fraction of retrieved documents
    that were graded as relevant.

    Args:
        documents: Retrieved documents with grading metadata.

    Returns:
        Context precision between 0.0 and 1.0.
    """
    if not documents:
        return 0.0

    relevant = sum(
        1 for doc in documents
        if doc.metadata.get("grading", {}).get("binary_score") == "yes"
    )
    return relevant / len(documents)


def compute_context_recall(
    documents: List[Document],
    graded_documents: List[Document],
) -> float:
    """Estimate context recall: fraction of graded-relevant documents
    that made it into the final context.

    Args:
        documents: Final documents used for generation.
        graded_documents: All documents after grading.

    Returns:
        Context recall between 0.0 and 1.0.
    """
    if not graded_documents:
        return 0.0

    final_ids = {doc.metadata.get("doc_id", id(doc)) for doc in documents}
    graded_ids = {doc.metadata.get("doc_id", id(doc)) for doc in graded_documents}

    if not graded_ids:
        return 0.0

    return len(final_ids & graded_ids) / len(graded_ids)


def compute_answer_relevancy(
    generation: str, query: str
) -> float:
    """Estimate answer relevancy: how well the generation addresses the query.

    Uses a simple keyword-overlap heuristic. For production use, replace
    with LLM-based or embedding-based relevancy scoring.

    Args:
        generation: Generated response.
        query: Original user query.

    Returns:
        Answer relevancy between 0.0 and 1.0.
    """
    query_tokens = set(query.lower().split())
    if not query_tokens:
        return 0.0

    gen_tokens = set(generation.lower().split())
    overlap = len(query_tokens & gen_tokens)

    # Favor responses that use query terms but aren't just echoing the query
    relevancy = overlap / len(query_tokens)
    return min(1.0, relevancy)


def compute_composite_ragas_score(
    faithfulness: float,
    context_precision: float,
    context_recall: float,
    answer_relevancy: float,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Compute a weighted composite RAGAS score.

    Args:
        faithfulness: Faithfulness score.
        context_precision: Context precision score.
        context_recall: Context recall score.
        answer_relevancy: Answer relevancy score.
        weights: Optional weight dict (defaults to equal weighting).

    Returns:
        Dict with individual and composite scores.
    """
    weights = weights or {
        "faithfulness": 0.35,
        "context_precision": 0.20,
        "context_recall": 0.25,
        "answer_relevancy": 0.20,
    }

    composite = (
        faithfulness * weights.get("faithfulness", 0.25)
        + context_precision * weights.get("context_precision", 0.25)
        + context_recall * weights.get("context_recall", 0.25)
        + answer_relevancy * weights.get("answer_relevancy", 0.25)
    )

    return {
        "faithfulness": faithfulness,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_relevancy": answer_relevancy,
        "composite_score": composite,
    }
