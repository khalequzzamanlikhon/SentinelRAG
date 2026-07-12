"""
Evaluation metrics for the SentinelRAG pipeline.

Computes retrieval quality metrics and generation quality metrics
including document coverage and relevance scoring.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_retrieval_metrics(query: str, documents: List[Document]) -> Dict[str, Any]:
    """Compute quality metrics for the document retrieval step.

    Args:
        query: The search query used.
        documents: The retrieved document chunks (with ``similarity_score``
                   in their metadata).

    Returns:
        A dictionary with ``document_count``, ``avg_similarity_score``,
        ``max_similarity_score``, ``min_similarity_score``, and a
        ``score_distribution`` histogram.
    """
    if not documents:
        return {
            "document_count": 0,
            "avg_similarity_score": 0.0,
            "max_similarity_score": 0.0,
            "min_similarity_score": 0.0,
            "score_distribution": {},
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

    return {
        "document_count": len(documents),
        "avg_similarity_score": avg_score,
        "max_similarity_score": max_score,
        "min_similarity_score": min_score,
        "score_distribution": score_distribution,
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

    doc_coverage = 0.0
    if documents:
        first_sentences = [
            ".".join(doc.page_content.split(".")[:2]) for doc in documents
        ]
        used_docs = sum(
            1 for first_sent in first_sentences if first_sent in generation
        )
        doc_coverage = used_docs / len(documents)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "document_coverage": doc_coverage,
        "grounded_score": hallucination_metrics.get("grounded_score", 0.0),
        "hallucination_count": len(hallucination_metrics.get("hallucinated_claims", [])),
    }
