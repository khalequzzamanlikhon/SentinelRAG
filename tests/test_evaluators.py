"""
Unit tests for SentinelRAG evaluator functions.

Tests cover:
  - compute_retrieval_metrics: empty docs, populated docs, histogram shape
  - compute_generation_metrics: word count, document coverage, hallucination metrics
"""

import os
import unittest

os.environ.setdefault("XAI_API_KEY", "test-xai-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from langchain_core.documents import Document
from src.evaluators import compute_retrieval_metrics, compute_generation_metrics


# ===========================================================================
# compute_retrieval_metrics
# ===========================================================================

class TestComputeRetrievalMetrics(unittest.TestCase):
    def test_empty_documents(self):
        """Empty document list should yield default zero-metrics."""
        metrics = compute_retrieval_metrics("query", [])
        self.assertEqual(metrics["document_count"], 0)
        self.assertEqual(metrics["avg_similarity_score"], 0.0)
        self.assertEqual(metrics["max_similarity_score"], 0.0)
        self.assertEqual(metrics["min_similarity_score"], 0.0)
        self.assertEqual(metrics["score_distribution"], {})

    def test_populated_documents(self):
        """Documents with similarity_scores should produce well-formed metrics."""
        docs = [
            Document(page_content="a", metadata={"similarity_score": 0.9}),
            Document(page_content="b", metadata={"similarity_score": 0.7}),
            Document(page_content="c", metadata={"similarity_score": 0.5}),
        ]
        metrics = compute_retrieval_metrics("query", docs)

        self.assertEqual(metrics["document_count"], 3)
        self.assertAlmostEqual(metrics["avg_similarity_score"], 0.7, places=4)
        self.assertEqual(metrics["max_similarity_score"], 0.9)
        self.assertEqual(metrics["min_similarity_score"], 0.5)

        # Should have a 5-bin histogram
        self.assertEqual(len(metrics["score_distribution"]), 5)

    def test_no_similarity_score_metadata(self):
        """Documents missing the similarity_score key should default to 0.0."""
        docs = [
            Document(page_content="a", metadata={}),
            Document(page_content="b", metadata={}),
        ]
        metrics = compute_retrieval_metrics("query", docs)
        self.assertEqual(metrics["document_count"], 2)
        self.assertEqual(metrics["avg_similarity_score"], 0.0)


# ===========================================================================
# compute_generation_metrics
# ===========================================================================

class TestComputeGenerationMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        """Basic word/sentence counts should be computed correctly."""
        generation = "This is a test. With two sentences."
        metrics = compute_generation_metrics(generation, [], {})
        self.assertEqual(metrics["word_count"], 7)
        self.assertEqual(metrics["sentence_count"], 2)
        self.assertAlmostEqual(metrics["avg_sentence_length"], 3.5, places=4)

    def test_hallucination_metrics_passed_through(self):
        """Hallucination metrics should be forwarded into the result."""
        hallucination_metrics = {
            "grounded_score": 0.85,
            "hallucinated_claims": ["claim 1"],
        }
        metrics = compute_generation_metrics("Some text.", [], hallucination_metrics)
        self.assertEqual(metrics["grounded_score"], 0.85)
        self.assertEqual(metrics["hallucination_count"], 1)

    def test_document_coverage(self):
        """Coverage should reflect which docs appear in the generation."""
        docs = [
            Document(page_content="RAG is a great technique."),
            Document(page_content="Embeddings are useful."),
        ]
        generation = "RAG is a great technique. It uses embedding."
        metrics = compute_generation_metrics(generation, docs, {})
        # The first sentence of doc1 appears verbatim in the generation
        self.assertGreater(metrics["document_coverage"], 0.0)

    def test_zero_sentence_handling(self):
        """Empty generation should not cause division by zero."""
        metrics = compute_generation_metrics("", [], {})
        self.assertEqual(metrics["word_count"], 0)
        self.assertEqual(metrics["sentence_count"], 0)
        self.assertEqual(metrics["avg_sentence_length"], 0.0)


if __name__ == "__main__":
    unittest.main()
