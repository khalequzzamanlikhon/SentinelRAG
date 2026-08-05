"""
Unit tests for SentinelRAG evaluator functions (v2.1).

Tests cover:
  - compute_retrieval_metrics: empty docs, populated, search_sources
  - compute_generation_metrics: word count, coverage, hallucination passthrough
  - NEW: compute_faithfulness
  - NEW: compute_context_precision / compute_context_recall
  - NEW: compute_answer_relevancy
  - NEW: compute_composite_ragas_score
"""

import unittest

from langchain_core.documents import Document
from src.evaluators import (
    compute_retrieval_metrics,
    compute_generation_metrics,
    compute_faithfulness,
    compute_context_precision,
    compute_context_recall,
    compute_answer_relevancy,
    compute_composite_ragas_score,
)


class TestComputeRetrievalMetrics(unittest.TestCase):
    def test_empty_documents(self):
        metrics = compute_retrieval_metrics("query", [])
        self.assertEqual(metrics["document_count"], 0)
        self.assertEqual(metrics["avg_similarity_score"], 0.0)

    def test_populated_documents(self):
        docs = [
            Document(page_content="a", metadata={"similarity_score": 0.9}),
            Document(page_content="b", metadata={"similarity_score": 0.7}),
            Document(page_content="c", metadata={"similarity_score": 0.5}),
        ]
        metrics = compute_retrieval_metrics("query", docs)
        self.assertEqual(metrics["document_count"], 3)
        self.assertAlmostEqual(metrics["avg_similarity_score"], 0.7, places=4)
        self.assertEqual(len(metrics["score_distribution"]), 5)

    def test_search_sources_tracked(self):
        docs = [
            Document(page_content="a", metadata={"similarity_score": 0.9, "search_source": "dense"}),
            Document(page_content="b", metadata={"similarity_score": 0.7, "search_source": "hybrid"}),
        ]
        metrics = compute_retrieval_metrics("query", docs)
        self.assertIn("search_sources", metrics)
        self.assertEqual(metrics["search_sources"].get("dense"), 1)
        self.assertEqual(metrics["search_sources"].get("hybrid"), 1)


class TestComputeGenerationMetrics(unittest.TestCase):
    def test_basic_metrics(self):
        generation = "This is a test. With two sentences."
        metrics = compute_generation_metrics(generation, [], {})
        self.assertEqual(metrics["word_count"], 7)
        self.assertEqual(metrics["sentence_count"], 2)

    def test_hallucination_metrics_passed_through(self):
        hallucination_metrics = {"grounded_score": 0.85, "hallucinated_claims": ["claim 1"]}
        metrics = compute_generation_metrics("Some text.", [], hallucination_metrics)
        self.assertEqual(metrics["grounded_score"], 0.85)
        self.assertEqual(metrics["hallucination_count"], 1)

    def test_document_coverage(self):
        docs = [
            Document(page_content="RAG is a great technique for improving LLMs"),
            Document(page_content="Embeddings are useful for semantic search"),
        ]
        generation = "RAG is a great technique. It uses embeddings for semantic search."
        metrics = compute_generation_metrics(generation, docs, {})
        self.assertGreater(metrics["document_coverage"], 0.0)

    def test_zero_sentence_handling(self):
        metrics = compute_generation_metrics("", [], {})
        self.assertEqual(metrics["word_count"], 0)
        self.assertEqual(metrics["avg_sentence_length"], 0.0)


# ===========================================================================
# RAGAS-style metrics (NEW)
# ===========================================================================

class TestComputeFaithfulness(unittest.TestCase):
    def test_fully_faithful(self):
        result = compute_faithfulness("A is B. C is D.", "", [])
        self.assertGreater(result, 0.8)

    def test_partially_faithful(self):
        result = compute_faithfulness("A is B. C is D. E is F.", "", ["C is D", "E is F"])
        self.assertLess(result, 0.5)

    def test_empty_generation(self):
        result = compute_faithfulness("", "", ["hallucination"])
        self.assertEqual(result, 0.0)


class TestComputeContextPrecision(unittest.TestCase):
    def test_all_relevant(self):
        docs = [
            Document(page_content="a", metadata={"grading": {"binary_score": "yes"}}),
            Document(page_content="b", metadata={"grading": {"binary_score": "yes"}}),
        ]
        self.assertEqual(compute_context_precision(docs), 1.0)

    def test_half_relevant(self):
        docs = [
            Document(page_content="a", metadata={"grading": {"binary_score": "yes"}}),
            Document(page_content="b", metadata={"grading": {"binary_score": "no"}}),
        ]
        self.assertEqual(compute_context_precision(docs), 0.5)

    def test_empty_docs(self):
        self.assertEqual(compute_context_precision([]), 0.0)


class TestComputeContextRecall(unittest.TestCase):
    def test_full_recall(self):
        graded = [Document(page_content="a", metadata={"doc_id": "1"})]
        final = [Document(page_content="a", metadata={"doc_id": "1"})]
        self.assertEqual(compute_context_recall(final, graded), 1.0)

    def test_partial_recall(self):
        graded = [
            Document(page_content="a", metadata={"doc_id": "1"}),
            Document(page_content="b", metadata={"doc_id": "2"}),
        ]
        final = [Document(page_content="a", metadata={"doc_id": "1"})]
        self.assertEqual(compute_context_recall(final, graded), 0.5)


class TestComputeAnswerRelevancy(unittest.TestCase):
    def test_high_relevancy(self):
        score = compute_answer_relevancy("RAG is retrieval augmented generation", "what is RAG")
        self.assertGreater(score, 0.3)

    def test_low_relevancy(self):
        score = compute_answer_relevancy("bananas are yellow fruits", "what is RAG")
        self.assertEqual(score, 0.0)


class TestComputeCompositeRagasScore(unittest.TestCase):
    def test_perfect_score(self):
        result = compute_composite_ragas_score(1.0, 1.0, 1.0, 1.0)
        self.assertAlmostEqual(result["composite_score"], 1.0, places=4)

    def test_zero_score(self):
        result = compute_composite_ragas_score(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(result["composite_score"], 0.0, places=4)

    def test_custom_weights(self):
        custom = {"faithfulness": 0.5, "context_precision": 0.0,
                  "context_recall": 0.5, "answer_relevancy": 0.0}
        result = compute_composite_ragas_score(0.8, 0.0, 0.6, 0.0, weights=custom)
        expected = 0.8 * 0.5 + 0.6 * 0.5
        self.assertAlmostEqual(result["composite_score"], expected, places=4)


if __name__ == "__main__":
    unittest.main()
