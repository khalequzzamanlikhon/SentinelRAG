"""
Unit tests for SentinelRAG edge routing functions (v2.1).

Tests cover:
  - route_post_grading: generate vs rewrite vs web_search (NEW triple route)
  - route_post_reranking: always proceeds to generate
  - route_post_generation: always finalises with quality warnings
"""

import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from src.edges import route_post_grading, route_post_reranking, route_post_generation


def _make_state(**overrides) -> dict:
    base = {
        "question": "What is RAG?",
        "current_query": "What is RAG?",
        "documents": [],
        "reranked_documents": [],
        "bm25_documents": [],
        "web_documents": [],
        "generation": None,
        "citations": [],
        "streaming_tokens": [],
        "conversation_history": [],
        "loop_count": 0,
        "web_search": False,
        "search_source": "unknown",
        "query_strategy": "semantic",
        "retrieval_metrics": {},
        "generation_metrics": {},
        "confidence_calibration": {},
        "guardrail_passed": True,
        "error": None,
        "metadata": {},
    }
    base.update(overrides)
    return base


# ===========================================================================
# route_post_grading (v2.1: 3-way routing)
# ===========================================================================

class TestRoutePostGrading(unittest.TestCase):
    @patch("src.edges.settings")
    def test_generates_when_max_loops_reached(self, mock_settings):
        """loop_count >= max_loop_count → 'generate' regardless of web_search."""
        mock_settings.max_loop_count = 3
        mock_settings.enable_web_search = True
        state = _make_state(loop_count=3, web_search=True)
        self.assertEqual(route_post_grading(state), "generate")

    @patch("src.edges.settings")
    def test_routes_to_web_search_when_enabled(self, mock_settings):
        """web_search=True + web enabled → 'web_search' (NEW v2.1 behavior)."""
        mock_settings.max_loop_count = 5
        mock_settings.enable_web_search = True
        state = _make_state(loop_count=1, web_search=True)
        result = route_post_grading(state)
        self.assertEqual(result, "web_search",
                         "Should route to web_search when web fallback is enabled")

    @patch("src.edges.settings")
    def test_routes_to_rewrite_when_web_disabled(self, mock_settings):
        """web_search=True + web DISABLED → 'rewrite' (old behavior)."""
        mock_settings.max_loop_count = 5
        mock_settings.enable_web_search = False
        state = _make_state(loop_count=1, web_search=True)
        result = route_post_grading(state)
        self.assertEqual(result, "rewrite",
                         "Should route to rewrite when web fallback is disabled")

    @patch("src.edges.settings")
    def test_generates_when_context_valid(self, mock_settings):
        """web_search=False + loops remaining → 'generate'."""
        mock_settings.max_loop_count = 5
        state = _make_state(
            loop_count=0,
            web_search=False,
            documents=[Document(page_content="A vector store is ...")],
        )
        self.assertEqual(route_post_grading(state), "generate")


# ===========================================================================
# route_post_reranking
# ===========================================================================

class TestRoutePostReranking(unittest.TestCase):
    @patch("src.edges.settings")
    def test_always_generates(self, mock_settings):
        mock_settings.similarity_threshold = 0.7
        state = _make_state()
        self.assertEqual(route_post_reranking(state), "generate")

    @patch("src.edges.settings")
    def test_generates_with_low_scores(self, mock_settings):
        """Even with low rerank scores, still proceeds to generate (with warning)."""
        mock_settings.similarity_threshold = 0.7
        state = _make_state(reranked_documents=[
            Document(page_content="x", metadata={"rerank_score": 0.1}),
        ])
        self.assertEqual(route_post_reranking(state), "generate")


# ===========================================================================
# route_post_generation
# ===========================================================================

class TestRoutePostGeneration(unittest.TestCase):
    def test_finalizes_with_documents(self):
        state = _make_state(
            loop_count=1,
            documents=[Document(page_content="RAG combines retrieval and generation.")],
            generation_metrics={"grounded_score": 0.95, "hallucination_count": 0},
        )
        self.assertEqual(route_post_generation(state), "finalize")

    def test_finalizes_without_documents(self):
        state = _make_state(loop_count=3)
        self.assertEqual(route_post_generation(state), "finalize")

    def test_finalizes_with_low_grounding(self):
        state = _make_state(
            loop_count=2,
            documents=[Document(page_content="Some context.")],
            generation_metrics={"grounded_score": 0.2, "hallucination_count": 5},
        )
        self.assertEqual(route_post_generation(state), "finalize")


if __name__ == "__main__":
    unittest.main()
