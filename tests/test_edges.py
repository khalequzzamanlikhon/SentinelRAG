"""
Unit tests for SentinelRAG edge routing functions.

Tests cover:
  - route_post_grading: generate vs rewrite decision
  - route_post_reranking: always proceeds to generate (NEW)
  - route_post_generation: always finalises with quality warnings
"""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("XAI_API_KEY", "test-xai-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from langchain_core.documents import Document
from src.edges import route_post_grading, route_post_reranking, route_post_generation


# ===========================================================================
# route_post_grading
# ===========================================================================

class TestRoutePostGrading(unittest.TestCase):
    @patch("src.edges.settings")
    def test_generates_when_max_loops_reached(self, mock_settings):
        """loop_count >= max_loop_count → 'generate' regardless of web_search."""
        mock_settings.max_loop_count = 3

        state = {
            "question": "What is RAG?",
            "current_query": "What is RAG?",
            "documents": [],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 3,
            "web_search": True,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "error": None,
            "metadata": {},
        }

        result = route_post_grading(state)
        self.assertEqual(result, "generate")

    @patch("src.edges.settings")
    def test_rewrites_when_web_search_true(self, mock_settings):
        """web_search=True + loops remaining → 'rewrite'."""
        mock_settings.max_loop_count = 5

        state = {
            "question": "Explain embeddings",
            "current_query": "Explain embeddings",
            "documents": [],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 1,
            "web_search": True,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "error": None,
            "metadata": {},
        }

        result = route_post_grading(state)
        self.assertEqual(result, "rewrite")

    @patch("src.edges.settings")
    def test_generates_when_context_valid(self, mock_settings):
        """web_search=False + loops remaining → 'generate'."""
        mock_settings.max_loop_count = 5

        state = {
            "question": "What is a vector store?",
            "current_query": "What is a vector store?",
            "documents": [Document(page_content="A vector store is ...")],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 0,
            "web_search": False,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "error": None,
            "metadata": {},
        }

        result = route_post_grading(state)
        self.assertEqual(result, "generate")


# ===========================================================================
# route_post_reranking  (NEW)
# ===========================================================================

class TestRoutePostReranking(unittest.TestCase):
    def test_always_generates(self):
        """route_post_reranking always returns 'generate'."""
        state = {
            "question": "test",
            "current_query": "test",
            "documents": [],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 0,
            "web_search": False,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "error": None,
            "metadata": {},
        }
        self.assertEqual(route_post_reranking(state), "generate")


# ===========================================================================
# route_post_generation
# ===========================================================================

class TestRoutePostGeneration(unittest.TestCase):
    def test_finalizes_with_documents(self):
        """Documents present → 'finalize'."""
        state = {
            "question": "Explain RAG",
            "current_query": "Explain RAG",
            "documents": [Document(page_content="RAG combines retrieval and generation.")],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 1,
            "web_search": False,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {
                "grounded_score": 0.95,
                "hallucination_count": 0,
            },
            "error": None,
            "metadata": {},
        }

        result = route_post_generation(state)
        self.assertEqual(result, "finalize")

    def test_finalizes_without_documents(self):
        """No documents → safe exit → 'finalize'."""
        state = {
            "question": "Unknown topic",
            "current_query": "Unknown topic",
            "documents": [],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 3,
            "web_search": False,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "error": None,
            "metadata": {},
        }

        result = route_post_generation(state)
        self.assertEqual(result, "finalize")

    def test_finalizes_with_low_grounding(self):
        """Low grounding score + high hallucination count → still 'finalize'."""
        state = {
            "question": "Complex topic",
            "current_query": "Complex topic",
            "documents": [Document(page_content="Some context.")],
            "reranked_documents": [],
            "generation": None,
            "citations": [],
            "conversation_history": [],
            "loop_count": 2,
            "web_search": False,
            "query_strategy": "semantic",
            "retrieval_metrics": {},
            "generation_metrics": {
                "grounded_score": 0.2,
                "hallucination_count": 5,
            },
            "error": None,
            "metadata": {},
        }

        result = route_post_generation(state)
        self.assertEqual(result, "finalize")


if __name__ == "__main__":
    unittest.main()
