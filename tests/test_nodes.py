"""
Unit tests for SentinelRAG node functions (v2.1).

Tests cover:
  - retrieve_node: hybrid (dense + BM25) success + error
  - grade_documents_node: relevance filtering + rate limit
  - web_search_node: Tavily integration (NEW)
  - rewrite_query_node: strategy rotation
  - rerank_documents_node: cross-encoder reranking (NEW behavior)
  - generate_node: generation with metrics + error raising
  - extract_citations_node: citation extraction

Every external dependency (ChatOpenAI, vectorstore, BM25, cross-encoder)
is mocked so these tests run offline.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from langchain_core.documents import Document
from src.state import AgentState, GradeDocument, QueryStrategy
from src.nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    rerank_documents_node,
    web_search_node,
    extract_citations_node,
)
from src.exceptions import RetrievalError, GenerationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> dict:
    """Build a minimal valid AgentState dict with all v2.1 fields."""
    base: dict = {
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
        "web_search": False,
        "search_source": "unknown",
        "query_strategy": QueryStrategy.SEMANTIC,
        "loop_count": 0,
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
# retrieve_node (v2.1: now uses hybrid search + raises RetrievalError)
# ===========================================================================

class TestRetrieveNode(unittest.TestCase):
    @patch("src.nodes.get_bm25_retriever")
    @patch("src.nodes.compute_document_similarity", return_value=0.85)
    @patch("src.nodes.compute_retrieval_metrics")
    @patch("src.nodes.settings")
    def test_retrieve_node_success(self, mock_settings, mock_metrics, mock_sim, mock_bm25):
        """Happy path: vectorstore + BM25, merged via RRF."""
        mock_settings.top_k_documents = 3
        mock_settings.enable_hybrid_search = True
        mock_settings.dense_weight = 0.7
        mock_settings.bm25_weight = 0.3

        mock_vs = MagicMock()
        doc1 = Document(page_content="Document about RAG", metadata={})
        doc2 = Document(page_content="Another document", metadata={})
        mock_vs.similarity_search.return_value = [doc1, doc2]

        # Mock BM25 to return results with proper float scores
        bm25_inst = MagicMock()
        bm25_doc = Document(page_content="BM25 doc", metadata={"bm25_score": 0.7})
        bm25_inst.search.return_value = [bm25_doc]
        mock_bm25.return_value = bm25_inst

        mock_metrics.return_value = {"document_count": 2, "avg_similarity_score": 0.85}

        state = _make_state(current_query="What is RAG?")
        result = retrieve_node(state, mock_vs)

        self.assertGreater(len(result["documents"]), 0)
        self.assertIn("retrieval_metrics", result)
        mock_vs.similarity_search.assert_called_once_with("What is RAG?", k=3)

    @patch("src.nodes.settings")
    def test_retrieve_node_raises_on_error(self, mock_settings):
        """Vectorstore exception → raises RetrievalError (v2.1 behavior)."""
        mock_settings.top_k_documents = 3
        mock_settings.enable_hybrid_search = False

        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = RuntimeError("DB connection lost")

        state = _make_state(current_query="broken query")
        with self.assertRaises(RetrievalError) as ctx:
            retrieve_node(state, mock_vs)
        self.assertIn("Retrieval failed", str(ctx.exception))
        self.assertIn("DB connection lost", str(ctx.exception))


# ===========================================================================
# grade_documents_node
# ===========================================================================

class TestGradeDocumentsNode(unittest.TestCase):
    @patch("src.nodes.check_rate_limit", return_value=True)
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_keeps_relevant(self, mock_settings, mock_llm_cls, mock_cache, mock_rl):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.similarity_threshold = 0.7

        grade_result = MagicMock()
        grade_result.binary_score = "yes"
        grade_result.relevance_score = 0.9
        grade_result.reasoning = "Directly relevant."
        mock_cache.return_value = grade_result

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = MagicMock()

        doc = Document(page_content="RAG uses retrieval and generation.", metadata={"doc_id": "d1"})
        state = _make_state(current_query="What is RAG?", documents=[doc], loop_count=0)

        result = grade_documents_node(state)
        self.assertEqual(len(result["documents"]), 1)
        self.assertFalse(result["web_search"])
        self.assertEqual(result["loop_count"], 1)

    @patch("src.nodes.check_rate_limit", return_value=True)
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_drops_irrelevant(self, mock_settings, mock_llm_cls, mock_cache, mock_rl):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.similarity_threshold = 0.7

        grade_result = MagicMock()
        grade_result.binary_score = "no"
        grade_result.relevance_score = 0.2
        grade_result.reasoning = "Off-topic."
        mock_cache.return_value = grade_result

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = MagicMock()

        doc = Document(page_content="Recipe for chocolate cake.", metadata={"doc_id": "d2"})
        state = _make_state(current_query="What is RAG?", documents=[doc], loop_count=1)

        result = grade_documents_node(state)
        self.assertEqual(len(result["documents"]), 0)
        self.assertTrue(result["web_search"])
        self.assertEqual(result["loop_count"], 2)

    @patch("src.nodes.check_rate_limit", return_value=False)
    @patch("src.nodes.settings")
    def test_rate_limited_skips_grading(self, mock_settings, mock_rl):
        """Rate limited → passes through ungraded documents."""
        mock_settings.temperature_deterministic = 0.0

        doc = Document(page_content="some content")
        state = _make_state(documents=[doc], loop_count=0)
        result = grade_documents_node(state)
        self.assertEqual(len(result["documents"]), 1)
        self.assertTrue(result["metadata"].get("rate_limited"))


# ===========================================================================
# web_search_node (NEW)
# ===========================================================================

class TestWebSearchNode(unittest.TestCase):
    @patch("src.nodes.settings")
    def test_web_search_disabled(self, mock_settings):
        """When web search is disabled, returns empty."""
        mock_settings.enable_web_search = False

        state = _make_state(current_query="test query")
        result = web_search_node(state)
        self.assertEqual(result["web_documents"], [])
        self.assertFalse(result["web_search"])

    @patch("src.nodes.settings")
    def test_web_search_returns_results(self, mock_settings):
        """Web search returns documents that replace the current doc list."""
        mock_settings.enable_web_search = True

        # Patch the import inside web_search_node
        with patch("src.web_search.search_web_safe") as mock_search:
            web_doc = Document(page_content="Web search result", metadata={"source": "web"})
            mock_search.return_value = [web_doc]

            state = _make_state(current_query="test query")
            result = web_search_node(state)

            self.assertEqual(len(result["web_documents"]), 1)
            self.assertEqual(len(result["documents"]), 1)
            self.assertEqual(result["search_source"], "web")
            self.assertTrue(result["metadata"].get("web_search_used"))
            self.assertTrue(result["metadata"].get("web_search_attempted"))


# ===========================================================================
# rewrite_query_node
# ===========================================================================

class TestRewriteQueryNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_rotates_strategies(self, mock_settings, mock_llm_cls, mock_cache):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_creative = 0.3

        mock_response = MagicMock()
        mock_response.content = "  optimized query text  "
        mock_cache.return_value = mock_response

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        strategies = list(QueryStrategy)
        for i, expected_strategy in enumerate(strategies):
            state = _make_state(current_query="failing query", loop_count=i)
            result = rewrite_query_node(state)
            self.assertEqual(result["query_strategy"], expected_strategy)
            self.assertEqual(result["current_query"], "optimized query text")
            self.assertFalse(result["web_search"])

    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_fallback_on_error(self, mock_settings, mock_llm_cls, mock_cache):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_creative = 0.3
        mock_cache.side_effect = RuntimeError("LLM unavailable")

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        state = _make_state(current_query="original query", loop_count=0)
        result = rewrite_query_node(state)
        self.assertEqual(result["current_query"], "original query")


# ===========================================================================
# rerank_documents_node (v2.1: now uses cross-encoder)
# ===========================================================================

class TestRerankDocumentsNode(unittest.TestCase):
    @patch("src.nodes.rerank_with_cross_encoder")
    @patch("src.nodes.settings")
    def test_rerank_uses_cross_encoder(self, mock_settings, mock_rerank):
        """Should call the cross-encoder reranker, not LLM."""
        mock_settings.enable_cross_encoder = True

        doc = Document(page_content="Doc about RAG", metadata={"source": "a.pdf"})
        doc.metadata["rerank_score"] = 0.85
        doc.metadata["rerank_method"] = "cross_encoder"
        mock_rerank.return_value = [doc]

        state = _make_state(current_query="What is RAG?", documents=[doc])
        result = rerank_documents_node(state)

        self.assertIn("reranked_documents", result)
        self.assertEqual(len(result["reranked_documents"]), 1)
        self.assertIn("rerank_score", result["reranked_documents"][0].metadata)
        self.assertTrue(result["metadata"].get("reranked"))
        mock_rerank.assert_called_once()

    @patch("src.nodes.settings")
    def test_rerank_disabled(self, mock_settings):
        """When cross-encoder disabled, passes docs through unchanged."""
        mock_settings.enable_cross_encoder = False

        doc = Document(page_content="Doc about RAG")
        state = _make_state(documents=[doc])
        result = rerank_documents_node(state)
        self.assertEqual(result["reranked_documents"], [doc])

    def test_rerank_empty_docs(self):
        state = _make_state(documents=[])
        result = rerank_documents_node(state)
        self.assertEqual(result["reranked_documents"], [])


# ===========================================================================
# generate_node (v2.1: now raises GenerationError)
# ===========================================================================

class TestGenerateNode(unittest.TestCase):
    @patch("src.nodes.detect_hallucinations")
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_returns_generation(self, mock_settings, mock_llm_cls, mock_cache, mock_halluc):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.groq_api_key = "test-key"
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3

        mock_gen = MagicMock()
        mock_gen.content = "RAG stands for Retrieval-Augmented Generation."
        mock_cache.return_value = mock_gen

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        mock_halluc.return_value = {
            "grounded_score": 0.95,
            "binary_score": "yes",
            "reasoning": "Fully grounded.",
            "hallucinated_claims": [],
        }

        doc = Document(page_content="RAG combines retrieval and generation.")
        state = _make_state(question="What is RAG?", documents=[doc])
        result = generate_node(state)

        self.assertIn("generation_metrics", result)
        self.assertEqual(result["generation_metrics"]["grounded_score"], 0.95)
        self.assertIn("streaming_tokens", result)

    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_raises_generation_error(self, mock_settings, mock_llm_cls, mock_cache):
        """Generation failure now raises GenerationError."""
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.groq_api_key = "test-key"
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3
        mock_cache.side_effect = RuntimeError("API timeout")

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        state = _make_state(question="What is RAG?", documents=[])
        with self.assertRaises(GenerationError) as ctx:
            generate_node(state)
        self.assertIn("Generation failed", str(ctx.exception))


# ===========================================================================
# extract_citations_node
# ===========================================================================

class TestExtractCitationsNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_extracts_citations(self, mock_settings, mock_llm_cls, mock_cache):
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3

        from src.state import Citation, CitationList
        result_citations = CitationList(citations=[
            Citation(claim="RAG uses retrieval", source_document="doc1.pdf",
                     source_excerpt="RAG uses retrieval", confidence=0.95)
        ])
        mock_cache.return_value = result_citations

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance
        mock_llm_instance.with_structured_output.return_value = MagicMock()

        state = _make_state(
            generation="RAG uses retrieval and generation.",
            documents=[Document(page_content="RAG uses retrieval", metadata={"source": "doc1.pdf"})],
        )
        result = extract_citations_node(state)

        self.assertIn("citations", result)
        self.assertEqual(len(result["citations"]), 1)
        self.assertEqual(result["citations"][0]["claim"], "RAG uses retrieval")

    @patch("src.nodes.settings")
    def test_empty_gen_returns_empty(self, mock_settings):
        state = _make_state(generation="", documents=[])
        result = extract_citations_node(state)
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
