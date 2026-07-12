"""
Unit tests for SentinelRAG node functions.

Tests cover:
  - retrieve_node: success path and error handling
  - grade_documents_node: relevance filtering
  - rewrite_query_node: strategy rotation
  - generate_node: generation with metrics
  - rerank_documents_node: cross-encoder style scoring (NEW)
  - extract_citations_node: citation extraction (NEW)

Every external dependency (ChatOpenAI, vectorstore, cache, embeddings)
is mocked so these tests run offline without API keys.
"""

import os
import unittest
from unittest.mock import patch, MagicMock, PropertyMock

os.environ.setdefault("XAI_API_KEY", "test-xai-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from langchain_core.documents import Document
from src.state import AgentState, GradeDocument, QueryStrategy
from src.nodes import (
    retrieve_node,
    grade_documents_node,
    rewrite_query_node,
    generate_node,
    rerank_documents_node,
    extract_citations_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> dict:
    """Build a minimal valid AgentState dict with sensible defaults."""
    base: dict = {
        "question": "What is RAG?",
        "current_query": "What is RAG?",
        "documents": [],
        "reranked_documents": [],
        "generation": None,
        "citations": [],
        "conversation_history": [],
        "web_search": False,
        "query_strategy": QueryStrategy.SEMANTIC,
        "loop_count": 0,
        "retrieval_metrics": {},
        "generation_metrics": {},
        "error": None,
        "metadata": {},
    }
    base.update(overrides)
    return base


# ===========================================================================
# retrieve_node
# ===========================================================================

class TestRetrieveNode(unittest.TestCase):
    @patch("src.nodes.compute_document_similarity", return_value=0.85)
    @patch("src.nodes.compute_retrieval_metrics")
    @patch("src.nodes.settings")
    def test_retrieve_node_success(self, mock_settings, mock_metrics, mock_sim):
        """Happy path: vectorstore returns docs, similarity computed, sorted."""
        mock_settings.top_k_documents = 3

        mock_vs = MagicMock()
        doc1 = Document(page_content="Document about RAG", metadata={})
        doc2 = Document(page_content="Another document", metadata={})
        mock_vs.similarity_search.return_value = [doc1, doc2]

        mock_metrics.return_value = {
            "document_count": 2,
            "avg_similarity_score": 0.85,
        }

        state = _make_state(current_query="What is RAG?")
        result = retrieve_node(state, mock_vs)

        self.assertEqual(len(result["documents"]), 2)
        self.assertEqual(result["retrieval_metrics"]["document_count"], 2)
        for doc in result["documents"]:
            self.assertIn("similarity_score", doc.metadata)
        mock_vs.similarity_search.assert_called_once_with("What is RAG?", k=3)

    @patch("src.nodes.settings")
    def test_retrieve_node_handles_error(self, mock_settings):
        """Vectorstore exception → empty doc list + error."""
        mock_settings.top_k_documents = 3

        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = RuntimeError("DB connection lost")

        state = _make_state(current_query="broken query")
        result = retrieve_node(state, mock_vs)

        self.assertEqual(result["documents"], [])
        self.assertIn("Retrieval failed", result["error"])
        self.assertIn("DB connection lost", result["error"])


# ===========================================================================
# grade_documents_node
# ===========================================================================

class TestGradeDocumentsNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_keeps_relevant(self, mock_settings, mock_llm_cls, mock_cache):
        """binary_score='yes' + high relevance → kept."""
        mock_settings.llm_model = "grok-3-beta"
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

        doc = Document(
            page_content="RAG uses retrieval and generation.",
            metadata={"doc_id": "d1"},
        )
        state = _make_state(current_query="What is RAG?", documents=[doc], loop_count=0)

        result = grade_documents_node(state)

        self.assertEqual(len(result["documents"]), 1, "Relevant doc should be kept")
        self.assertFalse(result["web_search"], "web_search False when docs found")
        self.assertEqual(result["loop_count"], 1, "loop_count increments")

    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_drops_irrelevant(self, mock_settings, mock_llm_cls, mock_cache):
        """binary_score='no' → dropped, web_search=True."""
        mock_settings.llm_model = "grok-3-beta"
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

        doc = Document(
            page_content="Recipe for chocolate cake.",
            metadata={"doc_id": "d2"},
        )
        state = _make_state(current_query="What is RAG?", documents=[doc], loop_count=1)

        result = grade_documents_node(state)

        self.assertEqual(len(result["documents"]), 0, "Irrelevant doc dropped")
        self.assertTrue(result["web_search"], "web_search True when all dropped")
        self.assertEqual(result["loop_count"], 2, "loop_count increments")


# ===========================================================================
# rewrite_query_node
# ===========================================================================

class TestRewriteQueryNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_rotates_strategies(self, mock_settings, mock_llm_cls, mock_cache):
        """Strategy cycles: loop 0→SEMANTIC, 1→KEYWORD, 2→HYBRID, 3→EXPANSION."""
        mock_settings.llm_model = "grok-3-beta"
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

            self.assertEqual(
                result["query_strategy"],
                expected_strategy,
                f"Loop {i} should select {expected_strategy.value}",
            )
            self.assertEqual(result["current_query"], "optimized query text")
            self.assertFalse(result["web_search"], "web_search resets to False")

    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_fallback_on_error(self, mock_settings, mock_llm_cls, mock_cache):
        """LLM error → fall back to original query."""
        mock_settings.llm_model = "grok-3-beta"
        mock_settings.temperature_creative = 0.3
        mock_cache.side_effect = RuntimeError("LLM unavailable")

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        state = _make_state(current_query="original query", loop_count=0)
        result = rewrite_query_node(state)

        self.assertEqual(result["current_query"], "original query")


# ===========================================================================
# generate_node
# ===========================================================================

class TestGenerateNode(unittest.TestCase):
    @patch("src.nodes.detect_hallucinations")
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_returns_generation(
        self, mock_settings, mock_llm_cls, mock_cache, mock_halluc
    ):
        """Happy path: generation with hallucination metrics."""
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.temperature_creative = 0.3
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
        self.assertIn("metadata", result)
        mock_halluc.assert_called_once()

    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_handles_error(self, mock_settings, mock_llm_cls, mock_cache):
        """LLM call failure → error field returned."""
        mock_settings.groq_model = "llama-3.3-70b-versatile"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.temperature_creative = 0.3
        mock_settings.groq_api_key = "test-key"
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3
        mock_cache.side_effect = RuntimeError("API timeout")

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        state = _make_state(question="What is RAG?", documents=[])
        result = generate_node(state)

        self.assertIn("error", result)
        self.assertIn("Generation failed", result["error"])


# ===========================================================================
# rerank_documents_node  (NEW)
# ===========================================================================

class TestRerankDocumentsNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_rerank_returns_sorted(self, mock_settings, mock_llm_cls, mock_cache):
        """Documents should be re-ranked and scored."""
        mock_settings.llm_model = "grok-3-beta"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3

        # Mock LLM response for reranking
        mock_response = MagicMock()
        mock_response.content = "0.85"
        mock_cache.return_value = mock_response

        mock_llm_instance = MagicMock()
        mock_llm_cls.return_value = mock_llm_instance

        docs = [
            Document(page_content="Doc about RAG", metadata={"source": "a.pdf"}),
            Document(page_content="Doc about embeddings", metadata={"source": "b.pdf"}),
        ]
        state = _make_state(
            current_query="What is RAG?",
            documents=docs,
        )

        result = rerank_documents_node(state)

        self.assertIn("reranked_documents", result)
        self.assertEqual(len(result["reranked_documents"]), 2)
        # Each doc should have a rerank_score
        for doc in result["reranked_documents"]:
            self.assertIn("rerank_score", doc.metadata)
        self.assertIn("reranked", result.get("metadata", {}))

    @patch("src.nodes.settings")
    def test_rerank_empty_docs(self, mock_settings):
        """Empty document list should return empty array."""
        state = _make_state(documents=[])
        result = rerank_documents_node(state)
        self.assertEqual(result["reranked_documents"], [])


# ===========================================================================
# extract_citations_node  (NEW)
# ===========================================================================

class TestExtractCitationsNode(unittest.TestCase):
    @patch("src.nodes.cache_llm_call")
    @patch("src.nodes.ChatOpenAI")
    @patch("src.nodes.settings")
    def test_extracts_citations(self, mock_settings, mock_llm_cls, mock_cache):
        """Citations should be extracted from generation text."""
        mock_settings.llm_model = "grok-3-beta"
        mock_settings.temperature_deterministic = 0.0
        mock_settings.llm_timeout_seconds = 60
        mock_settings.llm_max_retries = 3

        # Mock the extractor to return a list of citation dicts
        from src.state import Citation
        result_citations = [
            Citation(
                claim="RAG uses retrieval",
                source_document="doc1.pdf",
                source_excerpt="RAG uses retrieval",
                confidence=0.95,
            )
        ]
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
        self.assertEqual(result["citations"][0]["source_document"], "doc1.pdf")

    @patch("src.nodes.settings")
    def test_empty_gen_returns_empty(self, mock_settings):
        """No generation → empty citations list."""
        state = _make_state(generation="", documents=[])
        result = extract_citations_node(state)
        self.assertEqual(result["citations"], [])


if __name__ == "__main__":
    unittest.main()
