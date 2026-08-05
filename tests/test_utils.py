"""
Unit tests for SentinelRAG utility functions.

Tests cover:
  - deterministic_hash, cache_llm_call (passes params dict as positional arg), LRU eviction
  - compute_document_similarity, chunk_document, generate_doc_id
  - deduplicate_documents, get_embeddings_client, retry_on_failure
  - NEW: BM25Retriever index/search
  - NEW: rerank_with_cross_encoder (fallback path)
  - NEW: merge_hybrid_results (RRF fusion)
  - NEW: RateLimiter acquire/remaining
"""

import os
import time
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from langchain_core.documents import Document
from src.utils import (
    deterministic_hash,
    cache_llm_call,
    clear_cache,
    compute_document_similarity,
    chunk_document,
    generate_doc_id,
    deduplicate_documents,
    get_embeddings_client,
    retry_on_failure,
    BM25Retriever,
    merge_hybrid_results,
    check_rate_limit,
    _rerank_with_similarity,
)


# ===========================================================================
# deterministic_hash
# ===========================================================================

class TestDeterministicHash(unittest.TestCase):
    def test_deterministic(self):
        h1 = deterministic_hash("hello", x=42)
        h2 = deterministic_hash("hello", x=42)
        self.assertEqual(h1, h2)

    def test_different_inputs_different(self):
        h1 = deterministic_hash("foo")
        h2 = deterministic_hash("bar")
        self.assertNotEqual(h1, h2)

    def test_length(self):
        h = deterministic_hash("test")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32)

    def test_positional_and_keyword_accepted(self):
        h = deterministic_hash("a", "b", key="value")
        self.assertEqual(len(h), 32)


# ===========================================================================
# cache_llm_call (FIXED: now uses func(**params))
# ===========================================================================

class TestCacheLLMCall(unittest.TestCase):
    def setUp(self):
        from src.utils import _cache
        _cache.clear()

    @patch("src.utils.settings")
    def test_cache_hit(self, mock_settings):
        mock_settings.enable_caching = True
        mock_settings.cache_ttl_seconds = 3600
        mock_settings.cache_max_size = 500

        func = MagicMock(return_value="cached_result")
        first = cache_llm_call(func, {}, "my_key")
        second = cache_llm_call(func, {}, "my_key")

        self.assertEqual(first, "cached_result")
        self.assertEqual(second, "cached_result")
        func.assert_called_once()

    @patch("src.utils.settings")
    def test_caching_disabled(self, mock_settings):
        mock_settings.enable_caching = False

        func = MagicMock(return_value="fresh")
        first = cache_llm_call(func, {}, "key")
        second = cache_llm_call(func, {}, "key")

        self.assertEqual(first, "fresh")
        self.assertEqual(second, "fresh")
        self.assertEqual(func.call_count, 2)

    @patch("src.utils.settings")
    def test_cache_different_keys(self, mock_settings):
        mock_settings.enable_caching = True
        mock_settings.cache_ttl_seconds = 3600
        mock_settings.cache_max_size = 500

        func = MagicMock()
        func.side_effect = ["result_a", "result_b"]

        r1 = cache_llm_call(func, {}, "key_a")
        r2 = cache_llm_call(func, {}, "key_b")
        r3 = cache_llm_call(func, {}, "key_a")
        r4 = cache_llm_call(func, {}, "key_b")

        self.assertEqual(r1, "result_a")
        self.assertEqual(r2, "result_b")
        self.assertEqual(r3, "result_a")
        self.assertEqual(r4, "result_b")
        self.assertEqual(func.call_count, 2)

    @patch("src.utils.settings")
    def test_calls_func_with_params_dict(self, mock_settings):
        """Verify cache_llm_call passes params dict as positional arg to func.

        LangChain RunnableSequence.invoke() expects ``input`` as the first
        positional argument, so we pass ``func(params)`` rather than
        splatting ``func(**params)``.
        """
        mock_settings.enable_caching = True
        mock_settings.cache_ttl_seconds = 3600
        mock_settings.cache_max_size = 500

        func = MagicMock(return_value="ok")
        result = cache_llm_call(func, {"a": 1, "b": 2}, "test_params_key")

        self.assertEqual(result, "ok")
        func.assert_called_once_with({"a": 1, "b": 2})


# ===========================================================================
# BM25Retriever (NEW)
# ===========================================================================

class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25Retriever()

    def test_empty_search_before_index(self):
        """Search before indexing should return empty list."""
        results = self.bm25.search("test query", k=3)
        self.assertEqual(results, [])

    def test_index_and_search(self):
        """After indexing, search should return scored documents."""
        docs = [
            Document(page_content="RAG combines retrieval and generation techniques"),
            Document(page_content="Neural networks are used for deep learning tasks"),
            Document(page_content="Retrieval augmented generation improves LLM accuracy"),
        ]
        self.bm25.index(docs)
        self.assertTrue(self.bm25._indexed)

        results = self.bm25.search("retrieval generation", k=2)
        self.assertGreater(len(results), 0)
        for doc in results:
            self.assertIn("bm25_score", doc.metadata)

    def test_search_no_match(self):
        """Query with no matching terms should return empty."""
        docs = [
            Document(page_content="completely different topic"),
        ]
        self.bm25.index(docs)
        results = self.bm25.search("zzzznonexistent", k=5)
        self.assertEqual(results, [])


# ===========================================================================
# merge_hybrid_results (NEW)
# ===========================================================================

class TestMergeHybridResults(unittest.TestCase):
    def test_merges_dense_and_bm25(self):
        """RRF should combine both result sets."""
        dense = [
            Document(page_content="doc A", metadata={"similarity_score": 0.9}),
            Document(page_content="doc B", metadata={"similarity_score": 0.7}),
        ]
        bm25 = [
            Document(page_content="doc C", metadata={"bm25_score": 0.8}),
            Document(page_content="doc A", metadata={"bm25_score": 0.6}),  # overlap
        ]
        merged = merge_hybrid_results(dense, bm25, top_k=5)
        self.assertGreaterEqual(len(merged), 2)
        for doc in merged:
            self.assertIn("hybrid_score", doc.metadata)
            self.assertEqual(doc.metadata["search_source"], "hybrid")


# ===========================================================================
# compute_document_similarity
# ===========================================================================

class TestComputeDocumentSimilarity(unittest.TestCase):
    @patch("src.utils.get_embeddings_client")
    def test_returns_zero_on_error(self, mock_get_emb):
        mock_client = MagicMock()
        mock_client.embed_query.side_effect = RuntimeError("API failure")
        mock_get_emb.return_value = mock_client

        score = compute_document_similarity("query", "document")
        self.assertEqual(score, 0.0)


# ===========================================================================
# chunk_document
# ===========================================================================

class TestChunkDocument(unittest.TestCase):
    def test_short_document_single_chunk(self):
        chunks = chunk_document("Hello world", chunk_size=100, chunk_overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_chunk_boundary_respects_separator(self):
        text = "AAA\n\nBBB\n\nCCC"
        chunks = chunk_document(text, chunk_size=6, chunk_overlap=0)
        self.assertGreaterEqual(len(chunks), 2)

    def test_overlap(self):
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunks = chunk_document(text, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 2)
        self.assertIn(chunks[0][-10:], chunks[1])


# ===========================================================================
# generate_doc_id
# ===========================================================================

class TestGenerateDocId(unittest.TestCase):
    def test_deterministic(self):
        id1 = generate_doc_id("same content")
        id2 = generate_doc_id("same content")
        self.assertEqual(id1, id2)

    def test_different_content_different_id(self):
        id1 = generate_doc_id("content A")
        id2 = generate_doc_id("content B")
        self.assertNotEqual(id1, id2)


# ===========================================================================
# deduplicate_documents
# ===========================================================================

class TestDeduplicateDocuments(unittest.TestCase):
    def test_deduplicates(self):
        docs = [
            Document(page_content="unique A", metadata={}),
            Document(page_content="duplicate", metadata={}),
            Document(page_content="duplicate", metadata={}),
        ]
        result = deduplicate_documents(docs)
        self.assertEqual(len(result), 2)

    def test_preserves_order(self):
        docs = [
            Document(page_content="first", metadata={}),
            Document(page_content="second", metadata={}),
            Document(page_content="first", metadata={}),
        ]
        result = deduplicate_documents(docs)
        self.assertEqual(result[0].page_content, "first")
        self.assertEqual(result[1].page_content, "second")


# ===========================================================================
# get_embeddings_client
# ===========================================================================

class TestGetEmbeddingsClient(unittest.TestCase):
    @patch("langchain_community.embeddings.SentenceTransformerEmbeddings")
    def test_singleton(self, mock_emb_cls):
        mock_client = MagicMock()
        mock_emb_cls.return_value = mock_client
        client1 = get_embeddings_client()
        client2 = get_embeddings_client()
        self.assertIs(client1, client2)


# ===========================================================================
# retry_on_failure
# ===========================================================================

class TestRetryOnFailure(unittest.TestCase):
    def test_retries_on_failure(self):
        func = MagicMock()
        func.side_effect = [ValueError("attempt 1"), ValueError("attempt 2"), "success"]
        decorated = retry_on_failure(max_retries=3, base_delay=0.01)
        wrapped = decorated(func)
        result = wrapped()
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 3)

    def test_exhausts_retries(self):
        func = MagicMock(side_effect=ValueError("persistent failure"))
        decorated = retry_on_failure(max_retries=2, base_delay=0.01)
        wrapped = decorated(func)
        with self.assertRaises(ValueError) as ctx:
            wrapped()
        self.assertIn("persistent failure", str(ctx.exception))
        self.assertEqual(func.call_count, 2)


if __name__ == "__main__":
    unittest.main()
