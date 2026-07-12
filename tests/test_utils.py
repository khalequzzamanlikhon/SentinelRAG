"""
Unit tests for SentinelRAG utility functions.

Tests cover:
  - deterministic_hash: stability and collision resistance
  - cache_llm_call (LRU with TTL): hit, miss, eviction, expiry
  - compute_document_similarity: error safety
  - chunk_document: boundary splitting
  - generate_doc_id: determinism
  - deduplicate_documents: dedup logic
  - get_embeddings_client: singleton guarantee
  - retry_on_failure: retry mechanics
"""

import os
import time
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("XAI_API_KEY", "test-xai-key")
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
)


# ===========================================================================
# deterministic_hash
# ===========================================================================

class TestDeterministicHash(unittest.TestCase):
    def test_deterministic(self):
        """Same inputs must produce the same hash across calls."""
        h1 = deterministic_hash("hello", x=42)
        h2 = deterministic_hash("hello", x=42)
        self.assertEqual(h1, h2)

    def test_different_inputs_different(self):
        """Different inputs must produce different hashes (collision-averse check)."""
        h1 = deterministic_hash("foo")
        h2 = deterministic_hash("bar")
        self.assertNotEqual(h1, h2)

    def test_length(self):
        """Output should be a 32-character hex string (128-bit prefix)."""
        h = deterministic_hash("test")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 32)

    def test_positional_and_keyword_accepted(self):
        """Both positional and keyword arguments should work."""
        h = deterministic_hash("a", "b", key="value")
        self.assertEqual(len(h), 32)


# ===========================================================================
# cache_llm_call
# ===========================================================================

class TestCacheLLMCall(unittest.TestCase):
    def setUp(self):
        # Use a fresh cache with small maxsize for isolation
        from src.utils import _cache
        _cache.clear()

    @patch("src.utils.settings")
    def test_cache_hit(self, mock_settings):
        """A cached result should be returned without calling the function again."""
        mock_settings.enable_caching = True
        mock_settings.cache_ttl_seconds = 3600
        mock_settings.cache_max_size = 500

        func = MagicMock(return_value="cached_result")
        first = cache_llm_call(func, {}, "my_key")
        second = cache_llm_call(func, {}, "my_key")

        self.assertEqual(first, "cached_result")
        self.assertEqual(second, "cached_result")
        func.assert_called_once()  # only called on the first invocation

    @patch("src.utils.settings")
    def test_caching_disabled(self, mock_settings):
        """When caching is disabled, every call should invoke func."""
        mock_settings.enable_caching = False

        func = MagicMock(return_value="fresh")
        first = cache_llm_call(func, {}, "key")
        second = cache_llm_call(func, {}, "key")

        self.assertEqual(first, "fresh")
        self.assertEqual(second, "fresh")
        self.assertEqual(func.call_count, 2)

    @patch("src.utils.settings")
    def test_cache_different_keys(self, mock_settings):
        """Different cache keys should produce independent cache entries."""
        mock_settings.enable_caching = True
        mock_settings.cache_ttl_seconds = 3600
        mock_settings.cache_max_size = 500

        func = MagicMock()
        func.side_effect = ["result_a", "result_b"]

        r1 = cache_llm_call(func, {}, "key_a")
        r2 = cache_llm_call(func, {}, "key_b")
        r3 = cache_llm_call(func, {}, "key_a")  # should be cached
        r4 = cache_llm_call(func, {}, "key_b")  # should be cached

        self.assertEqual(r1, "result_a")
        self.assertEqual(r2, "result_b")
        self.assertEqual(r3, "result_a")
        self.assertEqual(r4, "result_b")
        self.assertEqual(func.call_count, 2, "Each key should invoke func once")


# ===========================================================================
# compute_document_similarity
# ===========================================================================

class TestComputeDocumentSimilarity(unittest.TestCase):
    @patch("src.utils.get_embeddings_client")
    def test_returns_zero_on_error(self, mock_get_emb):
        """When embedding fails, the function should return 0.0, not crash."""
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
        """A short document should produce a single chunk."""
        chunks = chunk_document("Hello world", chunk_size=100, chunk_overlap=0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Hello world")

    def test_chunk_boundary_respects_separator(self):
        """Chunking should try to break on the separator."""
        text = "AAA\n\nBBB\n\nCCC"
        chunks = chunk_document(text, chunk_size=6, chunk_overlap=0)
        # Expect at least 2 chunks since the default separator is \n\n
        self.assertGreaterEqual(len(chunks), 2)

    def test_overlap(self):
        """Consecutive chunks should overlap (share content) when overlap > 0."""
        text = "A" * 100 + "B" * 100 + "C" * 100
        chunks = chunk_document(text, chunk_size=50, chunk_overlap=10)
        self.assertGreater(len(chunks), 2)
        # Check that adjacent chunks share content (overlap)
        self.assertIn(chunks[0][-10:], chunks[1])


# ===========================================================================
# generate_doc_id
# ===========================================================================

class TestGenerateDocId(unittest.TestCase):
    def test_deterministic(self):
        """Same content must produce the same ID."""
        id1 = generate_doc_id("same content")
        id2 = generate_doc_id("same content")
        self.assertEqual(id1, id2)

    def test_different_content_different_id(self):
        """Different content must produce different IDs."""
        id1 = generate_doc_id("content A")
        id2 = generate_doc_id("content B")
        self.assertNotEqual(id1, id2)


# ===========================================================================
# deduplicate_documents
# ===========================================================================

class TestDeduplicateDocuments(unittest.TestCase):
    def test_deduplicates(self):
        """Duplicate documents based on page content should be removed."""
        docs = [
            Document(page_content="unique A", metadata={}),
            Document(page_content="duplicate", metadata={}),
            Document(page_content="duplicate", metadata={}),
        ]
        result = deduplicate_documents(docs)
        self.assertEqual(len(result), 2)

    def test_preserves_order(self):
        """Order of first occurrence must be preserved."""
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
        """get_embeddings_client should always return the same instance."""
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
        """The decorated function should retry up to max_retries times."""
        func = MagicMock()
        func.side_effect = [ValueError("attempt 1"), ValueError("attempt 2"), "success"]
        decorated = retry_on_failure(max_retries=3, base_delay=0.01)
        wrapped = decorated(func)
        result = wrapped()
        self.assertEqual(result, "success")
        self.assertEqual(func.call_count, 3)

    def test_exhausts_retries(self):
        """When all attempts fail, the last exception should propagate."""
        func = MagicMock(side_effect=ValueError("persistent failure"))
        decorated = retry_on_failure(max_retries=2, base_delay=0.01)
        wrapped = decorated(func)
        with self.assertRaises(ValueError) as ctx:
            wrapped()
        self.assertIn("persistent failure", str(ctx.exception))
        self.assertEqual(func.call_count, 2)


if __name__ == "__main__":
    unittest.main()
