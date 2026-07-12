"""
Utility functions for the SentinelRAG pipeline.

Provides:
- Deterministic hashing for cache keys
- LRU cache with TTL for LLM calls
- Singleton SentenceTransformer client (local embeddings, no API key)
- Retry decorator for API resilience
- Document chunking and similarity computation
"""

import hashlib
import time
import logging
import functools
from typing import Any, Callable, Dict, Optional, List
from collections import OrderedDict
from threading import Lock

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document

from src.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic hashing
# ---------------------------------------------------------------------------


def deterministic_hash(*args: Any, **kwargs: Any) -> str:
    """Produce a deterministic SHA-256 hex digest from positional/keyword args.

    Unlike Python's built-in ``hash()``, this is stable across interpreter
    restarts and platforms, making it suitable for persistent cache keys.
    """
    hasher = hashlib.sha256()
    for arg in args:
        hasher.update(str(arg).encode("utf-8"))
    for key in sorted(kwargs.keys()):
        hasher.update(str(key).encode("utf-8"))
        hasher.update(str(kwargs[key]).encode("utf-8"))
    return hasher.hexdigest()[:32]  # 128-bit prefix is sufficient


# ---------------------------------------------------------------------------
# Singleton SentenceTransformer (local embeddings, no API key needed)
# ---------------------------------------------------------------------------

_embeddings_client = None
_embeddings_lock = Lock()


def get_embeddings_client():
    """Return a shared singleton ``SentenceTransformerEmbeddings`` instance.

    Uses a local BGE model (no API key required). The model is downloaded
    once on first use and cached locally. Compatible with LangChain's
    ``Embeddings`` protocol (``embed_query`` / ``embed_documents``).
    """
    global _embeddings_client
    if _embeddings_client is not None:
        return _embeddings_client
    with _embeddings_lock:
        if _embeddings_client is None:
            from langchain_community.embeddings import SentenceTransformerEmbeddings

            logger.info(
                "Loading local embedding model '%s' (first load downloads the model)...",
                settings.embedding_model,
            )
            _embeddings_client = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model,
            )
            logger.info("Embedding model loaded successfully.")
        return _embeddings_client


# ---------------------------------------------------------------------------
# Thread-safe LRU cache with TTL
# ---------------------------------------------------------------------------


class _LRUCache:
    """Simple thread-safe LRU cache with TTL eviction, bound by *maxsize*."""

    def __init__(self, maxsize: int = 500, ttl: int = 3600):
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        """Return cached value if present and fresh, else ``None``."""
        with self._lock:
            if key not in self._store:
                return None
            entry = self._store[key]
            if time.time() - entry["timestamp"] >= self._ttl:
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry["result"]

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting LRU items if over capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = {"result": value, "timestamp": time.time()}
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)


_cache = _LRUCache(maxsize=settings.cache_max_size, ttl=settings.cache_ttl_seconds)


def cache_llm_call(
    func: Callable,
    params: Dict[str, Any],
    cache_key: str,
    ttl: Optional[int] = None,
) -> Any:
    """Cache the result of an LLM call to reduce costs and improve latency.

    Args:
        func: The callable to invoke on cache miss.
        params: Keyword arguments forwarded to *func*.
        cache_key: The cache lookup key.
        ttl: Override TTL in seconds (defaults to ``settings.cache_ttl_seconds``).

    Returns:
        The cached or freshly-computed result.
    """
    if not settings.enable_caching:
        return func(**params)

    cached = _cache.get(cache_key)
    if cached is not None:
        logger.debug("Cache HIT for key=%s", cache_key[:40])
        return cached

    logger.debug("Cache MISS for key=%s", cache_key[:40])
    result = func(params)
    _cache.set(cache_key, result)
    return result


def clear_cache() -> None:
    """Clear the entire LLM call cache. Useful for testing."""
    _cache.clear()


# ---------------------------------------------------------------------------
# Retry decorator for LLM API resilience
# ---------------------------------------------------------------------------


def retry_on_failure(
    max_retries: Optional[int] = None,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    allowed_exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator that retries a callable with exponential backoff.

    Args:
        max_retries: Max attempts (defaults to ``settings.llm_max_retries``).
        base_delay: Initial delay in seconds before the first retry.
        backoff_factor: Multiplier for the delay on each subsequent retry.
        allowed_exceptions: Tuple of exception types that trigger a retry.

    Usage::

        @retry_on_failure()
        def my_fallible_func(...):
            ...
    """
    if max_retries is None:
        max_retries = settings.llm_max_retries

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            delay = base_delay
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    last_exc = e
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. "
                            "Retrying in %.1fs...",
                            attempt,
                            max_retries,
                            getattr(func, "__name__", repr(func)),
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Document similarity
# ---------------------------------------------------------------------------


def compute_document_similarity(query: str, document: str) -> float:
    """Compute cosine similarity between *query* and *document* embeddings.

    Uses the shared ``SentenceTransformer`` singleton to avoid redundant
    model loading.
    """
    try:
        client = get_embeddings_client()
        query_emb = np.array(client.embed_query(query)).reshape(1, -1)
        doc_emb = np.array(client.embed_query(document)).reshape(1, -1)
        return float(cosine_similarity(query_emb, doc_emb)[0][0])
    except Exception as e:
        logger.error("Failed to compute similarity: %s", e)
        return 0.0


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------


def chunk_document(
    document: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separator: str = "\n\n",
) -> List[str]:
    """Split *document* into overlapping chunks on natural boundaries.

    Args:
        document: Raw text to split.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Overlap (in characters) between consecutive chunks.
        separator: Preferred boundary delimiter.

    Returns:
        List of text chunks (non-empty).
    """
    chunks: List[str] = []
    start = 0
    doc_length = len(document)

    while start < doc_length:
        end = start + chunk_size
        if end < doc_length:
            last_sep = document.rfind(separator, start, end)
            if last_sep > start:
                end = last_sep + len(separator)

        chunk = document[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap if end < doc_length else end

    return chunks


# ---------------------------------------------------------------------------
# Document ID generation
# ---------------------------------------------------------------------------


def generate_doc_id(content: str) -> str:
    """Generate a deterministic document ID from *content*."""
    return deterministic_hash(content)


# ---------------------------------------------------------------------------
# Document deduplication
# ---------------------------------------------------------------------------


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Remove duplicate documents based on page content hash.

    Preserves the order of first occurrence.
    """
    seen: set = set()
    unique: List[Document] = []
    for doc in documents:
        key = deterministic_hash(doc.page_content)
        if key not in seen:
            seen.add(key)
            unique.append(doc)
    return unique
