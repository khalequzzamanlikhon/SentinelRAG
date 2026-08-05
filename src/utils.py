"""
Utility functions for the SentinelRAG pipeline.

Provides:
- Deterministic hashing for cache keys
- LRU cache with TTL for LLM calls
- Singleton SentenceTransformer client (local embeddings)
- Cross-encoder reranker (NEW — BGE-reranker-v2-m3)
- BM25 sparse retrieval engine (NEW)
- Retry decorator for API resilience
- Document chunking and similarity computation
- Rate limiter for API protection (NEW)
"""

import hashlib
import time
import logging
import functools
import threading
from typing import Any, Callable, Dict, Optional, List
from collections import OrderedDict, defaultdict
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
# Cross-encoder reranker (NEW — replaces LLM-based reranking)
# ---------------------------------------------------------------------------

_reranker_model = None
_reranker_lock = Lock()


def get_reranker():
    """Return a shared singleton cross-encoder reranker.

    Uses ``BAAI/bge-reranker-v2-m3`` by default — a multilingual,
    lightweight cross-encoder that produces accurate relevance scores.
    Downloaded once on first use and cached locally.

    Returns:
        A callable ``CrossEncoder`` that accepts (query, document) pairs
        and returns relevance scores in [0, 1].
    """
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    with _reranker_lock:
        if _reranker_model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(
                    "Loading cross-encoder reranker '%s' ...",
                    settings.reranker_model,
                )
                _reranker_model = CrossEncoder(settings.reranker_model)
                logger.info("Cross-encoder loaded successfully.")
            except ImportError:
                logger.warning(
                    "sentence-transformers CrossEncoder not available. "
                    "Install with: pip install sentence-transformers. "
                    "Falling back to similarity-based reranking."
                )
                _reranker_model = None
        return _reranker_model


def rerank_with_cross_encoder(
    query: str,
    documents: List[Document],
    top_k: Optional[int] = None,
) -> List[Document]:
    """Rerank documents using a cross-encoder for precise relevance scoring.

    Falls back to cosine similarity if the cross-encoder is unavailable.

    Args:
        query: The search query.
        documents: Candidate documents to rerank.
        top_k: Number of top documents to return (defaults to settings.reranker_top_k).

    Returns:
        Documents sorted by cross-encoder score (descending), each with
        ``rerank_score`` in metadata.
    """
    if not documents:
        return []

    top_k = top_k or settings.reranker_top_k
    reranker = get_reranker()

    if reranker is not None:
        # Use cross-encoder
        pairs = [(query, doc.page_content[:2000]) for doc in documents]
        scores = reranker.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_method"] = "cross_encoder"

        scored = sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored][:top_k]
    else:
        # Fallback: cosine similarity
        logger.info("Cross-encoder unavailable — using cosine similarity for reranking.")
        return _rerank_with_similarity(query, documents, top_k)


def _rerank_with_similarity(
    query: str,
    documents: List[Document],
    top_k: int,
) -> List[Document]:
    """Fallback reranker using embedding cosine similarity."""
    client = get_embeddings_client()
    query_emb = np.array(client.embed_query(query)).reshape(1, -1)

    scored: list = []
    for doc in documents:
        try:
            doc_emb = np.array(client.embed_query(doc.page_content[:1000])).reshape(1, -1)
            score = float(cosine_similarity(query_emb, doc_emb)[0][0])
        except Exception:
            score = doc.metadata.get("grading", {}).get("relevance_score", 0.0)

        doc.metadata["rerank_score"] = score
        doc.metadata["rerank_method"] = "cosine_similarity"
        scored.append((score, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored][:top_k]


# ---------------------------------------------------------------------------
# BM25 Sparse Retrieval Engine (NEW)
# ---------------------------------------------------------------------------

class BM25Retriever:
    """BM25 sparse retrieval engine for hybrid search.

    Indexes a corpus of documents and retrieves the top-k matches for a
    query using the Okapi BM25 algorithm. Designed to complement dense
    vector search for improved keyword-matching recall.

    Usage::

        bm25 = BM25Retriever()
        bm25.index(corpus_documents)
        results = bm25.search("query text", k=5)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._corpus: List[Document] = []
        self._doc_freqs: Dict[str, int] = {}
        self._doc_lengths: List[int] = []
        self._avgdl: float = 0.0
        self._idf_cache: Dict[str, float] = {}
        self._indexed: bool = False
        self._lock = Lock()

    def _tokenize(self, text: str) -> List[str]:
        """Basic whitespace tokenizer with lowercasing."""
        return text.lower().split()

    def index(self, documents: List[Document]) -> None:
        """Build the BM25 index from a corpus of documents.

        Args:
            documents: List of LangChain Documents to index.
        """
        with self._lock:
            self._corpus = documents
            self._doc_lengths = []
            self._doc_freqs = defaultdict(int)
            self._idf_cache = {}

            for doc in documents:
                tokens = self._tokenize(doc.page_content)
                self._doc_lengths.append(len(tokens))
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self._doc_freqs[token] += 1

            self._avgdl = (
                sum(self._doc_lengths) / len(self._doc_lengths)
                if self._doc_lengths
                else 0.0
            )

            # Precompute IDF values
            N = len(documents)
            for token, df in self._doc_freqs.items():
                self._idf_cache[token] = np.log((N - df + 0.5) / (df + 0.5) + 1.0)

            self._indexed = True
            logger.info(
                "BM25 index built: %d docs, avg length %.1f, %d unique terms.",
                N,
                self._avgdl,
                len(self._doc_freqs),
            )

    def search(self, query: str, k: int = 5) -> List[Document]:
        """Search the BM25 index for the top-k matching documents.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            Top-k Documents sorted by BM25 score (descending).
        """
        if not self._indexed:
            logger.warning("BM25 index not built — returning empty results.")
            return []

        query_tokens = self._tokenize(query)
        scores: List[float] = []

        for i, doc in enumerate(self._corpus):
            doc_tokens = self._tokenize(doc.page_content)
            doc_len = self._doc_lengths[i]
            term_freqs: Dict[str, int] = {}
            for t in doc_tokens:
                term_freqs[t] = term_freqs.get(t, 0) + 1

            score = 0.0
            for token in set(query_tokens):
                if token not in self._idf_cache:
                    continue
                tf = term_freqs.get(token, 0)
                if tf == 0:
                    continue
                idf = self._idf_cache[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self._avgdl, 1))
                score += idf * numerator / denominator

            scores.append(score)

        scored_pairs = sorted(
            zip(scores, self._corpus), key=lambda x: x[0], reverse=True
        )
        results = []
        for score, doc in scored_pairs[:k]:
            if score > 0:
                new_doc = Document(
                    page_content=doc.page_content,
                    metadata={**doc.metadata, "bm25_score": float(score)},
                )
                results.append(new_doc)

        return results


_bm25_retriever: Optional[BM25Retriever] = None
_bm25_lock = Lock()


def get_bm25_retriever() -> BM25Retriever:
    """Return a shared singleton BM25Retriever instance."""
    global _bm25_retriever
    if _bm25_retriever is None:
        with _bm25_lock:
            if _bm25_retriever is None:
                _bm25_retriever = BM25Retriever(
                    k1=settings.bm25_k1,
                    b=settings.bm25_b,
                )
    return _bm25_retriever


def merge_hybrid_results(
    dense_docs: List[Document],
    bm25_docs: List[Document],
    dense_weight: float = 0.7,
    bm25_weight: float = 0.3,
    top_k: int = 10,
) -> List[Document]:
    """Merge dense and BM25 results using weighted reciprocal rank fusion.

    Args:
        dense_docs: Documents from vector search (with similarity_score).
        bm25_docs: Documents from BM25 search (with bm25_score).
        dense_weight: Weight for dense scores.
        bm25_weight: Weight for BM25 scores.
        top_k: Maximum number of merged documents to return.

    Returns:
        Merged and re-ranked list of Documents.
    """
    k = 60.0  # RRF constant
    score_map: Dict[str, tuple] = {}  # doc_id -> (accumulated_score, Document)

    for rank, doc in enumerate(dense_docs):
        doc_id = deterministic_hash(doc.page_content)
        rrf_score = dense_weight / (k + rank + 1)
        score_map[doc_id] = (rrf_score, doc)

    for rank, doc in enumerate(bm25_docs):
        doc_id = deterministic_hash(doc.page_content)
        rrf_score = bm25_weight / (k + rank + 1)
        if doc_id in score_map:
            existing_score, _ = score_map[doc_id]
            score_map[doc_id] = (existing_score + rrf_score, doc)
        else:
            score_map[doc_id] = (rrf_score, doc)

    merged = sorted(score_map.values(), key=lambda x: x[0], reverse=True)
    result = []
    for score, doc in merged[:top_k]:
        doc.metadata["hybrid_score"] = float(score)
        doc.metadata["search_source"] = "hybrid"
        result.append(doc)

    return result


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
            self._store.move_to_end(key)
            return entry["result"]

    def set(self, key: str, value: Any) -> None:
        """Store *value* under *key*, evicting LRU items if over capacity."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = {"result": value, "timestamp": time.time()}
            while len(self._store) > self._maxsize:
                self._store.popitem(last=False)

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
        params: Keyword arguments forwarded to *func* via ``func(**params)``.
        cache_key: The cache lookup key.
        ttl: Override TTL in seconds (defaults to ``settings.cache_ttl_seconds``).

    Returns:
        The cached or freshly-computed result.
    """
    if not settings.enable_caching:
        return func(params)

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
# Rate limiter (NEW)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple sliding-window rate limiter for API protection."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self._max = max_requests
        self._window = window_seconds
        self._timestamps: List[float] = []
        self._lock = Lock()

    def acquire(self) -> bool:
        """Try to acquire a request slot. Returns True if allowed."""
        now = time.time()
        with self._lock:
            cutoff = now - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) < self._max:
                self._timestamps.append(now)
                return True
            return False

    @property
    def remaining(self) -> int:
        with self._lock:
            cutoff = time.time() - self._window
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return max(0, self._max - len(self._timestamps))


_rate_limiter = RateLimiter(
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window_seconds,
)


def check_rate_limit() -> bool:
    """Check if the current request is within rate limits. Returns True if allowed."""
    return _rate_limiter.acquire()


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
                            "Attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
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
