"""
Document ingestion pipeline for SentinelRAG.

Parses PDF files from ``data/raw/``, chunks the text, generates embeddings
via local sentence-transformers, stores vectors in a Qdrant collection,
and builds the BM25 sparse index for hybrid search.

Two operating modes:
  - **embedded**  (default) — local on-disk Qdrant, zero infrastructure.
  - **server**               — connects to a standalone Qdrant instance (Docker).
"""

import os
import glob
import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import settings
from src.utils import chunk_document, get_embeddings_client, get_bm25_retriever

logger = logging.getLogger(__name__)


def _check_collection_exists() -> bool:
    """Return True if the Qdrant collection already exists and has vectors.

    Creates a throw-away client to probe — the client is released
    (lock freed) before any subsequent client creation.
    """
    try:
        if settings.qdrant_mode == "server":
            client = QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port
            )
        else:
            client = QdrantClient(path=settings.qdrant_persist_path)

        info = client.get_collection(settings.qdrant_collection_name)
        count = info.points_count if hasattr(info, "points_count") else 0
        del client
        return count > 0
    except Exception:
        return False


def _build_vectorstore(embeddings) -> QdrantVectorStore:
    """Create or connect to a Qdrant vector store based on ``settings.qdrant_mode``.

    Assumes the collection **already exists** — call ``ingest_data()`` first
    if it may not.
    """
    if settings.qdrant_mode == "server":
        logger.info(
            "Connecting to Qdrant server at %s:%s",
            settings.qdrant_host,
            settings.qdrant_port,
        )
        client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    else:
        logger.info("Connecting to embedded Qdrant at %s", settings.qdrant_persist_path)
        client = QdrantClient(path=settings.qdrant_persist_path)

    return QdrantVectorStore(
        client=client,
        collection_name=settings.qdrant_collection_name,
        embedding=embeddings,
        validate_collection_config=False,  # already checked by caller
    )


def _parse_pdfs(data_dir: str) -> list:
    """Parse all PDFs in *data_dir* into a flat list of text chunks (``Document``)."""
    pdf_files = glob.glob(os.path.join(data_dir, "*.pdf"))
    if not pdf_files:
        raise ValueError(
            f"No PDF files found in '{data_dir}/'. "
            f"Please add at least one PDF document."
        )

    all_chunks: list = []
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        logger.info("Parsing %s ...", fname)

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required for PDF parsing. Run: pip install PyMuPDF")

        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()

        chunks = chunk_document(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "source": fname,
                        "chunk_id": i,
                        "doc_id": f"{fname}_chunk_{i}",
                    },
                )
            )

    logger.info("Parsed %d chunks from %d PDF(s).", len(all_chunks), len(pdf_files))
    return all_chunks


def _client_options() -> dict:
    """Return the QdrantClient keyword args for the current mode."""
    if settings.qdrant_mode == "server":
        return {"host": settings.qdrant_host, "port": settings.qdrant_port}
    return {"path": settings.qdrant_persist_path}


def ingest_data(data_dir: str = "data/raw", force_reingest: bool = False) -> QdrantVectorStore:
    """Ingest PDF documents into the Qdrant vector store.

    Uses a **single** client internally: ``QdrantVectorStore.from_documents``
    is called (which creates its own client) — the earlier ``_build_vectorstore``
    is only used when the collection already exists and no ingestion is needed.
    This avoids the QdrantLocal \"already locked\" error.

    Args:
        data_dir: Directory containing PDF files to ingest.
        force_reingest: If ``True``, drop the existing collection and re-ingest
                        all documents. Otherwise skip if the collection already
                        has data.

    Returns:
        The initialised ``QdrantVectorStore`` instance (ready for retrieval).
    """
    embeddings = get_embeddings_client()
    has_data = _check_collection_exists()

    if has_data and not force_reingest:
        logger.info(
            "Collection '%s' already has vectors — skipping ingestion.",
            settings.qdrant_collection_name,
        )
        logger.info("  To force re-ingestion, pass force_reingest=True.")
        # Rebuild BM25 index (fast — parses PDFs, no re-embedding needed)
        if settings.enable_hybrid_search:
            try:
                all_chunks = _parse_pdfs(data_dir)
                bm25 = get_bm25_retriever()
                if not bm25._indexed:
                    logger.info("Rebuilding BM25 sparse index...")
                    bm25.index(all_chunks)
                    logger.info("BM25 index rebuilt (%d chunks).", len(all_chunks))
            except Exception as e:
                logger.warning("BM25 rebuild skipped: %s", e)
        return _build_vectorstore(embeddings)

    # ── Parse, chunk, embed, and store ─────────────────────────────
    logger.info("Creating fresh collection '%s' ...", settings.qdrant_collection_name)

    all_chunks = _parse_pdfs(data_dir)
    logger.info("Embedding and storing %d chunks via OpenAI ...", len(all_chunks))

    # ``from_documents`` creates its own QdrantClient internally, handling
    # collection creation and data insertion in a single step.
    vectorstore = QdrantVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        **_client_options(),
        collection_name=settings.qdrant_collection_name,
        force_recreate=force_reingest,
        validate_collection_config=False,
    )

    logger.info("Ingestion complete — %d vectors stored.", len(all_chunks))

    # Build BM25 sparse index for hybrid search
    if settings.enable_hybrid_search:
        logger.info("Building BM25 sparse index for hybrid search...")
        try:
            bm25 = get_bm25_retriever()
            bm25.index(all_chunks)
            logger.info("BM25 index built successfully.")
        except Exception as e:
            logger.warning("BM25 index build failed (hybrid search will use dense only): %s", e)

    return vectorstore
