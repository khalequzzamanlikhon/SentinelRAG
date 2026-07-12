"""
Document ingestion pipeline for SentinelRAG.

Handles PDF parsing, chunking, embedding, and vectorstore storage.
Supports both embedded (local) and server (Docker) Qdrant modes.
"""

from ingestion.load_documents import ingest_data

__all__ = ["ingest_data"]
