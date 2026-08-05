"""
Tavily web search integration for SentinelRAG.

Provides a fallback search source when the local vector store yields
insufficient or irrelevant results. Uses the Tavily Search API for
real-time web retrieval with structured result formatting.

NEW in v2.1.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.config import settings
from src.exceptions import WebSearchError

logger = logging.getLogger(__name__)

# Module-level Tavily client (lazy init)
_tavily_client = None


def _get_tavily_client():
    """Return a singleton Tavily client, initialising on first use."""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    if not settings.tavily_api_key or settings.tavily_api_key.startswith("your_"):
        raise WebSearchError(
            "TAVILY_API_KEY not configured. Get a free key at https://tavily.com",
            details={"source": "tavily"},
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        raise WebSearchError(
            "Tavily package not installed. Run: pip install tavily-python",
            details={"source": "tavily", "missing_package": "tavily-python"},
        )

    _tavily_client = TavilyClient(api_key=settings.tavily_api_key)
    logger.info("Tavily web search client initialised.")
    return _tavily_client


def search_web(query: str, max_results: Optional[int] = None) -> List[Document]:
    """Perform a web search via Tavily and return LangChain Documents.

    Args:
        query: The search query string.
        max_results: Maximum results to return (defaults to settings).

    Returns:
        List of Documents with page_content from web snippets and
        metadata including url, title, and relevance score.

    Raises:
        WebSearchError: If the Tavily API is unreachable or keys are missing.
    """
    if not settings.enable_web_search:
        logger.info("Web search disabled — returning empty results.")
        return []

    max_results = max_results or settings.tavily_max_results

    try:
        client = _get_tavily_client()
        response = client.search(
            query=query,
            search_depth=settings.tavily_search_depth,
            max_results=max_results,
            include_answer=True,
            include_raw_content=False,
        )

        documents: List[Document] = []

        # Include Tavily's generated answer if available
        if response.get("answer"):
            documents.append(
                Document(
                    page_content=response["answer"],
                    metadata={
                        "source": "tavily_answer",
                        "search_query": query,
                        "relevance_score": 1.0,
                        "search_source": "web",
                    },
                )
            )

        # Include individual search results
        for result in response.get("results", []):
            content = result.get("content", "")
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": result.get("url", "unknown"),
                        "title": result.get("title", ""),
                        "relevance_score": result.get("score", 0.5),
                        "search_source": "web",
                        "search_query": query,
                    },
                )
            )

        logger.info("Web search returned %d results for query: '%s'", len(documents), query)
        return documents

    except WebSearchError:
        raise
    except Exception as e:
        logger.exception("Unexpected error during web search")
        raise WebSearchError(
            f"Web search failed: {e}",
            details={"source": "tavily", "error": str(e)},
        )


def search_web_safe(query: str, max_results: Optional[int] = None) -> List[Document]:
    """Safe wrapper around search_web that never raises.

    Returns an empty list on failure instead of propagating exceptions.

    Args:
        query: The search query string.
        max_results: Maximum results to return.

    Returns:
        List of Documents, or empty list on error.
    """
    try:
        return search_web(query, max_results)
    except WebSearchError as e:
        logger.warning("Web search unavailable: %s", e)
        return []
    except Exception as e:
        logger.warning("Web search failed unexpectedly: %s", e)
        return []
