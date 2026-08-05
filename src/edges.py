"""
Conditional edge routing for the SentinelRAG LangGraph workflow.

Each function takes the current ``AgentState`` and returns the **name** of
the next node to execute.

NEW in v2.1:
  - route_post_grading now supports web_search → web_search node
  - route_post_reranking: optional re-retrieve if scores are too low
"""

from src.state import AgentState
from src.config import settings
import logging

logger = logging.getLogger(__name__)


def route_post_grading(state: AgentState) -> str:
    """Decide the next step after document grading.

    Returns:
        ``"generate"`` (via rerank) if valid context is found,
        ``"web_search"`` if all docs dropped, web search is enabled,
        and web search hasn't been attempted yet,
        ``"rewrite"`` if all docs dropped and web search is disabled
        or has already been attempted,
        ``"generate"`` if max loops reached (force proceed).
    """
    if state["loop_count"] >= settings.max_loop_count:
        logger.warning(
            "Max pipeline loops (%d) reached. Proceeding to generation with best available context.",
            settings.max_loop_count,
        )
        return "generate"

    if state.get("web_search", False):
        web_already_attempted = state.get("metadata", {}).get("web_search_attempted", False)
        if settings.enable_web_search and not web_already_attempted:
            logger.info("Insufficient local context — routing to Web Search.")
            return "web_search"
        else:
            logger.info(
                "Insufficient context — routing to Rewrite pipeline (web search %s).",
                "already attempted" if web_already_attempted else "disabled",
            )
            return "rewrite"

    logger.info("Context validated. Routing to Generation pipeline.")
    return "generate"


def route_post_reranking(state: AgentState) -> str:
    """Decide the next step after document reranking.

    Checks if reranking scores are high enough; if all scores are below
    threshold, could trigger a rewrite. Currently always proceeds to
    generation as a safe default.

    Returns:
        ``"generate"``.
    """
    reranked = state.get("reranked_documents", [])

    if reranked:
        avg_score = sum(
            d.metadata.get("rerank_score", 0.0) for d in reranked
        ) / len(reranked)
        if avg_score < settings.similarity_threshold / 2:
            logger.warning(
                "Reranking scores very low (avg=%.2f) — consider re-retrieval.",
                avg_score,
            )

    logger.info("Reranking complete — routing to Generation pipeline.")
    return "generate"


def route_post_generation(state: AgentState) -> str:
    """Decide the next step after response generation.

    Always returns ``"finalize"`` in the current implementation. Logs warnings
    when grounding quality is poor.

    Returns:
        ``"finalize"``.
    """
    logger.info("--- EDGE: POST-GENERATION ROUTING ---")
    documents = state.get("documents", [])
    generation_metrics = state.get("generation_metrics", {})

    if documents and generation_metrics:
        grounded_score = generation_metrics.get("grounded_score", 0.0)
        hallucination_count = len(generation_metrics.get("hallucinated_claims", []))

        if grounded_score < 0.5 or hallucination_count > 2:
            logger.warning(
                "Generation quality concern — grounded_score=%.2f, hallucination_count=%d. "
                "Consider improving context or retrying with stricter guidelines.",
                grounded_score,
                hallucination_count,
            )

    if not documents:
        logger.info("No source documents — skipping grounding check.")

    return "finalize"
