"""
Conditional edge routing for the SentinelRAG LangGraph workflow.

Each function takes the current ``AgentState`` and returns the **name** of
the next node to execute.
"""

from src.state import AgentState
from src.config import settings
import logging

logger = logging.getLogger(__name__)


def route_post_grading(state: AgentState) -> str:
    """Decide the next step after document grading.

    Returns:
        ``"generate"`` if valid context is found or the loop limit is reached,
        otherwise ``"rewrite"`` to trigger query rewriting.
    """
    if state["loop_count"] >= settings.max_loop_count:
        logger.warning(
            "Max pipeline loops (%d) reached. Proceeding to generation with best available context.",
            settings.max_loop_count,
        )
        return "generate"

    if state.get("web_search", False):
        logger.info("Insufficient context — routing to Rewrite pipeline.")
        return "rewrite"

    logger.info("Context validated. Routing to Generation pipeline.")
    return "generate"


def route_post_reranking(state: AgentState) -> str:
    """Decide the next step after document reranking.

    Currently always proceeds to generation, but provides a hook for future
    conditional logic (e.g., re-retrieve if reranking scores are too low).

    Returns:
        ``"generate"``.
    """
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
