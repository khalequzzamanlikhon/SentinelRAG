"""
LangGraph StateGraph assembly for the SentinelRAG agentic RAG pipeline.

Defines the complete workflow graph: nodes (retrieve → grade → rerank →
generate → detect → cite → finalize) and the conditional edges that
route execution based on evaluation outcomes.
"""

from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.edges import route_post_grading, route_post_reranking, route_post_generation
import src.nodes as nodes
import logging
from typing import Any

logger = logging.getLogger(__name__)


def assemble_agentic_rag_workflow(vectorstore: Any) -> StateGraph:
    """Compile the full agentic RAG workflow graph.

    Args:
        vectorstore: An initialised vector store instance (Qdrant or
                     compatible) used for document retrieval.

    Returns:
        A compiled ``StateGraph`` that can be invoked with
        ``workflow.invoke(initial_state)``.
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────
    # The retrieve_node needs a reference to the vectorstore, so we
    # wrap it with a helper that injects the dependency.
    def _retrieve_wrapper(state: AgentState) -> dict:
        return nodes.retrieve_node(state, vectorstore)

    workflow.add_node("retrieve", _retrieve_wrapper)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("rerank_documents", nodes.rerank_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("extract_citations", nodes.extract_citations_node)

    # ── Edges ──────────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    # Retrieve → Grade
    workflow.add_edge("retrieve", "grade_documents")

    # Grade → {Rewrite, Rerank}
    workflow.add_conditional_edges(
        "grade_documents",
        route_post_grading,
        {
            "rewrite": "rewrite_query",
            "generate": "rerank_documents",  # proceed to reranking before generation
        },
    )

    # Rewrite → Retrieve (loop back)
    workflow.add_edge("rewrite_query", "retrieve")

    # Rerank → Generate
    workflow.add_conditional_edges(
        "rerank_documents",
        route_post_reranking,
        {
            "generate": "generate",
        },
    )

    # Generate → Extract Citations → End
    workflow.add_edge("generate", "extract_citations")
    workflow.add_edge("extract_citations", END)

    compiled = workflow.compile()
    logger.info("Agentic RAG workflow compiled successfully — 6 nodes + 5 edges.")
    return compiled
