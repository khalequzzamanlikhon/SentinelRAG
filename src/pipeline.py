"""
LangGraph StateGraph assembly for the SentinelRAG agentic RAG pipeline.

Defines the complete workflow graph: nodes (retrieve → grade → rerank →
generate → detect → cite → finalize) and the conditional edges that
route execution based on evaluation outcomes.

NEW in v2.1:
  - web_search node for Tavily fallback when local retrieval fails
  - Re-route web_search results through grading for validation
  - Async workflow compilation support
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

    Workflow:
        retrieve → grade_documents → {rewrite → retrieve, web_search, generate}
        web_search → grade_documents  (validate web results too)
        rewrite_query → retrieve
        rerank_documents → generate → extract_citations → END

    Args:
        vectorstore: An initialised vector store instance (Qdrant or
                     compatible) used for document retrieval.

    Returns:
        A compiled ``StateGraph`` that can be invoked with
        ``workflow.invoke(initial_state)``.
    """
    workflow = StateGraph(AgentState)

    # ── Nodes ──────────────────────────────────────────────────────
    def _retrieve_wrapper(state: AgentState) -> dict:
        return nodes.retrieve_node(state, vectorstore)

    workflow.add_node("retrieve", _retrieve_wrapper)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("rerank_documents", nodes.rerank_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("extract_citations", nodes.extract_citations_node)

    # ── Edges ──────────────────────────────────────────────────────
    workflow.set_entry_point("retrieve")

    # Retrieve → Grade
    workflow.add_edge("retrieve", "grade_documents")

    # Grade → {Rewrite, Web Search, Rerank}
    workflow.add_conditional_edges(
        "grade_documents",
        route_post_grading,
        {
            "rewrite": "rewrite_query",
            "web_search": "web_search",
            "generate": "rerank_documents",
        },
    )

    # Rewrite → Retrieve (loop back)
    workflow.add_edge("rewrite_query", "retrieve")

    # Web Search → Grade (validate web results)
    workflow.add_edge("web_search", "grade_documents")

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
    logger.info(
        "Agentic RAG workflow compiled — 7 nodes + 7 edges (v2.1 with web search)."
    )
    return compiled


def assemble_agentic_rag_workflow_async(vectorstore: Any):
    """Compile the workflow with async checkpointing for streaming generation.

    Uses LangGraph's built-in async support. The returned graph supports
    ``.astream()`` for token-level streaming from ``generate_node_stream``.

    Args:
        vectorstore: An initialised vector store instance.

    Returns:
        A compiled ``StateGraph`` that supports ``.astream()`` with async nodes.
    """
    from langgraph.checkpoint.memory import MemorySaver

    workflow = StateGraph(AgentState)

    def _retrieve_wrapper(state: AgentState) -> dict:
        return nodes.retrieve_node(state, vectorstore)

    workflow.add_node("retrieve", _retrieve_wrapper)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("rerank_documents", nodes.rerank_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("generate", nodes.generate_node_stream)
    workflow.add_node("extract_citations", nodes.extract_citations_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        route_post_grading,
        {
            "rewrite": "rewrite_query",
            "web_search": "web_search",
            "generate": "rerank_documents",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_search", "grade_documents")

    workflow.add_conditional_edges(
        "rerank_documents",
        route_post_reranking,
        {"generate": "generate"},
    )

    workflow.add_edge("generate", "extract_citations")
    workflow.add_edge("extract_citations", END)

    # Use MemorySaver for async checkpointing support
    memory = MemorySaver()
    compiled = workflow.compile(checkpointer=memory)
    logger.info("Async agentic RAG workflow compiled with streaming generation + checkpointing.")
    return compiled
