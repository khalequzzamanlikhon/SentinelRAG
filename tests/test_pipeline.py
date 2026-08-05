"""
Unit tests for SentinelRAG pipeline assembly (v2.1).

Tests cover:
  - assemble_agentic_rag_workflow compiles without error (7 nodes now).
  - The compiled graph contains the expected node names.
  - NEW: web_search_node is included.

All external dependencies are mocked so these tests run offline.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")

from src.pipeline import assemble_agentic_rag_workflow, assemble_agentic_rag_workflow_async


class TestPipelineAssembly(unittest.TestCase):
    """Tests for assemble_agentic_rag_workflow(vectorstore)."""

    @patch("src.nodes.settings")
    @patch("src.edges.settings")
    @patch("src.pipeline.nodes")
    def test_workflow_compiles_successfully(
        self, mock_nodes_module, mock_edges_cfg, mock_nodes_cfg
    ):
        """Passing a mock vectorstore should produce a compiled graph without errors."""
        for cfg in (mock_edges_cfg, mock_nodes_cfg):
            cfg.max_loop_count = 3

        # Mock all 7 node functions (added web_search_node)
        for name in (
            "retrieve_node",
            "grade_documents_node",
            "rerank_documents_node",
            "rewrite_query_node",
            "web_search_node",
            "generate_node",
            "extract_citations_node",
        ):
            setattr(mock_nodes_module, name, MagicMock())

        mock_vectorstore = MagicMock()
        compiled = assemble_agentic_rag_workflow(mock_vectorstore)

        self.assertIsNotNone(compiled, "Compiled workflow should not be None")

    @patch("src.nodes.settings")
    @patch("src.edges.settings")
    @patch("src.pipeline.nodes")
    def test_workflow_has_expected_nodes(
        self, mock_nodes_module, mock_edges_cfg, mock_nodes_cfg
    ):
        """The compiled graph must include all 7 pipeline nodes (v2.1)."""
        for cfg in (mock_edges_cfg, mock_nodes_cfg):
            cfg.max_loop_count = 3

        for name in (
            "retrieve_node",
            "grade_documents_node",
            "rerank_documents_node",
            "rewrite_query_node",
            "web_search_node",
            "generate_node",
            "extract_citations_node",
        ):
            setattr(mock_nodes_module, name, MagicMock())

        mock_vectorstore = MagicMock()
        compiled = assemble_agentic_rag_workflow(mock_vectorstore)

        expected_nodes = {
            "retrieve",
            "grade_documents",
            "rerank_documents",
            "rewrite_query",
            "web_search",
            "generate",
            "extract_citations",
        }

        graph_nodes = set()
        if hasattr(compiled, "nodes"):
            graph_nodes = (
                set(compiled.nodes.keys())
                if isinstance(compiled.nodes, dict)
                else set(compiled.nodes)
            )
        elif hasattr(compiled, "get_graph"):
            graph = compiled.get_graph()
            if hasattr(graph, "nodes"):
                graph_nodes = (
                    set(graph.nodes.keys())
                    if isinstance(graph.nodes, dict)
                    else set(graph.nodes)
                )

        user_nodes = {n for n in graph_nodes if not n.startswith("__")}

        self.assertTrue(
            expected_nodes.issubset(user_nodes),
            f"Missing nodes: {expected_nodes - user_nodes}. Found: {user_nodes}",
        )

    def test_async_workflow_compiles(self):
        """assemble_agentic_rag_workflow_async should also produce a valid graph."""
        with patch("src.pipeline.assemble_agentic_rag_workflow") as mock_assemble:
            mock_assemble.return_value = MagicMock()
            result = assemble_agentic_rag_workflow_async(MagicMock())
            self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
