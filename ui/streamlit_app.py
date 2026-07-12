"""
SentinelRAG — Streamlit Chat Interface.

Provides:
  - Chat UI with conversation memory across turns
  - Streaming step-by-step agent progress display
  - Expandable audit trail with full pipeline metrics
  - Structured citation display
  - User feedback (thumbs up/down) on each response
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional

# ── Ensure the project root is on sys.path ──
# Streamlit may not include the working directory, which prevents
# "from src import ..." from resolving. This works regardless of
# how the app was launched (streamlit run, python -m, etc.)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.config import settings
from src.utils import get_embeddings_client, clear_cache
from ingestion.load_documents import ingest_data
from src.pipeline import assemble_agentic_rag_workflow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelRAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS tweaks
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stChatInput { position: fixed; bottom: 1rem; width: 60%; }
    .block-container { padding-bottom: 6rem; }
    div[data-testid="stExpander"] details summary p {
        font-size: 0.9rem;
        font-weight: 600;
    }
    .citation-box {
        background: #f0f2f6;
        border-left: 4px solid #4CAF50;
        padding: 0.5rem 1rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.85rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# System initialisation (cached)
# ---------------------------------------------------------------------------


@st.cache_resource
def _init_system():
    """Initialise vector store, ingest data, and compile the LangGraph workflow."""
    logger.info("Initialising SentinelRAG system ...")

    # Configure logging from settings
    settings.configure_logging()

    # Ingest or connect — handles both embedded and server modes internally.
    vectorstore = ingest_data()

    workflow = assemble_agentic_rag_workflow(vectorstore)
    logger.info("System initialisation complete.")
    return workflow


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state():
    """Ensure all session state keys exist with defaults."""
    defaults = {
        "workflow": None,
        "messages": [],
        "conversation_history": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def _add_message(role: str, content: str, audit: Optional[dict] = None):
    """Append a message dict to the chat history."""
    msg = {"role": role, "content": content}
    if audit:
        msg["audit"] = audit
    st.session_state.messages.append(msg)


# ---------------------------------------------------------------------------
# Audit trail helper
# ---------------------------------------------------------------------------


def _render_audit_trail(audit: dict) -> None:
    """Render an expandable audit trail JSON section."""
    with st.expander("🔍 View Agent Audit Trail"):
        # Show key metrics as columns first
        cols = st.columns(4)
        metrics = [
            ("Loops", audit.get("Pipeline Loops", "—")),
            ("Time", f"{audit.get('Execution Time (sec)', 0):.1f}s"),
            ("Citations", audit.get("Citation Count", "—")),
            ("Strategy", str(audit.get("Query Strategy Used", "—"))),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)

        st.divider()

        # Full audit JSON
        st.json(audit)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/shield.png", width=48)
    st.markdown("### 🛡️ SentinelRAG")
    st.caption("Self-Correcting Enterprise AI Agent")

    st.divider()

    st.markdown("**System Status**")
    if st.session_state.get("workflow"):
        st.success("✅ System Armed & Ready")
    else:
        st.warning("⏳ Initialising ...")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()

    if st.button("🔄 Force Re-Ingest PDFs", use_container_width=True):
        clear_cache()
        st.cache_resource.clear()
        st.success("Cache cleared. Reload to re-ingest.")
        st.rerun()

    st.divider()
    st.markdown("**Configuration**")
    st.code(
        f"""
Model: {settings.llm_model}
Embeddings: {settings.embedding_model}
Qdrant: {settings.qdrant_mode}
Top-K: {settings.top_k_documents}
Max Loops: {settings.max_loop_count}
Threshold: {settings.similarity_threshold}
Caching: {'On' if settings.enable_caching else 'Off'}
    """.strip()
    )

    st.divider()
    st.caption(
        "SentinelRAG v2.0 — "
        "Retrieve · Validate · Correct · Generate"
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("🛡️ SentinelRAG")
st.caption(
    "Ask questions about your HR documents. The agent retrieves, grades, "
    "reranks, generates, and audits its own responses."
)

_init_session_state()

# --- Bootstrap the workflow on first load ---
if st.session_state.workflow is None:
    with st.spinner("🛡️ Arming SentinelRAG system ..."):
        try:
            st.session_state.workflow = _init_system()
            st.success("System armed! Ask a question below.")
        except Exception as exc:
            st.error(f"Initialisation failed: {exc}")
            st.info(
                "Ensure your API keys are set in a `.env` file and PDFs "
                "are placed in `data/raw/`."
            )
            st.stop()

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Audit trail expander (assistant messages only)
        if msg.get("audit") and msg["role"] == "assistant":
            _render_audit_trail(msg["audit"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about your documents ..."):
    # 1. User message
    _add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Build initial state
    initial_state = {
        "question": prompt,
        "current_query": prompt,
        "documents": [],
        "reranked_documents": [],
        "generation": None,
        "citations": [],
        "web_search": False,
        "query_strategy": None,
        "loop_count": 0,
        "retrieval_metrics": {},
        "generation_metrics": {},
        "error": None,
        "metadata": {},
        "conversation_history": st.session_state.conversation_history[-10:],  # sliding window
    }

    # 3. Run the pipeline
    with st.chat_message("assistant"):
        start_time = time.time()
        final_state = None

        # Simulated progress display
        progress_placeholder = st.empty()
        response_placeholder = st.empty()

        steps = [
            ("🔎 Retrieving documents ...", "retrieve"),
            ("📋 Grading relevance ...", "grade_documents"),
            ("📊 Reranking context ...", "rerank_documents"),
            ("⚡ Generating response ...", "generate"),
            ("🛡️ Extracting citations ...", "extract_citations"),
        ]
        step_idx = 0

        with st.spinner("Agent is thinking ..."):
            try:
                # Run the pipeline ONCE — stream() yields every node's output
                for step in st.session_state.workflow.stream(
                    initial_state, {"recursion_limit": 25}
                ):
                    # Update progress display based on step count
                    if step_idx < len(steps):
                        progress_placeholder.info(steps[step_idx][0])
                        step_idx += 1

                    # Merge every step's output into initial_state
                    for node_name, state_update in step.items():
                        if isinstance(state_update, dict):
                            initial_state.update(state_update)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")

            progress_placeholder.empty()

        exec_time = time.time() - start_time

        # 4. Extract results from the accumulated state
        response_text = initial_state.get(
            "generation",
            "No response generated.",
        )
        if initial_state.get("error"):
            response_text = f"⚠️ System Error: {initial_state['error']}"

        st.markdown(response_text)

        # 5. Citations display
        citations = initial_state.get("citations", [])
        if citations:
            with st.expander("📚 View Source Citations"):
                for i, cit in enumerate(citations, 1):
                    st.markdown(
                        f'<div class="citation-box">'
                        f"<strong>{i}. Claim:</strong> {cit.get('claim', '')}<br>"
                        f"<strong>Source:</strong> {cit.get('source_document', 'unknown')}<br>"
                        f"<strong>Excerpt:</strong> “{cit.get('source_excerpt', '')}”<br>"
                        f"<strong>Confidence:</strong> {cit.get('confidence', 0.0):.0%}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # 6. Build audit trail
        audit_meta = {}
        if initial_state.get("question"):
            audit_meta = {
                "Original Query": initial_state.get("question", prompt),
                "Rewritten Queries": initial_state.get("metadata", {}).get("query_history", []),
                "Query Strategy Used": str(initial_state.get("query_strategy", "N/A")),
                "Pipeline Loops": initial_state.get("loop_count", 0),
                "Retrieval Metrics": initial_state.get("retrieval_metrics", {}),
                "Generation Metrics": initial_state.get("generation_metrics", {}),
                "Citation Count": len(citations),
                "Execution Time (sec)": round(exec_time, 2),
            }

        if audit_meta:
            _render_audit_trail(audit_meta)

        # 7. Feedback buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("👍", key=f"thumbs_up_{len(st.session_state.messages)}"):
                st.toast("Thanks for the positive feedback! 🎉", icon="👍")
        with col2:
            if st.button("👎", key=f"thumbs_down_{len(st.session_state.messages)}"):
                st.toast("Thanks for the feedback — we'll improve!", icon="👎")

        # 8. Save assistant message
        _add_message("assistant", response_text, audit=audit_meta)

        # 9. Update conversation history for next turn
        if initial_state.get("question"):
            st.session_state.conversation_history.append(
                {"role": "user", "content": prompt}
            )
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
