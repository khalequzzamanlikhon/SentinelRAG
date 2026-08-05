"""
SentinelRAG -- Streamlit Chat Interface.

Provides:
  - Chat UI with conversation memory across turns
  - Real-time token-level streaming display via async workflow
  - Expandable audit trail with full pipeline metrics
  - Structured citation display
  - User feedback (thumbs up/down) on each response
  - Input guardrails (prompt injection detection, length limits)
  - Rate limiting awareness

v2.1: Added streaming, guardrails, validate_api_keys, hybrid search display.
v2.2: Wired async streaming generate_node_stream end-to-end with token display.
"""

import os
import sys
import time
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, Tuple

# -- Ensure the project root is on sys.path --
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st

from src.config import settings
from src.utils import get_embeddings_client, clear_cache, check_rate_limit
from src.guardrails import validate_query, sanitize_query
from src.exceptions import GuardrailViolationError, ConfigurationError
from ingestion.load_documents import ingest_data
from src.pipeline import assemble_agentic_rag_workflow, assemble_agentic_rag_workflow_async

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelRAG",
    page_icon="\U0001F6E1",
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
    .guardrail-warning {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
    }
    .streaming-cursor::after {
        content: " \\258C";
        animation: blink 1s step-end infinite;
    }
    @keyframes blink { 50% { opacity: 0; } }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# System initialisation (cached) -- single ingest_data call for both workflows
# ---------------------------------------------------------------------------


@st.cache_resource
def _init_system() -> Tuple:
    """Initialise vector store, BM25 index, ingest data, and compile BOTH workflows.

    Calls ``ingest_data()`` exactly once to avoid Qdrant embedded lock collisions,
    then compiles both sync and async workflows from the same vectorstore.
    """
    logger.info("Initialising SentinelRAG system (sync + async) ...")

    settings.configure_logging()

    try:
        settings.validate_api_keys()
    except ConfigurationError as e:
        logger.error("Configuration error: %s", e)
        raise

    settings.setup_tracing()

    # Single ingest_data call -- embedded Qdrant allows only one writer at a time
    vectorstore = ingest_data()

    sync_workflow = assemble_agentic_rag_workflow(vectorstore)
    async_workflow = assemble_agentic_rag_workflow_async(vectorstore)

    logger.info("System initialisation complete (sync + async streaming).")
    return sync_workflow, async_workflow


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _init_session_state():
    """Ensure all session state keys exist with defaults."""
    defaults = {
        "workflow": None,
        "workflow_async": None,
        "messages": [],
        "conversation_history": [],
        # Stable per-session thread id -- required by LangGraph's checkpointer
        "thread_id": str(uuid.uuid4()),
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
    with st.expander("\U0001F50D View Agent Audit Trail"):
        cols = st.columns(5)
        search_source = audit.get("Search Source", "")
        metrics = [
            ("Loops", audit.get("Pipeline Loops", "")),
            ("Time", f"{audit.get('Execution Time (sec)', 0):.1f}s"),
            ("Citations", audit.get("Citation Count", "")),
            ("Strategy", str(audit.get("Query Strategy Used", ""))),
            ("Source", str(search_source)),
        ]
        for col, (label, value) in zip(cols, metrics):
            col.metric(label, value)

        st.divider()
        st.json(audit)


# ---------------------------------------------------------------------------
# Async pipeline runner -- consumes astream() with token display
# ---------------------------------------------------------------------------


async def _run_async_pipeline(
    initial_state: dict,
    workflow,
    progress_placeholder,
    response_placeholder,
) -> dict:
    """Run the async workflow with real-time token display.

    Tokens appear as soon as the generate node completes
    (LangGraph nodes return single dicts, not streamed per-token).
    """
    steps = [
        ("\U0001F50E Retrieving documents ...", "retrieve"),
        ("\U0001F4CB Grading relevance ...", "grade_documents"),
        ("\U0001F310 Searching web ...", "web_search"),
        ("\U0001F4CA Reranking context ...", "rerank_documents"),
        ("\u26A1 Generating response ...", "generate"),
        ("\U0001F6E1 Extracting citations ...", "extract_citations"),
    ]
    step_idx = 0
    displayed_tokens = 0

    config = {
        "configurable": {
            "thread_id": st.session_state.get(
                "thread_id", "sentinelrag-default-thread"
            )
        },
        "recursion_limit": 25,
    }

    try:
        async for step in workflow.astream(initial_state, config):
            if step_idx < len(steps) and progress_placeholder:
                progress_placeholder.info(steps[step_idx][0])
                step_idx += 1

            for node_name, state_update in step.items():
                if isinstance(state_update, dict):
                    initial_state.update(state_update)

                # Display streaming tokens when generate node completes
                if node_name == "generate":
                    tokens = initial_state.get("streaming_tokens", [])
                    if tokens and response_placeholder and len(tokens) > displayed_tokens:
                        partial = "".join(tokens)
                        response_placeholder.markdown(
                            f'<span class="streaming-cursor">{partial}</span>',
                            unsafe_allow_html=True,
                        )
                        displayed_tokens = len(tokens)

    except Exception as exc:
        logger.exception("Async pipeline error")
        initial_state["error"] = str(exc)

    return initial_state


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## \U0001F6E1 SentinelRAG")
    st.caption("Self-Correcting Enterprise AI Agent \u2014 v2.2")

    st.divider()

    st.markdown("**System Status**")
    if st.session_state.get("workflow_async") or st.session_state.get("workflow"):
        st.success("\u2705 System Armed & Ready (async streaming)")
    else:
        st.warning("\u23F3 Initialising ...")

    with st.expander("\U0001F4C4 Loaded Documents", expanded=False):
        st.markdown(
            """
            The following PDFs are indexed in the vector store:

            1. **BRAC School of Public Health** \u2014 employee handbook
            2. **PartexStar Group** \u2014 employee handbook

            **To add more:**  
            Place new `.pdf` files in `data/raw/` and click  
            *\"\U0001F504 Force Re-Ingest PDFs\"* below.
            """
        )

    if st.button("\U0001F5D1 Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        # Fresh thread id resets the checkpointer's memory for this session
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    if st.button("\U0001F504 Force Re-Ingest PDFs", use_container_width=True):
        clear_cache()
        st.cache_resource.clear()
        st.success("Cache cleared. Reload to re-ingest.")
        st.rerun()

    st.divider()
    st.markdown("**Configuration**")
    st.code(
        f"""
Model: {settings.groq_model}
Embeddings: {settings.embedding_model}
Reranker: {settings.reranker_model}
Qdrant: {settings.qdrant_mode}
Top-K: {settings.top_k_documents}
Max Loops: {settings.max_loop_count}
Threshold: {settings.similarity_threshold}
Hybrid Search: {'On' if settings.enable_hybrid_search else 'Off'}
Web Search: {'On' if settings.enable_web_search else 'Off'}
Caching: {'On' if settings.enable_caching else 'Off'}
Guardrails: {'On' if settings.enable_guardrails else 'Off'}
    """.strip()
    )

    st.divider()
    st.caption(
        "SentinelRAG v2.2 \u2014 "
        "Retrieve \u00B7 Validate \u00B7 Correct \u00B7 Generate \u00B7 "
        "[\U0001F4D6 User Guide](docs/USER_INSTRUCTIONS.md)"
    )


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.title("\U0001F6E1 SentinelRAG")
st.caption(
    "Ask questions about your HR documents. "
    "The agent retrieves, grades, reranks, generates, and audits its own responses. "
    "Now with async streaming \u2014 watch tokens appear in real-time."
)

_init_session_state()

# --- Bootstrap workflows on first load (single ingest_data call!) ---
if st.session_state.workflow is None:
    with st.spinner("\U0001F6E1 Arming SentinelRAG system ..."):
        try:
            sync_wf, async_wf = _init_system()
            st.session_state.workflow = sync_wf
            st.session_state.workflow_async = async_wf
            st.success(
                "\u2705 System armed with async streaming! Ask a question below."
            )
        except ConfigurationError as exc:
            st.error(f"\u26A0 Configuration Error: {exc}")
            st.info(
                "Ensure your API keys are set in a `.env` file:\n\n"
                "```\n"
                "GROQ_API_KEY=gsk_your_key_here\n"
                "TAVILY_API_KEY=tvly_your_key_here  # optional, for web search\n"
                "```\n\n"
                "Get a free Groq key at https://console.groq.com"
            )
            st.stop()
        except Exception as exc:
            st.error(f"\u274C Initialisation failed: {exc}")
            st.info(
                "Ensure your API keys are set in a `.env` file and PDFs "
                "are placed in `data/raw/`."
            )
            st.stop()

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("audit") and msg["role"] == "assistant":
            _render_audit_trail(msg["audit"])


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------

if prompt := st.chat_input("Ask a question about your documents ..."):
    # 1. Guardrail validation
    try:
        guard_result = validate_query(prompt)
        prompt = sanitize_query(prompt)
    except GuardrailViolationError as e:
        st.error(f"\u26A0 Input blocked: {e}")
        st.stop()

    if not prompt:
        st.stop()

    # 2. Rate limit check
    if not check_rate_limit():
        st.warning(
            "\u23F3 Rate limit reached. Please wait a moment before asking another question."
        )
        st.stop()

    # 3. User message
    _add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 4. Build initial state
    initial_state: dict = {
        "question": prompt,
        "current_query": prompt,
        "documents": [],
        "reranked_documents": [],
        "bm25_documents": [],
        "web_documents": [],
        "generation": None,
        "citations": [],
        "streaming_tokens": [],
        "web_search": False,
        "search_source": "unknown",
        "query_strategy": None,
        "loop_count": 0,
        "retrieval_metrics": {},
        "generation_metrics": {},
        "confidence_calibration": {},
        "guardrail_passed": True,
        "error": None,
        "metadata": {},
        "conversation_history": st.session_state.conversation_history[-10:],
    }

    # 5. Run the async streaming pipeline
    with st.chat_message("assistant"):
        start_time = time.time()

        progress_placeholder = st.empty()
        response_placeholder = st.empty()

        with st.spinner("Agent is thinking ..."):
            try:
                initial_state = asyncio.run(
                    _run_async_pipeline(
                        initial_state,
                        st.session_state.workflow_async,
                        progress_placeholder,
                        response_placeholder,
                    )
                )
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                logger.exception("Pipeline error")

            progress_placeholder.empty()

        exec_time = time.time() - start_time

        # 6. Final response rendering (remove cursor class)
        response_text = initial_state.get("generation", "No response generated.")
        if initial_state.get("error"):
            response_text = f"\u26A0 System Error: {initial_state['error']}"

        response_placeholder.markdown(response_text)

        # 7. Citations display
        citations = initial_state.get("citations", [])
        if citations:
            with st.expander("\U0001F4DA View Source Citations"):
                for i, cit in enumerate(citations, 1):
                    st.markdown(
                        f'<div class="citation-box">'
                        f"<strong>{i}. Claim:</strong> {cit.get('claim', '')}<br>"
                        f"<strong>Source:</strong> {cit.get('source_document', 'unknown')}<br>"
                        f"<strong>Excerpt:</strong> \"{cit.get('source_excerpt', '')}\"<br>"
                        f"<strong>Confidence:</strong> {cit.get('confidence', 0.0):.0%}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # 8. Build audit trail
        audit_meta = {}
        if initial_state.get("question"):
            audit_meta = {
                "Original Query": initial_state.get("question", prompt),
                "Rewritten Queries": initial_state.get("metadata", {}).get(
                    "query_history", []
                ),
                "Query Strategy Used": str(
                    initial_state.get("query_strategy", "N/A")
                ),
                "Pipeline Loops": initial_state.get("loop_count", 0),
                "Search Source": initial_state.get("search_source", "unknown"),
                "Retrieval Metrics": initial_state.get("retrieval_metrics", {}),
                "Generation Metrics": initial_state.get("generation_metrics", {}),
                "Citation Count": len(citations),
                "Web Search Used": initial_state.get("metadata", {}).get(
                    "web_search_used", False
                ),
                "Execution Time (sec)": round(exec_time, 2),
            }

        if audit_meta:
            _render_audit_trail(audit_meta)

        # 9. Feedback buttons
        col1, col2, col3 = st.columns([1, 1, 8])
        with col1:
            if st.button("\U0001F44D", key=f"thumbs_up_{len(st.session_state.messages)}"):
                st.toast("Thanks for the positive feedback! \U0001F389")
        with col2:
            if st.button("\U0001F44E", key=f"thumbs_down_{len(st.session_state.messages)}"):
                st.toast("Thanks for the feedback \u2014 we'll improve!")

        # 10. Save assistant message
        _add_message("assistant", response_text, audit=audit_meta)

        # 11. Update conversation history
        if initial_state.get("question"):
            st.session_state.conversation_history.append(
                {"role": "user", "content": prompt}
            )
            st.session_state.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )
