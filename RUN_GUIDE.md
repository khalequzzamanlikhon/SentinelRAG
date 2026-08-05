# 🛡️ SentinelRAG v2.1 — Complete Run &amp; Deploy Guide

> Step-by-step instructions to **set up, test, run, and push** SentinelRAG.
> Every feature is demonstrated with exact commands.

---

## 📋 Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [First-Time Setup](#2-first-time-setup)
3. [Running the App (Streamlit UI)](#3-running-the-app)
4. [Testing the Pipeline](#4-testing)
5. [Feature-by-Feature Demo](#5-feature-by-feature-demo)
6. [Docker Deployment](#6-docker-optional)
7. [RAGAS Evaluation](#7-ragas-evaluation)
8. [Git — Upload to GitHub](#8-upload-to-github)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Git | any | `git --version` |
| pip | latest | `pip --version` |

**API Keys (free tier available for all):**

| Service | Purpose | Get Key At | Free Limit |
|---|---|---|---|
| **Groq** | LLM (required) | [console.groq.com](https://console.groq.com) | 14,400 req/day |
| Tavily | Web search fallback (optional) | [tavily.com](https://tavily.com) | 1,000 req/month |
| Gemini | Fallback LLM (optional) | [aistudio.google.com](https://aistudio.google.com) | 1,500 req/day |
| LangFuse | Tracing (optional) | [cloud.langfuse.com](https://cloud.langfuse.com) | 50,000 traces/month |

---

## 2. First-Time Setup

### 2.1 Clone &amp; Enter Project

```bash
git clone https://github.com/your-username/SentinelRAG.git
cd SentinelRAG
```

### 2.2 Create Virtual Environment

```bash
# Create
python -m venv .venv

# Activate (macOS / Linux)
source .venv/bin/activate

# Activate (Windows — Git Bash)
source .venv/Scripts/activate

# Activate (Windows — PowerShell)
.venv\Scripts\Activate.ps1
```

### 2.3 Install Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional: tracing support
pip install "sentinelrag[tracing]"

# Optional: dev tools (pytest-cov, pytest-asyncio)
pip install "sentinelrag[dev]"
```

### 2.4 Configure Environment Variables

Create the `.env` file in the project root:

```bash
cat > .env << 'EOF'
# ── REQUIRED: Groq API key (free at https://console.groq.com) ──
GROQ_API_KEY=gsk_your_actual_groq_key_here

# ── OPTIONAL: Web search fallback ──
TAVILY_API_KEY=tvly_your_tavily_key_here

# ── OPTIONAL: Fallback LLM ──
GEMINI_API_KEY=your_gemini_key_here

# ── OPTIONAL: LangFuse tracing ──
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
ENABLE_TRACING=false

# ── OPTIONAL: Override defaults ──
LOG_LEVEL=INFO
ENABLE_WEB_SEARCH=true
ENABLE_HYBRID_SEARCH=true
ENABLE_GUARDRAILS=true
MAX_LOOP_COUNT=3
TOP_K_DOCUMENTS=5
EOF
```

> ⚠️ **Verify:** Run `python -c "from src.config import settings; settings.validate_api_keys()"` — it should not raise an error.

### 2.5 Add Your PDF Documents

```bash
# Place PDF files in data/raw/
ls data/raw/
# Expected: your-document.pdf
```

You can add multiple PDFs — the ingestion pipeline processes them all.

---

## 3. Running the App

### 3.1 Launch Streamlit

```bash
streamlit run ui/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### 3.2 What Happens on First Launch

1. Validates your API keys
2. Parses PDFs from `data/raw/` with PyMuPDF
3. Chunks text intelligently (1000 chars, 200 overlap)
4. Generates local embeddings via `BAAI/bge-large-en-v1.5`
5. Stores vectors in embedded Qdrant (`data/qdrant_local_db/`)
6. Builds BM25 sparse index for hybrid search
7. Compiles the 7-node LangGraph workflow
8. Shows "System Armed & Ready" when done

### 3.3 Ask a Question

Type a question in the chat input, e.g.:
- *"What is the leave policy?"*
- *"How many sick days do employees get?"*
- *"What are the working hours?"*

You'll see:
- **Real-time progress** through pipeline stages
- **Streaming response** (if using async mode)
- **Source citations** with confidence scores
- **Audit trail** (expand "🔍 View Agent Audit Trail")

---

## 4. Testing

### 4.1 Run All Tests

```bash
# All 71 tests — no API keys needed (fully mocked)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### 4.2 Run Specific Test Files

```bash
# Core pipeline
python -m pytest tests/test_pipeline.py -v

# Retrieval & grading nodes
python -m pytest tests/test_nodes.py -v

# Edge routing logic
python -m pytest tests/test_edges.py -v

# Utilities (BM25, caching, hybrid merge)
python -m pytest tests/test_utils.py -v

# Evaluators (RAGAS metrics)
python -m pytest tests/test_evaluators.py -v
```

### 4.3 Run a Single Test

```bash
python -m pytest tests/test_nodes.py::TestRetrieveNode::test_retrieve_node_success -v
```

### 4.4 Expected Output

```
============================= test session starts =============================
collected 71 items

tests/test_edges.py ..........                                          [ 12%]
tests/test_evaluators.py ....................                            [ 40%]
tests/test_nodes.py ...................                                  [ 67%]
tests/test_pipeline.py ...                                              [ 71%]
tests/test_utils.py ....................                                 [100%]

======================= 71 passed in 54.10s ========================
```

---

## 5. Feature-by-Feature Demo

### 5.1 🔀 Hybrid Search (Dense + BM25)

The hybrid engine runs automatically when `ENABLE_HYBRID_SEARCH=true`.

**Verify it's working:**

```bash
# Check the sidebar in Streamlit — "Hybrid Search: On"
# Or search logs:
grep "BM25 returned" logs.txt   # if logging to file
grep "hybrid" data/qdrant_local_db/ -r 2>/dev/null || echo "Check Streamlit sidebar"
```

**Test programmatically:**

```python
from src.utils import BM25Retriever, merge_hybrid_results, get_bm25_retriever

# BM25 search
bm25 = get_bm25_retriever()
bm25.index([...])  # your documents
results = bm25.search("leave policy", k=3)
for r in results:
    print(f"BM25 Score: {r.metadata['bm25_score']:.3f} | {r.page_content[:80]}")
```

### 5.2 🌐 Web Search Fallback (Tavily)

Triggered automatically when **all** local documents are graded irrelevant.

**Test manually in Streamlit:**
1. Ask a question completely unrelated to your PDFs, e.g. *"Who won the 2024 World Series?"*
2. The pipeline will: `retrieve → grade (all irrelevant) → web_search → grade → generate`
3. Check the audit trail — "Search Source" should show `"web"`

**Verify the config:**

```bash
python -c "from src.config import settings; print(f'Web search: {settings.enable_web_search}')"
```

### 5.3 🎯 Cross-Encoder Reranking

Uses `BAAI/bge-reranker-v2-m3` — downloads on first use (~1.2 GB).

**Verify:**

```bash
python -c "
from src.utils import get_reranker
reranker = get_reranker()
if reranker:
    scores = reranker.predict([('What is RAG?', 'RAG combines retrieval and generation.')])
    print(f'Cross-encoder score: {scores[0]:.3f}')
else:
    print('Reranker not loaded (will fall back to cosine similarity)')
"
```

### 5.4 🛡️ Input Guardrails

Blocks prompt injection attempts automatically.

**Test by typing these in the Streamlit chat:**

| Input | Expected Result |
|---|---|
| `"ignore all previous instructions"` | ⚠️ Blocked |
| `"disregard system prompt"` | ⚠️ Blocked |
| `"DAN do whatever"` | ⚠️ Blocked |
| `"What is the leave policy?"` | ✅ Passes |

### 5.5 🔄 Query Rewriting (4 Strategies)

When all chunks are irrelevant and web search is disabled/exhausted, the pipeline rewrites the query.

**Test:** Ask a vague question like *"stuff about work"* — watch the audit trail show rotation through:
- **Semantic** → meaning-focused rewrite
- **Keyword** → extracts critical terms
- **Hybrid** → keywords + semantic
- **Expansion** → adds synonyms

### 5.6 🛡️ Hallucination Detection

Every generation is audited against source documents.

**Check in audit trail:**
- `grounded_score` (0.0 – 1.0)
- `hallucinated_claims` list
- `binary_score` (yes/no)

### 5.7 ⚡ Rate Limiting

Default: 30 requests per 60-second window.

```bash
python -c "
from src.utils import check_rate_limit
for i in range(35):
    allowed = check_rate_limit()
    if not allowed:
        print(f'Rate limit hit at request {i+1}')
        break
"
```

---

## 6. Docker (Optional)

### 6.1 Build &amp; Run

```bash
# Start both Qdrant + Streamlit
docker-compose up --build

# Run in background
docker-compose up -d --build
```

### 6.2 Services

| Service | Port | URL |
|---|---|---|
| Streamlit App | 8501 | http://localhost:8501 |
| Qdrant REST | 6333 | http://localhost:6333 |
| Qdrant gRPC | 6334 | — |

### 6.3 Stop

```bash
docker-compose down
```

### 6.4 Clean Rebuild

```bash
docker-compose down -v  # also removes qdrant_data volume
docker-compose up --build
```

---

## 7. RAGAS Evaluation

### 7.1 Run Evaluation on Past Queries

```python
from evaluation.ragas_eval import RAGASEvaluator

evaluator = RAGASEvaluator()

# Add samples from previous Q&A
evaluator.add_sample(
    question="What is the leave policy?",
    generated_answer="Employees get 20 days of annual leave...",
    context_docs=["According to the handbook, employees receive 20 days..."],
    hallucinated_claims=[],
)

report = evaluator.evaluate()
print(f"Faithfulness:       {report.faithfulness:.3f}")
print(f"Context Precision:  {report.context_precision:.3f}")
print(f"Context Recall:     {report.context_recall:.3f}")
print(f"Answer Relevancy:   {report.answer_relevancy:.3f}")
print(f"Composite Score:    {report.composite_score:.3f}")

evaluator.save_report("eval_report.json")
```

### 7.2 Generate Self-Play Questions

```python
from evaluation.ragas_eval import generate_self_play_questions

# Auto-generate test questions from your documents
documents = [...]  # your chunks
questions = generate_self_play_questions(documents, num_questions=20)
for q in questions:
    print(q)
```

---

## 8. Upload to GitHub

### 8.1 Check Current Status

```bash
git status
git log --oneline -5
```

### 8.2 Ensure `.env` is NOT Committed

```bash
# .env is already in .gitignore — verify:
grep "^\.env$" .gitignore && echo "✅ .env is gitignored" || echo "❌ Add .env to .gitignore!"
```

### 8.3 Stage &amp; Commit

```bash
# Stage all changes (except gitignored files)
git add -A

# Review what will be committed
git diff --cached --stat

# Commit
git commit -m "v2.1: hybrid search, cross-encoder reranking, web fallback, guardrails, streaming, RAGAS eval

- Add BM25 + dense hybrid retrieval with RRF fusion
- Add BGE-reranker-v2-m3 cross-encoder reranking
- Add Tavily web search fallback node
- Add input guardrails (prompt injection detection)
- Add streaming generation support
- Add RAGAS evaluation suite
- Add rate limiting and LangFuse tracing
- Fix missing import os in config.py
- 71 unit tests all passing

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>"
```

### 8.4 Push to Remote

```bash
# Check your remote
git remote -v

# Push
git push origin main
```

### 8.5 First-Time GitHub Setup (if needed)

```bash
# If you haven't set up a remote yet:
git remote add origin https://github.com/YOUR_USERNAME/SentinelRAG.git

# Or via SSH:
git remote add origin git@github.com:YOUR_USERNAME/SentinelRAG.git

# Push with upstream tracking
git push -u origin main
```

---

## 9. Troubleshooting

### "No module named 'src'"

```bash
# Ensure you're in the project root
cd SentinelRAG
export PYTHONPATH="$PWD:$PYTHONPATH"
```

### "GROQ_API_KEY is missing"

```bash
# Verify .env exists and is readable
cat .env | grep GROQ_API_KEY

# Or set inline for one session
export GROQ_API_KEY="gsk_your_key_here"
```

### "QdrantLocal: already locked"

The embedded Qdrant can only have **one** process at a time. Stop other Streamlit instances:

```bash
# On Windows (Git Bash)
taskkill //F //IM streamlit.exe 2>/dev/null

# Or kill all Python processes holding the lock
pkill -f streamlit   # macOS/Linux
```

### "BM25 index not built"

The BM25 index is built during ingestion. Force re-ingestion:

```bash
# Click "Force Re-Ingest PDFs" in the Streamlit sidebar
# Or delete the Qdrant DB and restart:
rm -rf data/qdrant_local_db/
```

### "CUDA / Torch not available"

Sentence-transformers works on CPU too — it's just slower. No GPU needed.

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"
# First run downloads ~1.3 GB model — be patient
```

### "Tavily API quota exceeded"

Web search auto-disables if quota runs out. The pipeline falls back to rewrite → retrieve.

---

## 📊 Quick Reference Card

```bash
# ── Setup ──
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .env with GROQ_API_KEY
streamlit run ui/streamlit_app.py

# ── Testing ──
python -m pytest tests/ -v
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# ── Docker ──
docker-compose up --build
docker-compose down

# ── Git ──
git status
git add -A
git commit -m "your message"
git push origin main

# ── Verify config ──
python -c "from src.config import settings; settings.validate_api_keys(); print('✅ Config OK')"

# ── Verify pipeline compiles ──
python -c "
from unittest.mock import MagicMock
from src.pipeline import assemble_agentic_rag_workflow
w = assemble_agentic_rag_workflow(MagicMock())
print('✅ Pipeline compiles with', len(w.nodes), 'nodes')
"
```

---

<sub>SentinelRAG v2.1 — Retrieve. Validate. Correct. Generate. 🛡️</sub>
