# 🚀 Pushing SentinelRAG to GitHub — Commit & Push Guide

This guide turns your current working-tree changes (v2.0 → v2.1) into a set of
**meaningful, logical commits** and pushes them to GitHub.

> ✅ Already done for you: `.gitignore` now excludes `.freebuff/` (Freebuff's local
> SQLite DB) and `pdf_extract.txt` (scratch extraction dump) — so neither will be
> committed or pushed. Verify with `git status --short` below.

---

## 0. Pre-flight checks (always do this first)

```bash
# 1) What has changed?
git status --short

# 2) Overview of the diff
git diff --stat

# 3) Make sure the test suite passes BEFORE you commit
pytest
```

Confirm that `.freebuff/` and `pdf_extract.txt` no longer appear as untracked.
If they still do, run `git status --ignored --short | grep -E 'freebuff|pdf_extract'`
to double-check the ignore rules.

---

## 1. Create logical commits (recommended)

Each commit below is one coherent unit. Copy-paste the `git add` + `git commit`
pair per block. **Review with `git status` after each commit.**

### Commit 1 — Core pipeline features (guardrails, web search, hybrid retrieval, reranking)

```bash
git add src/ ingestion/
git commit -m "feat(core): add guardrails, web search fallback, hybrid retrieval and cross-encoder reranking" \
  -m "- Input guardrails: prompt injection detection, homoglyph checks, length limits
- Tavily web search fallback when local context is insufficient
- BM25 sparse index fused with dense Qdrant retrieval (reciprocal rank fusion)
- BGE-reranker-v2-m3 cross-encoder reranking
- Sliding-window rate limiting, streaming-friendly pipeline state"
```

### Commit 2 — Evaluation suite

```bash
git add evaluation/
git commit -m "feat(eval): add RAGAS evaluation with self-play question generation" \
  -m "- Faithfulness, context precision/recall, answer relevancy scoring
- Composite score with configurable weights and JSON report export
- Self-play question generation from ingested documents"
```

### Commit 3 — Tests

```bash
git add tests/
git commit -m "test: update unit tests for v2.1 pipeline"
```

### Commit 4 — Streamlit UI

```bash
git add ui/streamlit_app.py
git commit -m "feat(ui): token streaming chat with audit trail and citations"
```

### Commit 5 — Documentation

```bash
git add README.md RUN_GUIDE.md
# optional: also include this guide itself
# git add GIT_COMMIT_GUIDE.md
git commit -m "docs: revamp README for v2.1 and add run guide"
```

### Commit 6 — Project config & environment template

```bash
git add .env.example pyproject.toml docker-compose.yml
git commit -m "chore: update env template, project metadata and docker compose"
```

> 💡 **Tip:** Want to fold the guide itself into the repo? Add it to Commit 5:
> `git add README.md RUN_GUIDE.md GIT_COMMIT_GUIDE.md`

---

## 2. Review your history

```bash
# Should show 6 new commits on top of 'initial commit'
git log --oneline -10

# See the full picture of what's in each commit
git show --stat HEAD        # most recent commit
git show --stat HEAD~1      # one before, etc.
```

---

## 3. Push to GitHub

If the remote is already configured (check with `git remote -v`):

```bash
git push origin main
```

If you get a **fatal: 'origin' does not appear to be a git repository** —
this is your first push, so add the remote first:

```bash
git remote add origin https://github.com/<your-username>/SentinelRAG.git
git push -u origin main
```

> 🔒 The `.env` file is gitignored, so your real `GROQ_API_KEY` / `TAVILY_API_KEY`
> are **never** pushed. Only the `.env.example` template goes in.

---

## 4. Oops, I messed up — quick undo recipes

| Situation | Command |
|---|---|
| Staged the wrong file (not committed yet) | `git restore --staged <file>` |
| Last commit message is wrong / forgot a file | `git add <file>` then `git commit --amend -m "new message"` |
| Want to merge the last 2 commits into one | `git reset --soft HEAD~2` then commit again |
| Want to cancel everything back to before Commit 1 | `git reset --soft <hash of initial commit>` |

> ⚠️ Only use `reset` **before** pushing, or on a branch nobody else uses.

---

## 5. Alternative: a single "one big commit"

If you prefer one commit over six, replace Section 1 with:

```bash
git add .env.example README.md docker-compose.yml ingestion/ pyproject.toml \
        src/ tests/ ui/ evaluation/ RUN_GUIDE.md
git commit -m "feat: upgrade SentinelRAG to v2.1 with hybrid retrieval, guardrails, web fallback, reranking and RAGAS evaluation"
```

Then continue with Sections 2 and 3.
