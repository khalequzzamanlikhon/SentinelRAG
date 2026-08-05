"""
LangGraph nodes for the SentinelRAG agentic RAG pipeline.

Each node is a callable that accepts ``state: AgentState`` (plus optional
keyword dependencies) and returns a dict of state updates.

Nodes (v2.1):
  - ``retrieve_node``              — hybrid dense+BM25 retrieval
  - ``grade_documents_node``       — LLM relevance grading
  - ``rewrite_query_node``         — multi-strategy query rewriting
  - ``rerank_documents_node``      — cross-encoder reranking (real model)
  - ``web_search_node``            — Tavily web search fallback (NEW)
  - ``generate_node``              — response synthesis with streaming
  - ``extract_citations_node``     — source attribution
"""

import logging
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.config import settings
from src.state import (
    AgentState,
    GradeDocument,
    GradeHallucination,
    Citation,
    CitationList,
    QueryStrategy,
)
from src.utils import (
    compute_document_similarity,
    cache_llm_call,
    deterministic_hash,
    deduplicate_documents,
    rerank_with_cross_encoder,
    get_bm25_retriever,
    merge_hybrid_results,
    check_rate_limit,
)
from src.evaluators import compute_retrieval_metrics
from src.exceptions import (
    RetrievalError,
    GradingError,
    QueryRewriteError,
    GenerationError,
    RerankerError,
)

logger = logging.getLogger(__name__)


# ===================================================================
# Shared helpers
# ===================================================================


def _get_llm(temperature: float) -> ChatOpenAI:
    """Build a Groq LLM client via Groq's OpenAI-compatible endpoint.

    Groq provides a free API with a generous rate limit (14,400 req/day).
    Includes timeout and retry configuration for production resilience.
    """
    return ChatOpenAI(
        model=settings.groq_model,
        temperature=temperature,
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


# ===================================================================
# 1. Retrieval Node (hybrid: dense + BM25)
# ===================================================================


def retrieve_node(state: AgentState, vectorstore) -> dict:
    """Query the vector database + BM25 with the current active query string.

    When hybrid search is enabled, merges dense (Qdrant) and sparse (BM25)
    results using reciprocal rank fusion. Returns sorted, deduplicated documents
    with retrieval metrics.

    Raises:
        RetrievalError: If retrieval fails critically.
    """
    logger.info("--- NODE: RETRIEVING DOCUMENTS (hybrid) ---")
    query = state["current_query"]
    metadata = state.get("metadata", {})

    try:
        # Dense retrieval
        retrieved_docs = vectorstore.similarity_search(query, k=settings.top_k_documents)

        docs_with_scores: List[Document] = []
        for doc in retrieved_docs:
            score = compute_document_similarity(query, doc.page_content)
            doc.metadata["similarity_score"] = score
            doc.metadata["search_source"] = "dense"
            docs_with_scores.append(doc)

        docs_with_scores.sort(
            key=lambda x: x.metadata.get("similarity_score", 0), reverse=True
        )
        docs_with_scores = deduplicate_documents(docs_with_scores)

        # BM25 sparse retrieval (if enabled)
        bm25_docs: List[Document] = []
        if settings.enable_hybrid_search:
            try:
                bm25 = get_bm25_retriever()
                bm25_docs = bm25.search(query, k=settings.top_k_documents)
                logger.info("BM25 returned %d documents.", len(bm25_docs))
            except Exception as e:
                logger.warning("BM25 search failed (continuing with dense only): %s", e)

        # Merge hybrid results
        if bm25_docs:
            final_docs = merge_hybrid_results(
                dense_docs=docs_with_scores,
                bm25_docs=bm25_docs,
                dense_weight=settings.dense_weight,
                bm25_weight=settings.bm25_weight,
                top_k=settings.top_k_documents,
            )
        else:
            final_docs = docs_with_scores

        retrieval_metrics = compute_retrieval_metrics(query, final_docs)

        logger.info(
            "Retrieved %d documents (dense=%d, bm25=%d, merged=%d).",
            len(final_docs),
            len(docs_with_scores),
            len(bm25_docs),
            len(final_docs),
        )

        return {
            "documents": final_docs,
            "bm25_documents": bm25_docs,
            "retrieval_metrics": retrieval_metrics,
            "search_source": "hybrid" if bm25_docs else "dense",
            "metadata": {**metadata, "last_retrieval_count": len(final_docs)},
        }
    except Exception as e:
        logger.exception("Error during retrieval")
        raise RetrievalError(
            f"Retrieval failed: {e}",
            details={"query": query, "error": str(e)},
        )


# ===================================================================
# 2. Document Grading Node
# ===================================================================


def grade_documents_node(state: AgentState) -> dict:
    """Evaluate retrieved documents for relevance to the current query.

    Drops irrelevant chunks and sets ``web_search`` to ``True`` when *all*
    chunks are dropped to trigger external search.

    Raises:
        GradingError: If grading fails for all documents.
    """
    logger.info("--- NODE: GRADING DOCUMENTS ---")
    query = state["current_query"]
    documents = state.get("documents", [])
    current_loops = state.get("loop_count", 0)
    metadata = state.get("metadata", {})

    # Check rate limit
    if not check_rate_limit():
        logger.warning("Rate limit exceeded — proceeding with ungraded documents.")
        return {
            "documents": documents,
            "web_search": False,
            "loop_count": current_loops + 1,
            "metadata": {**metadata, "rate_limited": True},
        }

    llm = _get_llm(temperature=settings.temperature_deterministic)
    structured_grader = llm.with_structured_output(GradeDocument, method="function_calling")

    system_prompt = (
        "You are an objective auditor assessing if a retrieved document contains semantic information "
        "relevant to answering the user query. Provide a relevance score between 0.0 and 1.0, "
        "a binary grade ('yes' or 'no'), and a one-sentence rationale."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User Query: {query}\n\nRetrieved Document Chunks:\n{doc_content}"),
    ])
    grader_chain = prompt | structured_grader

    valid_documents: List[Document] = []
    grading_results: list = []

    for doc in documents:
        try:
            cache_key = "grade_{}_{}".format(query, deterministic_hash(doc.page_content[:200]))
            result = cache_llm_call(
                grader_chain.invoke,
                {"query": query, "doc_content": doc.page_content},
                cache_key=cache_key,
            )

            doc.metadata["grading"] = {
                "relevance_score": result.relevance_score,
                "binary_score": result.binary_score,
                "reasoning": result.reasoning,
            }

            grading_results.append({
                "doc_id": doc.metadata.get("doc_id", "unknown"),
                "relevance_score": result.relevance_score,
                "binary_score": result.binary_score,
            })

            if result.binary_score == "yes" and result.relevance_score >= settings.similarity_threshold:
                logger.info(
                    "[KEEP] Chunk relevant (score: %.2f). %s",
                    result.relevance_score,
                    result.reasoning,
                )
                valid_documents.append(doc)
            else:
                logger.info(
                    "[DROP] Chunk irrelevant (score: %.2f). %s",
                    result.relevance_score,
                    result.reasoning,
                )
        except Exception as e:
            logger.error("Error grading document: %s", e)

    trigger_rewrite = len(valid_documents) == 0
    if trigger_rewrite:
        logger.info("All chunks dropped — triggering external search/rewrite.")

    return {
        "documents": valid_documents,
        "web_search": trigger_rewrite,
        "loop_count": current_loops + 1,
        "metadata": {
            **metadata,
            "grading_results": grading_results,
            "valid_document_count": len(valid_documents),
        },
    }


# ===================================================================
# 3. Web Search Node (NEW — Tavily fallback)
# ===================================================================


def web_search_node(state: AgentState) -> dict:
    """Search the web via Tavily when local retrieval is insufficient.

    Triggered when all documents are graded irrelevant AND max loops
    haven't been reached yet.
    """
    logger.info("--- NODE: WEB SEARCH (Tavily) ---")
    query = state["current_query"]
    metadata = state.get("metadata", {})

    if not settings.enable_web_search:
        logger.info("Web search disabled — returning empty.")
        return {"web_documents": [], "web_search": False}

    try:
        from src.web_search import search_web_safe
        web_docs = search_web_safe(query)
        logger.info("Web search returned %d documents.", len(web_docs))
    except Exception as e:
        logger.error("Web search failed: %s", e)
        web_docs = []

    return {
        "web_documents": web_docs,
        "documents": web_docs,  # use web results for generation
        "web_search": False,    # reset flag
        "search_source": "web",
        "metadata": {
            **metadata,
            "web_search_attempted": True,
            "web_search_used": True,
            "web_result_count": len(web_docs),
        },
    }


# ===================================================================
# 4. Reranking Node (cross-encoder — real model)
# ===================================================================


def rerank_documents_node(state: AgentState) -> dict:
    """Re-rank graded documents using a cross-encoder model.

    Uses ``BAAI/bge-reranker-v2-m3`` (or similar) for precise relevance
    scoring. Falls back to cosine similarity if the model is unavailable.

    Raises:
        RerankerError: If reranking fails.
    """
    logger.info("--- NODE: RERANKING DOCUMENTS (cross-encoder) ---")
    documents = state.get("documents", [])
    query = state["current_query"]
    metadata = state.get("metadata", {})

    if not documents:
        return {"reranked_documents": []}

    if not settings.enable_cross_encoder:
        logger.info("Cross-encoder disabled — skipping rerank.")
        return {"reranked_documents": documents}

    try:
        reranked = rerank_with_cross_encoder(query, documents)
        logger.info("Re-ranked %d documents with cross-encoder.", len(reranked))

        return {
            "reranked_documents": reranked,
            "metadata": {**metadata, "reranked": True, "rerank_method": "cross_encoder"},
        }
    except Exception as e:
        logger.error("Error reranking documents: %s", e)
        raise RerankerError(
            f"Reranking failed: {e}",
            details={"query": query, "doc_count": len(documents)},
        )


# ===================================================================
# 5. Query Rewriting Node
# ===================================================================


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite the current query using a rotating strategy.

    Cycles through SEMANTIC → KEYWORD → HYBRID → EXPANSION based on
    ``loop_count``.

    Raises:
        QueryRewriteError: If rewriting fails and fallback is needed.
    """
    logger.info("--- NODE: REWRITING USER QUERY ---")
    bad_query = state["current_query"]
    loop_count = state.get("loop_count", 0)
    metadata = state.get("metadata", {})

    strategies = list(QueryStrategy)
    strategy = strategies[loop_count % len(strategies)]

    llm = _get_llm(temperature=settings.temperature_creative)

    strategy_prompts = {
        QueryStrategy.SEMANTIC: (
            "You are an expert search-optimization engine. Analyze the provided failing query and "
            "rewrite it to improve its semantic overlap with relevant documents. "
            "Focus on meaning rather than exact keywords. "
            "Output ONLY the optimized query string text. Do not add introductions or markdown formatting."
        ),
        QueryStrategy.KEYWORD: (
            "You are an expert search-optimization engine. Analyze the provided failing query and "
            "extract the most important keywords. Create a new query with just these keywords, "
            "reordered by importance. "
            "Output ONLY the optimized query string text. Do not add introductions or markdown formatting."
        ),
        QueryStrategy.HYBRID: (
            "You are an expert search-optimization engine. Analyze the provided failing query and "
            "create a hybrid query that combines important keywords with semantic phrases. "
            "Output ONLY the optimized query string text. Do not add introductions or markdown formatting."
        ),
        QueryStrategy.EXPANSION: (
            "You are an expert search-optimization engine. Analyze the provided failing query and "
            "expand it by adding synonyms, related terms, and broader concepts. "
            "Output ONLY the optimized query string text. Do not add introductions or markdown formatting."
        ),
    }

    system_prompt = strategy_prompts[strategy]
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Failing query target: {bad_query}"),
    ])

    rewriter_chain = prompt | llm
    try:
        cache_key = "rewrite_{}_{}".format(bad_query, strategy.value)
        optimized_response = cache_llm_call(
            rewriter_chain.invoke,
            {"bad_query": bad_query},
            cache_key=cache_key,
        )
        new_query = optimized_response.content.strip()
        logger.info("Query adapted via %s: '%s' → '%s'", strategy.value, bad_query, new_query)
    except Exception as e:
        logger.error("Error rewriting query: %s", e)
        new_query = bad_query
        logger.info("Falling back to original query.")

    return {
        "current_query": new_query,
        "web_search": False,
        "query_strategy": strategy,
        "metadata": {
            **metadata,
            "last_rewrite_strategy": strategy.value,
            "query_history": metadata.get("query_history", []) + [bad_query],
        },
    }


# ===================================================================
# 6. Generation Node (with async streaming support)
# ===================================================================


def generate_node(state: AgentState) -> dict:
    """Synthesise the final answer from validated context.

    Uses reranked documents if available, otherwise falls back to
    graded documents. Supports token-level streaming via
    ``generate_node_stream`` for async usage.

    Raises:
        GenerationError: If generation fails.
    """
    logger.info("--- NODE: GENERATING RESPONSE ---")
    question = state["question"]
    documents = state.get("reranked_documents") or state.get("documents", [])
    metadata = state.get("metadata", {})

    context = (
        "\n\n".join([doc.page_content for doc in documents])
        if documents
        else "No valid context found."
    )

    llm = _get_llm(temperature=settings.temperature_deterministic)

    system_prompt = (
        "You are an enterprise technical support expert assistant. Synthesize a professional, "
        "accurate, and fully complete response based strictly on the provided context block. "
        "If the context does not provide data to answer, state clearly that the answer cannot be computed. "
        "Do not make up information not present in the context.\n\n"
        "When making claims, cite the source filename and provide the exact supporting excerpt "
        "from the context. Format citations as: [Source: <filename>, Excerpt: \"...\"]"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context Materials:\n{context}\n\nUser Question: {question}"),
    ])

    try:
        generator_chain = prompt | llm
        context_hash = deterministic_hash(context)
        cache_key = "generate_{}_{}".format(question, context_hash)

        generation = cache_llm_call(
            generator_chain.invoke,
            {"context": context, "question": question},
            cache_key=cache_key,
        )

        generated_text = generation.content

        generation_metrics: Dict[str, Any] = {}
        if documents:
            generation_metrics = detect_hallucinations(generated_text, documents)

        return {
            "generation": generated_text,
            "streaming_tokens": [],
            "current_query": question,
            "documents": documents,
            "generation_metrics": generation_metrics,
            "metadata": {
                **metadata,
                "generation_length": len(generated_text),
                "generation_token_count": len(generated_text.split()),
            },
        }
    except Exception as e:
        logger.exception("Error during generation")
        raise GenerationError(
            f"Generation failed: {e}",
            details={"question": question, "doc_count": len(documents)},
        )


async def generate_node_stream(state: AgentState) -> dict:
    """Generate response with async LLM streaming, returning the complete result.

    Uses the LLM's ``.astream()`` internally for non-blocking token collection.
    All tokens are accumulated into ``streaming_tokens`` before returning.
    The UI consumes partial tokens via the async workflow's ``.astream()``.

    Args:
        state: Current pipeline state.

    Returns:
        Dict with ``generation``, ``streaming_tokens``, and full metrics.

    Raises:
        GenerationError: If generation fails.
    """
    logger.info("--- NODE: GENERATING RESPONSE (async streaming) ---")
    question = state["question"]
    documents = state.get("reranked_documents") or state.get("documents", [])
    metadata = state.get("metadata", {})

    context = (
        "\n\n".join([doc.page_content for doc in documents])
        if documents
        else "No valid context found."
    )

    llm = _get_llm(temperature=settings.temperature_deterministic)

    system_prompt = (
        "You are an enterprise technical support expert assistant. Synthesize a professional, "
        "accurate, and fully complete response based strictly on the provided context block. "
        "If the context does not provide data to answer, state clearly that the answer cannot be computed. "
        "Do not make up information not present in the context.\n\n"
        "When making claims, cite the source filename and provide the exact supporting excerpt "
        "from the context. Format citations as: [Source: <filename>, Excerpt: \"...\"]"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context Materials:\n{context}\n\nUser Question: {question}"),
    ])

    generator_chain = prompt | llm
    tokens: List[str] = []
    full_text = ""

    try:
        async for chunk in generator_chain.astream(
            {"context": context, "question": question}
        ):
            if hasattr(chunk, "content") and chunk.content:
                token = chunk.content
                tokens.append(token)
                full_text += token
    except Exception as e:
        logger.exception("Error during async generation")
        raise GenerationError(
            f"Generation failed: {e}",
            details={"question": question, "doc_count": len(documents)},
        )

    generation_metrics: Dict[str, Any] = {}
    if documents:
        generation_metrics = detect_hallucinations(full_text, documents)

    return {
        "generation": full_text,
        "streaming_tokens": tokens,
        "current_query": question,
        "documents": documents,
        "generation_metrics": generation_metrics,
        "metadata": {
            **metadata,
            "generation_length": len(full_text),
            "generation_token_count": len(tokens),
        },
    }


# ===================================================================
# 7. Hallucination Detection
# ===================================================================


def detect_hallucinations(generation: str, documents: List[Document]) -> Dict[str, Any]:
    """Audit the generated response for factual grounding against source documents.

    Returns detailed metrics including a list of specific unsupported claims.
    Also computes confidence calibration metrics.
    """
    logger.info("--- DETECTING HALLUCINATIONS ---")

    llm = _get_llm(temperature=settings.temperature_deterministic)
    structured_grader = llm.with_structured_output(GradeHallucination, method="function_calling")

    context = "\n\n".join([doc.page_content for doc in documents])

    system_prompt = (
        "You are an objective auditor assessing if a generated response is fully grounded in the "
        "provided context. Identify any claims in the generation that are not supported by the context. "
        "Provide a grounded score between 0.0 and 1.0, a binary grade ('yes' or 'no'), "
        "an explanation, and a list of specific hallucinated claims if any."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context Materials:\n{context}\n\nGenerated Response:\n{generation}"),
    ])

    grader_chain = prompt | structured_grader

    try:
        cache_key = "hallucination_{}".format(deterministic_hash(generation))
        result = cache_llm_call(
            grader_chain.invoke,
            {"context": context, "generation": generation},
            cache_key=cache_key,
        )

        logger.info(
            "Hallucination check: %s (score: %.2f, claims: %d)",
            result.binary_score,
            result.grounded_score,
            len(result.hallucinated_claims),
        )

        return {
            "grounded_score": result.grounded_score,
            "binary_score": result.binary_score,
            "reasoning": result.reasoning,
            "hallucinated_claims": result.hallucinated_claims,
        }
    except Exception as e:
        logger.error("Error during hallucination detection: %s", e)
        return {
            "grounded_score": 0.0,
            "binary_score": "no",
            "reasoning": f"Hallucination detection failed: {e}",
            "hallucinated_claims": [],
        }


# ===================================================================
# 8. Citation Extraction Node
# ===================================================================


def extract_citations_node(state: AgentState) -> dict:
    """Extract structured source citations from the generated response.

    Uses the LLM to parse the generation and map each claim back to its
    supporting source document and excerpt.
    """
    logger.info("--- NODE: EXTRACTING CITATIONS ---")
    generation = state.get("generation", "")
    documents = state.get("reranked_documents") or state.get("documents", [])
    metadata = state.get("metadata", {})

    if not generation or not documents:
        return {"citations": []}

    llm = _get_llm(temperature=settings.temperature_deterministic)
    structured_extractor = llm.with_structured_output(CitationList, method="function_calling")

    # Use up to 10 documents with 1000 chars each (was 5/500)
    context_preview = "\n\n".join([
        f"[Doc: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:1000]}"
        for doc in documents[:10]
    ])

    system_prompt = (
        "You are a citation extraction engine. Your job is to read a generated response "
        "and the provided source documents, then extract structured citations for each "
        "factual claim made in the response that can be attributed to a specific source.\n\n"
        "For each citation, provide:\n"
        "  - claim: the factual claim from the generation\n"
        "  - source_document: the filename of the supporting document\n"
        "  - source_excerpt: the exact text from the document that supports the claim\n"
        "  - confidence: your confidence (0.0-1.0) that this source truly supports the claim\n\n"
        "Output ONLY a list of Citation objects. If no citations can be extracted, "
        "output an empty list."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            "Generated Response:\n{generation}\n\n"
            "Source Documents:\n{context_preview}\n\n"
            "Extract citations from the generated response.",
        ),
    ])

    extractor_chain = prompt | structured_extractor

    try:
        cache_key = "citations_{}".format(deterministic_hash(generation))
        result = cache_llm_call(
            extractor_chain.invoke,
            {"generation": generation, "context_preview": context_preview},
            cache_key=cache_key,
        )

        citations = [c.model_dump() for c in result.citations]

        logger.info("Extracted %d citations.", len(citations))
        return {"citations": citations}
    except Exception as e:
        logger.error("Error extracting citations: %s", e)
        return {"citations": []}
