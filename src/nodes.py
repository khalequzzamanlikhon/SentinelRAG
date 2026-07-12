"""
LangGraph nodes for the SentinelRAG agentic RAG pipeline.

Each node is a callable that accepts ``state: AgentState`` (plus optional
keyword dependencies) and returns a dict of state updates.

Nodes:
  - ``retrieve_node``          — vector similarity search
  - ``grade_documents_node``   — LLM relevance grading
  - ``rewrite_query_node``     — multi-strategy query rewriting
  - ``generate_node``          — response synthesis
  - ``detect_hallucinations``  — grounding audit
  - ``rerank_documents_node``  — cross-encoder reranking (NEW)
  - ``extract_citations_node`` — source attribution (NEW)
"""

import logging
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from src.config import settings
from src.state import (
    AgentState,
    GradeDocument,
    GradeHallucination,
    Citation,
    QueryStrategy,
)
from src.utils import (
    compute_document_similarity,
    cache_llm_call,
    deterministic_hash,
    deduplicate_documents,
)
from src.evaluators import compute_retrieval_metrics

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
# 1. Retrieval Node
# ===================================================================


def retrieve_node(state: AgentState, vectorstore) -> dict:
    """Query the vector database with the current active query string.

    Returns documents sorted by cosine similarity along with retrieval metrics.
    """
    logger.info("--- NODE: RETRIEVING DOCUMENTS ---")
    query = state["current_query"]
    metadata = state.get("metadata", {})

    try:
        retrieved_docs = vectorstore.similarity_search(query, k=settings.top_k_documents)

        docs_with_scores: List[Document] = []
        for doc in retrieved_docs:
            score = compute_document_similarity(query, doc.page_content)
            doc.metadata["similarity_score"] = score
            docs_with_scores.append(doc)

        docs_with_scores.sort(
            key=lambda x: x.metadata.get("similarity_score", 0), reverse=True
        )

        # Deduplicate
        docs_with_scores = deduplicate_documents(docs_with_scores)

        retrieval_metrics = compute_retrieval_metrics(query, docs_with_scores)

        logger.info("Retrieved %d candidate documents.", len(docs_with_scores))

        return {
            "documents": docs_with_scores,
            "retrieval_metrics": retrieval_metrics,
            "metadata": {**metadata, "last_retrieval_count": len(docs_with_scores)},
        }
    except Exception as e:
        logger.exception("Error during retrieval")
        return {
            "documents": [],
            "error": f"Retrieval failed: {e}",
            "metadata": {**metadata, "retrieval_error": str(e)},
        }


# ===================================================================
# 2. Document Grading Node
# ===================================================================


def grade_documents_node(state: AgentState) -> dict:
    """Evaluate retrieved documents for relevance to the current query.

    Drops irrelevant chunks and sets ``web_search`` to ``True`` when *all*
    chunks are dropped to trigger a query rewrite.
    """
    logger.info("--- NODE: GRADING DOCUMENTS ---")
    query = state["current_query"]
    documents = state.get("documents", [])
    current_loops = state.get("loop_count", 0)
    metadata = state.get("metadata", {})

    llm = _get_llm(temperature=settings.temperature_deterministic)
    structured_grader = llm.with_structured_output(GradeDocument, method='function_calling')

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
        logger.info("All chunks dropped — marking for query rewrite.")

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
# 3. Reranking Node  (NEW)
# ===================================================================


def rerank_documents_node(state: AgentState) -> dict:
    """Re-rank graded documents using cross-encoder style scoring.

    In a production system this would call a dedicated cross-encoder model.
    Here we re-score with a focused LLM prompt that considers the full
    query-document pair, then sort by the new score.
    """
    logger.info("--- NODE: RERANKING DOCUMENTS ---")
    documents = state.get("documents", [])
    query = state["current_query"]
    metadata = state.get("metadata", {})

    if not documents:
        return {"reranked_documents": []}

    llm = _get_llm(temperature=settings.temperature_deterministic)

    system_prompt = (
        "You are a precise relevance rater. Given a query and a document chunk, "
        "output ONLY a single float score between 0.0 (completely irrelevant) "
        "and 1.0 (perfectly relevant). Do NOT include any explanation or formatting."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Query: {query}\n\nDocument:\n{doc_content}"),
    ])
    chain = prompt | llm

    scored: list = []
    for doc in documents:
        try:
            cache_key = "rerank_{}_{}".format(query, deterministic_hash(doc.page_content[:200]))
            response = cache_llm_call(
                chain.invoke,
                {"query": query, "doc_content": doc.page_content[:1500]},
                cache_key=cache_key,
            )
            try:
                rerank_score = float(response.content.strip())
            except ValueError:
                rerank_score = doc.metadata.get("grading", {}).get("relevance_score", 0.0)

            doc.metadata["rerank_score"] = rerank_score
            scored.append((rerank_score, doc))
        except Exception as e:
            logger.error("Error reranking document: %s", e)
            doc.metadata["rerank_score"] = 0.0
            scored.append((0.0, doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored]

    logger.info("Re-ranked %d documents.", len(reranked))

    return {
        "reranked_documents": reranked,
        "metadata": {**metadata, "reranked": True},
    }


# ===================================================================
# 4. Query Rewriting Node
# ===================================================================


def rewrite_query_node(state: AgentState) -> dict:
    """Rewrite the current query using a rotating strategy.

    Cycles through SEMANTIC → KEYWORD → HYBRID → EXPANSION based on
    ``loop_count``.
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
# 5. Generation Node
# ===================================================================


def generate_node(state: AgentState) -> dict:
    """Synthesise the final answer from validated context.

    Uses reranked documents if available, otherwise falls back to
    graded documents.
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
        "from the context. Format citations as: [Source: <filename>, Excerpt: \"...\"]"  # (prepares for citation extraction)
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
        return {
            "generation": "I encountered an error while generating.",
            "current_query": question,
            "documents": documents,
            "error": f"Generation failed: {e}",
            "metadata": {**metadata, "generation_error": str(e)},
        }


# ===================================================================
# 6. Hallucination Detection
# ===================================================================


def detect_hallucinations(generation: str, documents: List[Document]) -> Dict[str, Any]:
    """Audit the generated response for factual grounding against source documents.

    Returns detailed metrics including a list of specific unsupported claims.
    """
    logger.info("--- DETECTING HALLUCINATIONS ---")

    llm = _get_llm(temperature=settings.temperature_deterministic)
    structured_grader = llm.with_structured_output(GradeHallucination, method='function_calling')

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

        logger.info("Hallucination check: %s (score: %.2f)", result.binary_score, result.grounded_score)
        if result.hallucinated_claims:
            logger.info("Identified %d hallucinated claims.", len(result.hallucinated_claims))

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
# 7. Citation Extraction Node  (NEW)
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
    structured_extractor = llm.with_structured_output(Citation, method='function_calling')

    context_preview = "\n\n".join([
        f"[Doc: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content[:500]}"
        for doc in documents[:5]
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

        # structured_extractor returns a list of Citation objects
        if isinstance(result, list):
            citations = [c.model_dump() for c in result]
        else:
            citations = [result.model_dump()]

        logger.info("Extracted %d citations.", len(citations))
        return {"citations": citations}
    except Exception as e:
        logger.error("Error extracting citations: %s", e)
        return {"citations": []}
