"""
RAGAS-style evaluation framework for SentinelRAG.

Provides automated evaluation of the RAG pipeline against standard metrics
and generates self-play evaluation datasets from indexed documents.

Usage::

    python -m evaluation.ragas_eval  # run the evaluation suite
    python -m evaluation.ragas_eval --generate-questions  # generate test Q&A
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field, asdict

from src.config import settings
from src.evaluators import (
    compute_faithfulness,
    compute_context_precision,
    compute_context_recall,
    compute_answer_relevancy,
    compute_composite_ragas_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample with ground truth and predictions."""
    question: str
    ground_truth_answer: str = ""
    generated_answer: str = ""
    context_docs: List[str] = field(default_factory=list)
    hallucinated_claims: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Aggregated evaluation report."""
    samples: List[EvalSample] = field(default_factory=list)
    avg_faithfulness: float = 0.0
    avg_context_precision: float = 0.0
    avg_context_recall: float = 0.0
    avg_answer_relevancy: float = 0.0
    composite_score: float = 0.0
    total_samples: int = 0

    def compute(self) -> None:
        """Aggregate metrics across all samples."""
        if not self.samples:
            return

        n = len(self.samples)
        self.total_samples = n
        self.avg_faithfulness = sum(s.metrics.get("faithfulness", 0) for s in self.samples) / n
        self.avg_context_precision = sum(s.metrics.get("context_precision", 0) for s in self.samples) / n
        self.avg_context_recall = sum(s.metrics.get("context_recall", 0) for s in self.samples) / n
        self.avg_answer_relevancy = sum(s.metrics.get("answer_relevancy", 0) for s in self.samples) / n

        composite = compute_composite_ragas_score(
            faithfulness=self.avg_faithfulness,
            context_precision=self.avg_context_precision,
            context_recall=self.avg_context_recall,
            answer_relevancy=self.avg_answer_relevancy,
        )
        self.composite_score = composite["composite_score"]


class RAGASEvaluator:
    """Evaluates a RAG pipeline using RAGAS-inspired metrics.

    Usage::

        evaluator = RAGASEvaluator()
        evaluator.add_sample(question="...", generated="...", context=[...])
        report = evaluator.evaluate()
        print(f"Composite Score: {report.composite_score:.2f}")
    """

    def __init__(self):
        self._samples: List[EvalSample] = []

    def add_sample(
        self,
        question: str,
        generated_answer: str,
        context_docs: List[str],
        hallucinated_claims: List[str] | None = None,
        ground_truth: str = "",
    ) -> None:
        """Add an evaluation sample."""
        claims = hallucinated_claims or []

        # Compute individual metrics
        context_str = "\n\n".join(context_docs)
        faithfulness = compute_faithfulness(generated_answer, context_str, claims)

        sample = EvalSample(
            question=question,
            ground_truth_answer=ground_truth,
            generated_answer=generated_answer,
            context_docs=context_docs,
            hallucinated_claims=claims,
            metrics={
                "faithfulness": faithfulness,
            },
        )
        self._samples.append(sample)

    def evaluate(self) -> EvalReport:
        """Compute aggregate metrics across all samples."""
        report = EvalReport(samples=self._samples)
        report.compute()
        return report

    def save_report(self, filepath: str) -> None:
        """Save the evaluation report as JSON."""
        report = self.evaluate()
        data = {
            "total_samples": report.total_samples,
            "avg_faithfulness": report.avg_faithfulness,
            "avg_context_precision": report.avg_context_precision,
            "avg_context_recall": report.avg_context_recall,
            "avg_answer_relevancy": report.avg_answer_relevancy,
            "composite_score": report.composite_score,
            "samples": [
                {
                    "question": s.question,
                    "generated_answer": s.generated_answer[:500],
                    "metrics": s.metrics,
                    "hallucinated_claims": s.hallucinated_claims,
                }
                for s in self._samples
            ],
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to %s", filepath)


def generate_self_play_questions(
    documents: List[str],
    num_questions: int = 10,
) -> List[str]:
    """Generate test questions from documents using the LLM.

    This creates a self-play evaluation dataset: the LLM reads documents
    and generates questions that can be answered from them.

    Args:
        documents: List of document texts to generate questions from.
        num_questions: Number of questions to generate.

    Returns:
        List of generated questions.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.groq_model,
        temperature=0.7,
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
    )

    context = "\n\n".join(documents[:5])  # Sample from first 5 docs
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a test-set generator for a RAG system. Given document excerpts, "
         "generate {num} diverse, factual questions that can be answered from the provided text. "
         "Output ONLY the numbered questions, one per line. "
         "Make questions vary in complexity: fact lookup, comparison, summarisation, reasoning."),
        ("human", "Documents:\n{context}\n\nGenerate {num} questions:"),
    ])

    chain = prompt | llm
    response = chain.invoke({"context": context, "num": num_questions})

    questions = [
        q.strip().lstrip("0123456789. )- ")
        for q in response.content.strip().split("\n")
        if q.strip() and len(q.strip()) > 10
    ]
    logger.info("Generated %d self-play questions.", len(questions))
    return questions[:num_questions]


# -- CLI entry point --
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RAGAS Evaluation Framework for SentinelRAG v2.1")
    print("=" * 50)
    print()
    print("Usage:")
    print("  from evaluation.ragas_eval import RAGASEvaluator")
    print("  evaluator = RAGASEvaluator()")
    print("  evaluator.add_sample(...)")
    print("  report = evaluator.evaluate()")
    print("  evaluator.save_report('eval_report.json')")
