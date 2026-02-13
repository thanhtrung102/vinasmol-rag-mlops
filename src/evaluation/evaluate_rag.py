"""RAG evaluation using Ragas and custom metrics."""

from dataclasses import dataclass
from typing import Any

import mlflow
from datasets import Dataset


@dataclass
class RAGEvalResult:
    """Results from RAG evaluation."""

    faithfulness: float
    answer_relevance: float
    context_relevance: float
    context_recall: float
    hallucination_score: float
    overall_score: float
    details: dict[str, Any]


class RAGEvaluator:
    """Evaluate RAG system using Ragas metrics."""

    def __init__(
        self,
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """Initialize the evaluator.

        Args:
            llm_model: LLM for evaluation (Ragas requires OpenAI by default).
            embedding_model: Embedding model for similarity metrics.
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self._metrics = None

    def _load_metrics(self):
        """Lazy load Ragas metrics."""
        if self._metrics is None:
            try:
                # Try new import path (ragas >= 0.2)
                from ragas.metrics.collections import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
            except ImportError:
                # Fallback to old import path
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )

            # Instantiate metrics (Ragas requires initialized objects)
            self._metrics = {
                "faithfulness": faithfulness(),
                "answer_relevancy": answer_relevancy(),
                "context_precision": context_precision(),
                "context_recall": context_recall(),
            }
        return self._metrics

    def prepare_dataset(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str] | None = None,
    ) -> Dataset:
        """Prepare dataset for Ragas evaluation.

        Args:
            questions: List of questions.
            answers: List of generated answers.
            contexts: List of retrieved context lists.
            ground_truths: Optional list of ground truth answers.

        Returns:
            Dataset formatted for Ragas.
        """
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths
            data["reference"] = ground_truths  # Ragas context_precision requires 'reference'

        return Dataset.from_dict(data)

    def evaluate(
        self,
        dataset: Dataset,
        metrics: list[str] | None = None,
    ) -> RAGEvalResult:
        """Run Ragas evaluation on the dataset.

        Args:
            dataset: Prepared evaluation dataset.
            metrics: List of metric names to compute. If None, computes all.

        Returns:
            Evaluation results.
        """
        from ragas import evaluate as ragas_evaluate

        available_metrics = self._load_metrics()

        if metrics is None:
            metrics_to_use = list(available_metrics.values())
        else:
            metrics_to_use = [available_metrics[m] for m in metrics if m in available_metrics]

        # Run evaluation
        results = ragas_evaluate(
            dataset=dataset,
            metrics=metrics_to_use,
        )

        # Extract scores
        scores = results.to_pandas().mean().to_dict()

        # Calculate hallucination score (inverse of faithfulness)
        faithfulness_score = scores.get("faithfulness", 0.5)
        hallucination_score = 1.0 - faithfulness_score

        # Calculate overall score
        overall = sum(scores.values()) / len(scores) if scores else 0.0

        return RAGEvalResult(
            faithfulness=faithfulness_score,
            answer_relevance=scores.get("answer_relevancy", 0.0),
            context_relevance=scores.get("context_precision", 0.0),
            context_recall=scores.get("context_recall", 0.0),
            hallucination_score=hallucination_score,
            overall_score=overall,
            details=scores,
        )

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGEvalResult:
        """Evaluate a single RAG response.

        Args:
            question: The question asked.
            answer: The generated answer.
            contexts: Retrieved context documents.
            ground_truth: Optional ground truth answer.

        Returns:
            Evaluation results for this query.
        """
        dataset = self.prepare_dataset(
            questions=[question],
            answers=[answer],
            contexts=[contexts],
            ground_truths=[ground_truth] if ground_truth else None,
        )

        return self.evaluate(dataset)


class HallucinationDetector:
    """Detect hallucinations in RAG responses."""

    def __init__(self, threshold: float = 0.5):
        """Initialize the detector.

        Args:
            threshold: Score below which response is considered hallucinated.
        """
        self.threshold = threshold
        self.evaluator = RAGEvaluator()

    def detect(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Detect if the answer contains hallucinations.

        Args:
            question: The question asked.
            answer: The generated answer.
            contexts: Retrieved context documents.

        Returns:
            Detection result with score and verdict.
        """
        result = self.evaluator.evaluate_single(
            question=question,
            answer=answer,
            contexts=contexts,
        )

        is_hallucinated = result.hallucination_score > self.threshold

        return {
            "is_hallucinated": is_hallucinated,
            "hallucination_score": result.hallucination_score,
            "faithfulness_score": result.faithfulness,
            "threshold": self.threshold,
            "verdict": "HALLUCINATED" if is_hallucinated else "GROUNDED",
        }


def log_evaluation_to_mlflow(result: RAGEvalResult, run_name: str = "rag_evaluation") -> None:
    """Log evaluation results to MLflow.

    Args:
        result: RAG evaluation results.
        run_name: Name for the MLflow run.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "faithfulness": result.faithfulness,
            "answer_relevance": result.answer_relevance,
            "context_relevance": result.context_relevance,
            "context_recall": result.context_recall,
            "hallucination_score": result.hallucination_score,
            "overall_score": result.overall_score,
        })

        mlflow.log_dict(result.details, "evaluation_details.json")


def main():
    """Main evaluation entrypoint."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate RAG system")
    parser.add_argument("--input", type=str, required=True, help="Path to evaluation data JSON")
    parser.add_argument("--output", type=str, help="Path to save results")
    args = parser.parse_args()

    # Load evaluation data
    with open(args.input) as f:
        eval_data = json.load(f)

    evaluator = RAGEvaluator()
    dataset = evaluator.prepare_dataset(
        questions=eval_data["questions"],
        answers=eval_data["answers"],
        contexts=eval_data["contexts"],
        ground_truths=eval_data.get("ground_truths"),
    )

    result = evaluator.evaluate(dataset)

    # Log to MLflow
    log_evaluation_to_mlflow(result)

    # Print results
    print(f"\n{'='*50}")
    print("RAG Evaluation Results")
    print(f"{'='*50}")
    print(f"Faithfulness:      {result.faithfulness:.3f}")
    print(f"Answer Relevance:  {result.answer_relevance:.3f}")
    print(f"Context Relevance: {result.context_relevance:.3f}")
    print(f"Context Recall:    {result.context_recall:.3f}")
    print(f"Hallucination:     {result.hallucination_score:.3f}")
    print(f"Overall Score:     {result.overall_score:.3f}")
    print(f"{'='*50}\n")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "faithfulness": result.faithfulness,
                "answer_relevance": result.answer_relevance,
                "context_relevance": result.context_relevance,
                "context_recall": result.context_recall,
                "hallucination_score": result.hallucination_score,
                "overall_score": result.overall_score,
                "details": result.details,
            }, f, indent=2)


if __name__ == "__main__":
    main()
