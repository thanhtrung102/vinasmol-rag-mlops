"""Evaluation module for RAG and LLM quality assessment.

Provides comprehensive evaluation tools:
- Ragas metrics integration
- Hallucination detection
- Vietnamese benchmark dataset
"""

from src.evaluation.evaluate_rag import RAGEvalResult, RAGEvaluator
from src.evaluation.hallucination_detector import (
    AdvancedHallucinationDetector,
    HallucinationDetectionResult,
    SimpleHallucinationDetector,
)
from src.evaluation.vietnamese_benchmark import BenchmarkQuestion, VietnameseBenchmark

__all__ = [
    "RAGEvaluator",
    "RAGEvalResult",
    "AdvancedHallucinationDetector",
    "SimpleHallucinationDetector",
    "HallucinationDetectionResult",
    "VietnameseBenchmark",
    "BenchmarkQuestion",
]
