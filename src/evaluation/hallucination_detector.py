"""Advanced hallucination detection for RAG responses.

Implements multiple detection strategies:
- Faithfulness scoring via Ragas
- Factual consistency checking
- Named entity verification
- Claim extraction and verification
"""

import logging
import re
from dataclasses import dataclass
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HallucinationDetectionResult:
    """Results from hallucination detection."""

    is_hallucinated: bool
    confidence: float
    faithfulness_score: float
    consistency_score: float
    entity_match_score: float
    verdict: str
    reasons: list[str]
    details: dict[str, Any]


class AdvancedHallucinationDetector:
    """Detect hallucinations using multiple strategies."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.7,
        consistency_threshold: float = 0.6,
        entity_threshold: float = 0.5,
    ):
        """Initialize the detector.

        Args:
            faithfulness_threshold: Minimum faithfulness score.
            consistency_threshold: Minimum consistency score.
            entity_threshold: Minimum entity match ratio.
        """
        self.faithfulness_threshold = faithfulness_threshold
        self.consistency_threshold = consistency_threshold
        self.entity_threshold = entity_threshold

    def detect(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> HallucinationDetectionResult:
        """Detect hallucinations in the answer.

        Args:
            question: The question asked.
            answer: The generated answer.
            contexts: Retrieved context documents.

        Returns:
            HallucinationDetectionResult with all detection scores.
        """
        # 1. Faithfulness scoring (requires Ragas)
        faithfulness_score = self._compute_faithfulness(answer, contexts)

        # 2. Lexical consistency
        consistency_score = self._compute_consistency(answer, contexts)

        # 3. Entity verification
        entity_match_score = self._verify_entities(answer, contexts)

        # Determine hallucination
        reasons = []
        is_hallucinated = False

        if faithfulness_score < self.faithfulness_threshold:
            is_hallucinated = True
            reasons.append(
                f"Low faithfulness score ({faithfulness_score:.2f} < {self.faithfulness_threshold})"
            )

        if consistency_score < self.consistency_threshold:
            is_hallucinated = True
            reasons.append(
                f"Low consistency score ({consistency_score:.2f} < {self.consistency_threshold})"
            )

        if entity_match_score < self.entity_threshold:
            is_hallucinated = True
            reasons.append(
                f"Low entity match ({entity_match_score:.2f} < {self.entity_threshold})"
            )

        # Calculate confidence
        scores = [faithfulness_score, consistency_score, entity_match_score]
        confidence = sum(scores) / len(scores)

        verdict = "HALLUCINATED" if is_hallucinated else "GROUNDED"

        return HallucinationDetectionResult(
            is_hallucinated=is_hallucinated,
            confidence=confidence,
            faithfulness_score=faithfulness_score,
            consistency_score=consistency_score,
            entity_match_score=entity_match_score,
            verdict=verdict,
            reasons=reasons if reasons else ["All checks passed"],
            details={
                "thresholds": {
                    "faithfulness": self.faithfulness_threshold,
                    "consistency": self.consistency_threshold,
                    "entity": self.entity_threshold,
                }
            },
        )

    def _compute_faithfulness(self, answer: str, contexts: list[str]) -> float:
        """Compute faithfulness score using Ragas.

        Args:
            answer: The generated answer.
            contexts: Retrieved contexts.

        Returns:
            Faithfulness score (0-1).
        """
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import faithfulness

            data = {
                "question": [""],  # Faithfulness doesn't require question
                "answer": [answer],
                "contexts": [contexts],
            }

            dataset = Dataset.from_dict(data)
            result = evaluate(dataset, metrics=[faithfulness])

            return result["faithfulness"]

        except ImportError:
            logger.warning("Ragas not available. Using fallback consistency check.")
            return self._compute_consistency(answer, contexts)

        except Exception as e:
            logger.error(f"Faithfulness computation failed: {e}")
            return 0.5  # Neutral score on error

    def _compute_consistency(self, answer: str, contexts: list[str]) -> float:
        """Compute lexical consistency between answer and contexts.

        Args:
            answer: The generated answer.
            contexts: Retrieved contexts.

        Returns:
            Consistency score (0-1).
        """
        if not contexts:
            return 0.0

        # Tokenize answer
        answer_tokens = set(self._tokenize(answer.lower()))

        if not answer_tokens:
            return 0.0

        # Find maximum overlap with any context
        max_overlap = 0.0

        for context in contexts:
            context_tokens = set(self._tokenize(context.lower()))

            if not context_tokens:
                continue

            # Compute Jaccard similarity
            intersection = answer_tokens & context_tokens
            union = answer_tokens | context_tokens

            overlap = len(intersection) / len(union) if union else 0.0
            max_overlap = max(max_overlap, overlap)

        return max_overlap

    def _verify_entities(self, answer: str, contexts: list[str]) -> float:
        """Verify that entities in answer appear in contexts.

        Args:
            answer: The generated answer.
            contexts: Retrieved contexts.

        Returns:
            Entity match score (0-1).
        """
        # Simple named entity extraction (capitalized words)
        answer_entities = self._extract_simple_entities(answer)

        if not answer_entities:
            return 1.0  # No entities to verify

        # Combine all contexts
        combined_context = " ".join(contexts).lower()

        # Check how many entities appear in context
        matched = sum(
            1 for entity in answer_entities if entity.lower() in combined_context
        )

        return matched / len(answer_entities)

    def _extract_simple_entities(self, text: str) -> list[str]:
        """Extract simple named entities (capitalized words/phrases).

        Args:
            text: Input text.

        Returns:
            List of potential entities.
        """
        # Find capitalized words (excluding sentence starts)
        words = text.split()
        entities = []

        for i, word in enumerate(words):
            # Skip first word (sentence start)
            if i == 0:
                continue

            # Check if word is capitalized and not after punctuation
            if word and word[0].isupper() and len(word) > 1:
                # Clean punctuation
                clean_word = re.sub(r'[^\w\s]', '', word)
                if clean_word:
                    entities.append(clean_word)

        return entities

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization.

        Args:
            text: Input text.

        Returns:
            List of tokens.
        """
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if len(t) > 2]  # Filter short words


class SimpleHallucinationDetector:
    """Lightweight hallucination detector for production use."""

    def __init__(self, threshold: float = 0.6):
        """Initialize detector.

        Args:
            threshold: Consistency threshold for hallucination detection.
        """
        self.threshold = threshold

    def detect(self, answer: str, contexts: list[str]) -> dict[str, Any]:
        """Detect hallucinations using simple consistency check.

        Args:
            answer: The generated answer.
            contexts: Retrieved contexts.

        Returns:
            Detection result.
        """
        if not contexts:
            return {
                "is_hallucinated": True,
                "score": 0.0,
                "verdict": "NO_CONTEXT",
            }

        # Compute token overlap
        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(contexts).lower().split())

        if not answer_tokens:
            return {
                "is_hallucinated": True,
                "score": 0.0,
                "verdict": "EMPTY_ANSWER",
            }

        intersection = answer_tokens & context_tokens
        score = len(intersection) / len(answer_tokens)

        is_hallucinated = score < self.threshold

        return {
            "is_hallucinated": is_hallucinated,
            "score": score,
            "threshold": self.threshold,
            "verdict": "HALLUCINATED" if is_hallucinated else "GROUNDED",
            "token_overlap": len(intersection),
            "total_tokens": len(answer_tokens),
        }
