"""Text processing utilities for Vietnamese content."""

import re
import unicodedata
from dataclasses import dataclass


@dataclass
class ProcessedText:
    """Container for processed text with metadata."""

    text: str
    word_count: int
    char_count: int
    is_valid: bool
    quality_score: float


class TextProcessor:
    """Process and clean Vietnamese text for LLM training and RAG."""

    # Minimum quality thresholds
    MIN_WORDS = 20
    MIN_CHARS = 100
    MAX_REPETITION_RATIO = 0.3

    def __init__(
        self,
        min_words: int = MIN_WORDS,
        min_chars: int = MIN_CHARS,
        normalize_unicode: bool = True,
    ):
        """Initialize the text processor.

        Args:
            min_words: Minimum word count for valid text.
            min_chars: Minimum character count for valid text.
            normalize_unicode: Whether to normalize Unicode characters.
        """
        self.min_words = min_words
        self.min_chars = min_chars
        self.normalize_unicode = normalize_unicode

    def clean(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text.
        """
        if not text:
            return ""

        # Normalize Unicode (NFC for Vietnamese)
        if self.normalize_unicode:
            text = unicodedata.normalize("NFC", text)

        # Remove control characters except newlines
        text = re.sub(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "", text)

        # Clean up extra spaces
        text = re.sub(r" +", " ", text)
        text = text.strip()

        return text

    def calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score for the text.

        Args:
            text: Input text to score.

        Returns:
            Quality score between 0 and 1.
        """
        if not text:
            return 0.0

        score = 1.0
        words = text.split()
        word_count = len(words)

        # Penalize very short texts
        if word_count < self.min_words:
            score *= word_count / self.min_words

        # Check for excessive repetition
        if word_count > 0:
            unique_words = set(words)
            repetition_ratio = 1 - (len(unique_words) / word_count)
            if repetition_ratio > self.MAX_REPETITION_RATIO:
                score *= 1 - (repetition_ratio - self.MAX_REPETITION_RATIO)

        # Check for balanced punctuation
        chars = set(text)
        if "(" in chars or ")" in chars:
            open_count = text.count("(")
            close_count = text.count(")")
            if open_count != close_count:
                score *= 0.9

        # Penalize texts with too many numbers
        num_digits = sum(c.isdigit() for c in text)
        digit_ratio = num_digits / len(text) if text else 0
        if digit_ratio > 0.3:
            score *= 0.8

        return max(0.0, min(1.0, score))

    def process(self, text: str) -> ProcessedText:
        """Process text and return with metadata.

        Args:
            text: Raw input text.

        Returns:
            ProcessedText with cleaned text and metadata.
        """
        cleaned = self.clean(text)
        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)
        quality_score = self.calculate_quality_score(cleaned)

        is_valid = (
            word_count >= self.min_words
            and char_count >= self.min_chars
            and quality_score >= 0.5
        )

        return ProcessedText(
            text=cleaned,
            word_count=word_count,
            char_count=char_count,
            is_valid=is_valid,
            quality_score=quality_score,
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> list[str]:
        """Split text into overlapping chunks for RAG.

        Args:
            text: Input text to chunk.
            chunk_size: Target size of each chunk in characters.
            overlap: Number of characters to overlap between chunks.

        Returns:
            List of text chunks.
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                for sep in [".", "!", "?", "\n"]:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start + chunk_size // 2:
                        end = last_sep + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - overlap

        return chunks
