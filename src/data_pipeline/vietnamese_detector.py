"""Vietnamese language detection module."""

import re
from pathlib import Path

import fasttext


class VietnameseDetector:
    """Detect Vietnamese text using FastText language identification."""

    VIETNAMESE_CHARS = re.compile(
        r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
        r"ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]"
    )

    def __init__(self, model_path: str | None = None, threshold: float = 0.7):
        """Initialize the Vietnamese detector.

        Args:
            model_path: Path to FastText language identification model.
                       If None, downloads the model automatically.
            threshold: Confidence threshold for language detection.
        """
        self.threshold = threshold
        self._model = None
        self._model_path = model_path

    @property
    def model(self):
        """Lazy load the FastText model."""
        if self._model is None:
            if self._model_path and Path(self._model_path).exists():
                self._model = fasttext.load_model(self._model_path)
            else:
                # Will need to download lid.176.bin
                raise ValueError(
                    "FastText model not found. Download from: "
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
                )
        return self._model

    def detect(self, text: str) -> tuple[str, float]:
        """Detect the language of the given text.

        Args:
            text: Input text to classify.

        Returns:
            Tuple of (language_code, confidence_score).
        """
        # Clean text for detection
        text = text.replace("\n", " ").strip()
        if not text:
            return ("unknown", 0.0)

        predictions = self.model.predict(text, k=1)
        lang = predictions[0][0].replace("__label__", "")
        confidence = float(predictions[1][0])

        return (lang, confidence)

    def is_vietnamese(self, text: str) -> bool:
        """Check if text is Vietnamese with sufficient confidence.

        Args:
            text: Input text to check.

        Returns:
            True if text is Vietnamese with confidence above threshold.
        """
        lang, confidence = self.detect(text)
        return lang == "vi" and confidence >= self.threshold

    def has_vietnamese_chars(self, text: str) -> bool:
        """Quick check for Vietnamese-specific characters.

        Args:
            text: Input text to check.

        Returns:
            True if text contains Vietnamese diacritical characters.
        """
        return bool(self.VIETNAMESE_CHARS.search(text))

    def filter_vietnamese(self, texts: list[str], use_quick_check: bool = True) -> list[str]:
        """Filter a list of texts to keep only Vietnamese.

        Args:
            texts: List of texts to filter.
            use_quick_check: If True, first checks for Vietnamese chars
                           before running full detection.

        Returns:
            List of texts detected as Vietnamese.
        """
        result = []
        for text in texts:
            if use_quick_check and not self.has_vietnamese_chars(text):
                continue
            if self.is_vietnamese(text):
                result.append(text)
        return result
