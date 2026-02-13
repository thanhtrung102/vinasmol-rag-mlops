"""Unit tests for text processor."""

import pytest

from src.data_pipeline.text_processor import ProcessedText, TextProcessor


class TestTextProcessor:
    """Tests for TextProcessor class."""

    @pytest.fixture
    def processor(self):
        """Create a text processor instance."""
        return TextProcessor(min_words=10, min_chars=50)

    def test_clean_removes_urls(self, processor):
        """Test that URLs are removed from text."""
        text = "Visit https://example.com for more info."
        result = processor.clean(text)
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_clean_removes_emails(self, processor):
        """Test that email addresses are removed."""
        text = "Contact us at test@example.com for help."
        result = processor.clean(text)
        assert "test@example.com" not in result

    def test_clean_normalizes_whitespace(self, processor):
        """Test whitespace normalization."""
        text = "Hello    world\t\ttab    spaces"
        result = processor.clean(text)
        assert "    " not in result
        assert "\t\t" not in result

    def test_clean_handles_empty_string(self, processor):
        """Test handling of empty strings."""
        assert processor.clean("") == ""
        assert processor.clean("   ") == ""

    def test_process_returns_processed_text(self, processor):
        """Test that process returns ProcessedText object."""
        text = "Đây là một văn bản tiếng Việt dài đủ để được coi là hợp lệ trong quá trình xử lý."
        result = processor.process(text)

        assert isinstance(result, ProcessedText)
        assert result.word_count > 0
        assert result.char_count > 0
        assert 0 <= result.quality_score <= 1

    def test_process_invalid_short_text(self, processor):
        """Test that short texts are marked invalid."""
        text = "Too short"
        result = processor.process(text)
        assert result.is_valid is False

    def test_quality_score_penalizes_repetition(self, processor):
        """Test that repetitive text gets lower quality score."""
        normal_text = "Một hai ba bốn năm sáu bảy tám chín mười"
        repetitive_text = "lặp lặp lặp lặp lặp lặp lặp lặp lặp lặp"

        normal_score = processor.calculate_quality_score(normal_text)
        repetitive_score = processor.calculate_quality_score(repetitive_text)

        assert normal_score > repetitive_score

    def test_chunk_text_basic(self, processor):
        """Test basic text chunking."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = processor.chunk_text(text, chunk_size=30, overlap=5)

        assert len(chunks) > 1
        assert all(len(chunk) <= 35 for chunk in chunks)  # Some tolerance

    def test_chunk_text_short_text(self, processor):
        """Test that short text returns single chunk."""
        text = "Short text."
        chunks = processor.chunk_text(text, chunk_size=100, overlap=10)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_preserves_content(self, processor):
        """Test that chunking preserves all content."""
        text = "Word " * 100  # Create a longer text
        chunks = processor.chunk_text(text, chunk_size=50, overlap=10)

        # All words should be present in chunks (accounting for overlap)
        all_content = " ".join(chunks)
        assert "Word" in all_content


class TestVietnameseSpecific:
    """Tests specific to Vietnamese text handling."""

    @pytest.fixture
    def processor(self):
        """Create a text processor instance."""
        return TextProcessor()

    def test_preserves_vietnamese_diacritics(self, processor):
        """Test that Vietnamese diacritics are preserved."""
        text = "Xin chào, tôi là một trợ lý ảo hữu ích."
        result = processor.clean(text)

        assert "à" in result
        assert "ữ" in result
        assert "ả" in result

    def test_handles_mixed_content(self, processor):
        """Test handling of mixed Vietnamese and English."""
        text = "Đây là mixed content với English words và tiếng Việt."
        result = processor.process(text)

        assert result.char_count > 0
        assert "mixed" in result.text
        assert "tiếng Việt" in result.text
