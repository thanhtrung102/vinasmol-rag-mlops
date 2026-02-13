"""Integration test configuration with mocked ML models.

Patches heavy model downloads (SentenceTransformer, PhoGPT-4B) to allow
integration tests to run in CI without downloading multi-GB models.
Qdrant and Redis remain real services.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

EMBEDDING_DIM = 384


@pytest.fixture(autouse=True)
def mock_ml_models():
    """Mock heavy ML model loading for CI environment."""
    # Mock SentenceTransformer to avoid downloading embedding model
    mock_encoder = MagicMock()
    mock_encoder.get_sentence_embedding_dimension.return_value = EMBEDDING_DIM
    mock_encoder.encode.side_effect = lambda x, **_kwargs: (
        np.random.rand(len(x), EMBEDDING_DIM).astype(np.float32)
        if isinstance(x, list)
        else np.random.rand(EMBEDDING_DIM).astype(np.float32)
    )

    # Mock transformers to avoid downloading LLM
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()

    with (
        patch("src.rag.retriever.SentenceTransformer", return_value=mock_encoder),
        patch(
            "src.rag.generator.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "src.rag.generator.AutoModelForCausalLM.from_pretrained",
            return_value=mock_model,
        ),
        patch("src.rag.generator.torch.cuda.is_available", return_value=False),
    ):
        # Reset the global pipeline state before each test
        import src.api.main as main_module

        main_module.rag_pipeline = None

        def patched_generate(question, context_docs, **kwargs):
            from src.rag.generator import GenerationResult

            return GenerationResult(
                answer=f"Mocked answer for: {question}",
                context_used=context_docs,
                metadata={"mocked": True},
            )

        # Patch RAGGenerator.generate
        with patch(
            "src.rag.generator.RAGGenerator.generate", side_effect=patched_generate
        ):
            yield
