"""Unit tests for the RAG retriever."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.rag.retriever import QdrantRetriever, RetrievedDocument


class TestQdrantRetriever:
    """Tests for QdrantRetriever class."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create a mock Qdrant client."""
        with patch("src.rag.retriever.QdrantClient") as mock_client:
            yield mock_client

    @pytest.fixture
    def mock_encoder(self):
        """Create a mock sentence transformer."""
        with patch("src.rag.retriever.SentenceTransformer") as mock_st:
            mock_instance = MagicMock()
            mock_instance.get_sentence_embedding_dimension.return_value = 384
            mock_instance.encode.return_value = np.array([0.1] * 384)
            mock_st.return_value = mock_instance
            yield mock_st

    def test_initialization(self, mock_qdrant_client, mock_encoder):
        """Test retriever initialization."""
        retriever = QdrantRetriever(
            collection_name="test_collection",
            host="localhost",
            port=6333,
        )

        assert retriever.collection_name == "test_collection"
        mock_qdrant_client.assert_called_once_with(host="localhost", port=6333)

    def test_create_collection_new(self, mock_qdrant_client, mock_encoder):
        """Test creating a new collection."""
        mock_client_instance = MagicMock()
        mock_client_instance.get_collections.return_value.collections = []
        mock_qdrant_client.return_value = mock_client_instance

        retriever = QdrantRetriever(collection_name="new_collection")
        retriever.create_collection()

        mock_client_instance.create_collection.assert_called_once()

    def test_create_collection_exists(self, mock_qdrant_client, mock_encoder):
        """Test that existing collection is not recreated."""
        mock_collection = MagicMock()
        mock_collection.name = "existing_collection"

        mock_client_instance = MagicMock()
        mock_client_instance.get_collections.return_value.collections = [mock_collection]
        mock_qdrant_client.return_value = mock_client_instance

        retriever = QdrantRetriever(collection_name="existing_collection")
        retriever.create_collection(recreate=False)

        mock_client_instance.create_collection.assert_not_called()

    def test_retrieve_returns_documents(self, mock_qdrant_client, mock_encoder):
        """Test document retrieval."""
        # Mock search results
        mock_hit = MagicMock()
        mock_hit.payload = {"content": "Test document", "source": "test"}
        mock_hit.score = 0.85

        mock_response = MagicMock()
        mock_response.points = [mock_hit]

        mock_client_instance = MagicMock()
        mock_client_instance.query_points.return_value = mock_response
        mock_qdrant_client.return_value = mock_client_instance

        retriever = QdrantRetriever(collection_name="test")
        results = retriever.retrieve("test query", top_k=5)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedDocument)
        assert results[0].content == "Test document"
        assert results[0].score == 0.85

    def test_retrieve_empty_results(self, mock_qdrant_client, mock_encoder):
        """Test retrieval with no matching documents."""
        mock_response = MagicMock()
        mock_response.points = []

        mock_client_instance = MagicMock()
        mock_client_instance.query_points.return_value = mock_response
        mock_qdrant_client.return_value = mock_client_instance

        retriever = QdrantRetriever(collection_name="test")
        results = retriever.retrieve("no match query")

        assert len(results) == 0


class TestRetrievedDocument:
    """Tests for RetrievedDocument dataclass."""

    def test_creation(self):
        """Test RetrievedDocument creation."""
        doc = RetrievedDocument(
            content="Test content",
            score=0.9,
            metadata={"source": "test.txt"},
        )

        assert doc.content == "Test content"
        assert doc.score == 0.9
        assert doc.metadata["source"] == "test.txt"

    def test_empty_metadata(self):
        """Test RetrievedDocument with empty metadata."""
        doc = RetrievedDocument(
            content="Content",
            score=0.5,
            metadata={},
        )

        assert doc.metadata == {}
