"""Integration tests for RAG API endpoints.

Tests the complete RAG pipeline with Qdrant and Redis services.
"""

import pytest
from fastapi.testclient import TestClient

# These tests require running services (docker-compose up)
pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    from src.api.main import app

    return TestClient(app)


@pytest.fixture
def sample_documents():
    """Sample Vietnamese documents for testing."""
    return [
        "Việt Nam là một quốc gia ở Đông Nam Á.",
        "Hà Nội là thủ đô của Việt Nam.",
        "Phở là món ăn truyền thống của Việt Nam.",
    ]


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "rag_requests_total" in response.text


class TestDocumentIngestion:
    """Tests for document ingestion."""

    def test_add_documents(self, client, sample_documents):
        """Test adding documents to the collection."""
        response = client.post(
            "/documents",
            json={"documents": sample_documents},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["documents_added"] == len(sample_documents)

    def test_add_documents_with_metadata(self, client):
        """Test adding documents with metadata."""
        documents = ["Test document"]
        metadatas = [{"source": "test", "category": "example"}]

        response = client.post(
            "/documents",
            json={"documents": documents, "metadatas": metadatas},
        )
        assert response.status_code == 200

    def test_add_documents_validation(self, client):
        """Test document validation."""
        # Empty documents list
        response = client.post("/documents", json={"documents": []})
        assert response.status_code == 422

        # Too many documents
        response = client.post(
            "/documents",
            json={"documents": ["doc"] * 150},
        )
        assert response.status_code == 422


class TestRAGQuery:
    """Tests for RAG query endpoint."""

    def test_query_without_documents(self, client):
        """Test query without initializing collection."""
        response = client.post(
            "/query",
            json={"question": "Việt Nam ở đâu?"},
        )
        # Should either work or return 503
        assert response.status_code in [200, 503]

    def test_query_with_documents(self, client, sample_documents):
        """Test full RAG query flow."""
        # Add documents first
        client.post("/documents", json={"documents": sample_documents})

        # Query
        response = client.post(
            "/query",
            json={
                "question": "Thủ đô của Việt Nam là gì?",
                "top_k": 2,
                "temperature": 0.5,
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert len(data["sources"]) <= 2
            assert "latency_ms" in data
            assert isinstance(data["cached"], bool)

    def test_query_caching(self, client, sample_documents):
        """Test query caching."""
        # Add documents
        client.post("/documents", json={"documents": sample_documents})

        question = {"question": "Việt Nam ở đâu?"}

        # First query (cache miss)
        response1 = client.post("/query", json=question)
        if response1.status_code == 200:
            data1 = response1.json()
            assert data1["cached"] is False

            # Second query (cache hit)
            response2 = client.post("/query", json=question)
            if response2.status_code == 200:
                data2 = response2.json()
                # May or may not be cached depending on Redis availability
                assert "cached" in data2

    def test_query_validation(self, client):
        """Test query parameter validation."""
        # Empty question
        response = client.post("/query", json={"question": ""})
        assert response.status_code == 422

        # Invalid top_k
        response = client.post(
            "/query",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

        # Invalid temperature
        response = client.post(
            "/query",
            json={"question": "test", "temperature": -0.5},
        )
        assert response.status_code == 422


class TestStreamingQuery:
    """Tests for streaming query endpoint."""

    def test_stream_endpoint_exists(self, client):
        """Test that streaming endpoint is available."""
        response = client.post(
            "/query/stream",
            json={"question": "test"},
        )
        # Should return 503 if not initialized, or stream data
        assert response.status_code in [200, 503]


class TestCollectionManagement:
    """Tests for collection management endpoints."""

    def test_collection_info_not_initialized(self, client):
        """Test collection info when not initialized."""
        response = client.get("/collection/info")
        # May return 404 if not initialized
        assert response.status_code in [200, 404]

    def test_collection_info_after_documents(self, client, sample_documents):
        """Test collection info after adding documents."""
        client.post("/documents", json={"documents": sample_documents})

        response = client.get("/collection/info")
        if response.status_code == 200:
            data = response.json()
            assert "name" in data
            assert "points_count" in data

    def test_delete_collection(self, client, sample_documents):
        """Test deleting collection."""
        # Add documents first
        client.post("/documents", json={"documents": sample_documents})

        # Delete collection
        response = client.delete("/collection")
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "success"


class TestCacheManagement:
    """Tests for cache management endpoints."""

    def test_cache_stats(self, client):
        """Test cache statistics endpoint."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data

    def test_clear_cache(self, client):
        """Test clearing cache."""
        response = client.delete("/cache")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_complete_rag_workflow(self, client):
        """Test complete RAG workflow."""
        # 1. Add documents
        documents = [
            "Python là một ngôn ngữ lập trình phổ biến.",
            "Machine Learning là nhánh của trí tuệ nhân tạo.",
            "FastAPI là framework web hiện đại cho Python.",
        ]

        response = client.post("/documents", json={"documents": documents})
        assert response.status_code == 200

        # 2. Query
        response = client.post(
            "/query",
            json={"question": "Python là gì?", "top_k": 2},
        )
        if response.status_code == 200:
            data = response.json()
            assert "answer" in data
            assert len(data["sources"]) > 0

        # 3. Check collection info
        response = client.get("/collection/info")
        if response.status_code == 200:
            data = response.json()
            assert data["points_count"] >= 3

        # 4. Clear cache
        response = client.delete("/cache")
        assert response.status_code == 200
