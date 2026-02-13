"""Pytest configuration and shared fixtures."""

import os

import pytest


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_vietnamese_texts():
    """Sample Vietnamese texts for testing."""
    return [
        "Xin chào, tôi là một trợ lý ảo được phát triển để hỗ trợ người dùng.",
        "Việt Nam là một quốc gia nằm ở Đông Nam Á với lịch sử văn hóa phong phú.",
        "Trí tuệ nhân tạo đang phát triển nhanh chóng và có nhiều ứng dụng trong cuộc sống.",
        "Hà Nội là thủ đô của Việt Nam, nổi tiếng với các di tích lịch sử và ẩm thực đường phố.",
        "Machine learning là một nhánh của trí tuệ nhân tạo giúp máy tính học từ dữ liệu.",
    ]


@pytest.fixture
def sample_qa_pairs():
    """Sample question-answer pairs for RAG testing."""
    return [
        {
            "question": "Việt Nam nằm ở đâu?",
            "answer": "Việt Nam nằm ở Đông Nam Á.",
            "context": ["Việt Nam là một quốc gia nằm ở Đông Nam Á với lịch sử văn hóa phong phú."],
        },
        {
            "question": "Hà Nội nổi tiếng với điều gì?",
            "answer": "Hà Nội nổi tiếng với các di tích lịch sử và ẩm thực đường phố.",
            "context": ["Hà Nội là thủ đô của Việt Nam, nổi tiếng với các di tích lịch sử và ẩm thực đường phố."],
        },
    ]


@pytest.fixture(autouse=True)
def set_test_env():
    """Set environment variables for testing."""
    os.environ["TESTING"] = "true"
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///test_mlflow.db"
    yield
    os.environ.pop("TESTING", None)


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "evaluation: marks tests as model evaluation tests")
