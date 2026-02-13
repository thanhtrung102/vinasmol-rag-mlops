"""RAG system configuration management.

Provides structured configuration for retrieval, generation, reranking, and caching.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RetrieverConfig:
    """Retriever configuration."""

    collection_name: str = "vietnamese_docs"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    host: str = "localhost"
    port: int = 6333
    top_k: int = 5
    score_threshold: float = 0.5


@dataclass
class GeneratorConfig:
    """Generator configuration."""

    model_name: str = "vinai/PhoGPT-4B-Chat"
    device: str | None = None
    load_in_8bit: bool = True
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    prompt_template: str | None = None


@dataclass
class RerankerConfig:
    """Reranker configuration."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_hybrid: bool = True
    hybrid_alpha: float = 0.5
    top_k: int | None = None


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = True
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    ttl: int = 3600


@dataclass
class RAGConfig:
    """Complete RAG system configuration."""

    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RAGConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated RAGConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raw_config = {}

        return cls._from_dict(raw_config)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary."""
        return cls(
            retriever=cls._parse_section(data.get("retriever", {}), RetrieverConfig),
            generator=cls._parse_section(data.get("generator", {}), GeneratorConfig),
            reranker=cls._parse_section(data.get("reranker", {}), RerankerConfig),
            cache=cls._parse_section(data.get("cache", {}), CacheConfig),
        )

    @staticmethod
    def _parse_section(data: dict[str, Any], dataclass_type: type) -> Any:
        """Parse a config section into a dataclass."""
        if not data:
            return dataclass_type()

        valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return dataclass_type(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "retriever": {
                "collection_name": self.retriever.collection_name,
                "embedding_model": self.retriever.embedding_model,
                "top_k": self.retriever.top_k,
            },
            "generator": {
                "model_name": self.generator.model_name,
                "load_in_8bit": self.generator.load_in_8bit,
                "temperature": self.generator.temperature,
            },
            "reranker": {
                "enabled": self.reranker.enabled,
                "use_hybrid": self.reranker.use_hybrid,
            },
            "cache": {
                "enabled": self.cache.enabled,
                "ttl": self.cache.ttl,
            },
        }

    def validate(self) -> list[str]:
        """Validate configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        issues = []

        if self.retriever.top_k < 1:
            issues.append("Retriever top_k must be at least 1")

        if not (0 <= self.retriever.score_threshold <= 1):
            issues.append("Retriever score_threshold must be between 0 and 1")

        if self.generator.temperature < 0:
            issues.append("Generator temperature must be non-negative")

        if self.generator.max_new_tokens < 1:
            issues.append("Generator max_new_tokens must be at least 1")

        if self.reranker.use_hybrid and not (0 <= self.reranker.hybrid_alpha <= 1):
            issues.append("Reranker hybrid_alpha must be between 0 and 1")

        if self.cache.ttl < 0:
            issues.append("Cache TTL must be non-negative")

        return issues
