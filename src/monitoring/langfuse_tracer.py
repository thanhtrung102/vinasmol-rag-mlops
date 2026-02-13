"""LangFuse integration for LLM tracing and observability.

Provides decorators and context managers for tracing RAG pipeline operations,
including retrieval, generation, and reranking steps.
"""

import logging
import os
from typing import Any

from langfuse import Langfuse

logger = logging.getLogger(__name__)


class LangFuseTracer:
    """Wrapper for LangFuse tracing functionality."""

    def __init__(
        self,
        public_key: str | None = None,
        secret_key: str | None = None,
        host: str | None = None,
        enabled: bool = True,
    ):
        """Initialize LangFuse tracer.

        Args:
            public_key: LangFuse public key (or from LANGFUSE_PUBLIC_KEY env)
            secret_key: LangFuse secret key (or from LANGFUSE_SECRET_KEY env)
            host: LangFuse host URL (or from LANGFUSE_HOST env)
            enabled: Whether tracing is enabled
        """
        self.enabled = enabled and bool(public_key or os.getenv("LANGFUSE_PUBLIC_KEY"))

        if self.enabled:
            try:
                self.client = Langfuse(
                    public_key=public_key or os.getenv("LANGFUSE_PUBLIC_KEY"),
                    secret_key=secret_key or os.getenv("LANGFUSE_SECRET_KEY"),
                    host=host or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                )
                logger.info("LangFuse tracing initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize LangFuse: {e}")
                self.enabled = False
                self.client = None
        else:
            logger.info("LangFuse tracing disabled (no API keys)")
            self.client = None

    def trace_rag_query(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        latency_ms: float,
        cached: bool = False,
        metadata: dict[str, Any] | None = None,
    ):
        """Trace a complete RAG query.

        Args:
            question: User question
            answer: Generated answer
            sources: Retrieved sources
            latency_ms: Query latency in milliseconds
            cached: Whether result was cached
            metadata: Additional metadata
        """
        if not self.enabled or not self.client:
            return

        try:
            trace = self.client.trace(
                name="rag_query",
                input=question,
                output=answer,
                metadata={
                    "latency_ms": latency_ms,
                    "cached": cached,
                    "num_sources": len(sources),
                    "source_scores": [s.get("score", 0) for s in sources],
                    **(metadata or {}),
                },
            )
            trace.update()

        except Exception as e:
            logger.error(f"Error tracing RAG query: {e}")

    def flush(self):
        """Flush pending traces to LangFuse."""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                logger.error(f"Error flushing LangFuse traces: {e}")


# Global tracer instance
_tracer: LangFuseTracer | None = None


def get_tracer() -> LangFuseTracer:
    """Get or create the global LangFuse tracer instance.

    Returns:
        LangFuseTracer instance
    """
    global _tracer
    if _tracer is None:
        _tracer = LangFuseTracer()
    return _tracer


def initialize_tracer(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool = True,
) -> LangFuseTracer:
    """Initialize the global LangFuse tracer.

    Args:
        public_key: LangFuse public key
        secret_key: LangFuse secret key
        host: LangFuse host URL
        enabled: Whether tracing is enabled

    Returns:
        Initialized LangFuseTracer
    """
    global _tracer
    _tracer = LangFuseTracer(
        public_key=public_key,
        secret_key=secret_key,
        host=host,
        enabled=enabled,
    )
    return _tracer
