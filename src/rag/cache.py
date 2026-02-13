"""Redis caching layer for RAG responses.

Caches query results to reduce latency and LLM API costs for common questions.
"""

import hashlib
import json
import logging
from typing import Any

import redis

logger = logging.getLogger(__name__)


class RAGCache:
    """Redis-based cache for RAG query results."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        ttl: int = 3600,
        enabled: bool = True,
    ):
        """Initialize Redis cache.

        Args:
            host: Redis server host.
            port: Redis server port.
            db: Redis database number.
            ttl: Time-to-live for cached entries (seconds).
            enabled: Whether caching is enabled.
        """
        self.ttl = ttl
        self.enabled = enabled

        if enabled:
            try:
                self.client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                self.client.ping()
                logger.info(f"Redis cache connected: {host}:{port}")
            except (redis.ConnectionError, redis.TimeoutError) as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.enabled = False
                self.client = None
        else:
            self.client = None

    def _generate_key(self, query: str, **kwargs: Any) -> str:
        """Generate cache key from query and parameters.

        Args:
            query: The query text.
            **kwargs: Additional parameters (top_k, temperature, etc.).

        Returns:
            Cache key string.
        """
        # Create deterministic key from query and parameters
        params_str = json.dumps(kwargs, sort_keys=True)
        content = f"{query}:{params_str}"
        key_hash = hashlib.md5(content.encode()).hexdigest()
        return f"rag:query:{key_hash}"

    def get(self, query: str, **kwargs: Any) -> dict[str, Any] | None:
        """Retrieve cached result for a query.

        Args:
            query: The query text.
            **kwargs: Query parameters.

        Returns:
            Cached result dict or None if not found.
        """
        if not self.enabled or self.client is None:
            return None

        try:
            key = self._generate_key(query, **kwargs)
            cached = self.client.get(key)

            if cached:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return json.loads(cached)

            logger.debug(f"Cache miss for query: {query[:50]}...")
            return None

        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None

    def set(self, query: str, result: dict[str, Any], **kwargs: Any) -> bool:
        """Cache a query result.

        Args:
            query: The query text.
            result: The result dictionary to cache.
            **kwargs: Query parameters.

        Returns:
            True if cached successfully, False otherwise.
        """
        if not self.enabled or self.client is None:
            return False

        try:
            key = self._generate_key(query, **kwargs)
            serialized = json.dumps(result)
            self.client.setex(key, self.ttl, serialized)
            logger.debug(f"Cached result for query: {query[:50]}...")
            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def invalidate(self, pattern: str = "rag:query:*") -> int:
        """Invalidate cached entries matching pattern.

        Args:
            pattern: Redis key pattern to match.

        Returns:
            Number of keys deleted.
        """
        if not self.enabled or self.client is None:
            return 0

        try:
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        if not self.enabled or self.client is None:
            return {"enabled": False}

        try:
            info = self.client.info("stats")
            keys_count = len(self.client.keys("rag:query:*"))

            return {
                "enabled": True,
                "keys_count": keys_count,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0)
                    / max(
                        info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0),
                        1,
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}

    def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            self.client.close()
