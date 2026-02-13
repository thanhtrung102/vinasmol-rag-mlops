"""FastAPI application for Vietnamese RAG service.

Provides REST API endpoints for document ingestion, query processing,
and system monitoring with caching and streaming support.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.monitoring import get_tracer
from src.rag import RAGCache, RAGPipeline
from src.rag.config import RAGConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangFuse tracer
langfuse_tracer = get_tracer()

# Metrics
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total RAG requests",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "rag_request_latency_seconds",
    "RAG request latency",
    ["endpoint"],
)
RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "RAG retrieval scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
CACHE_HITS = Counter("rag_cache_hits_total", "Total cache hits")
CACHE_MISSES = Counter("rag_cache_misses_total", "Total cache misses")


# Request/Response models
class QueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)
    use_cache: bool = Field(default=True)


class DocumentRequest(BaseModel):
    """Document ingestion request."""

    documents: list[str] = Field(..., min_items=1, max_items=100)
    metadatas: list[dict[str, Any]] | None = None


class RetrievedDoc(BaseModel):
    """Retrieved document in response."""

    content: str
    score: float
    metadata: dict[str, Any]


class QueryResponse(BaseModel):
    """RAG query response."""

    question: str
    answer: str
    sources: list[RetrievedDoc]
    latency_ms: float
    cached: bool
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict[str, str]


class CollectionInfo(BaseModel):
    """Collection information."""

    name: str
    vectors_count: int
    points_count: int
    status: str


# Global state
rag_pipeline: RAGPipeline | None = None
rag_cache: RAGCache | None = None
config: RAGConfig | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifespan handler."""
    global rag_pipeline, rag_cache, config

    try:
        # Load configuration
        config = RAGConfig.from_yaml("configs/rag_config.yaml")
        logger.info("RAG configuration loaded")

        # Initialize cache
        if config.cache.enabled:
            rag_cache = RAGCache(
                host=config.cache.host,
                port=config.cache.port,
                db=config.cache.db,
                ttl=config.cache.ttl,
            )
            logger.info("Cache initialized")

    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        config = RAGConfig()

    yield

    # Cleanup
    if rag_cache:
        rag_cache.close()

    # Flush LangFuse traces
    langfuse_tracer.flush()


app = FastAPI(
    title="VinaSmol RAG API",
    description="Vietnamese RAG (Retrieval-Augmented Generation) Service",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    services = {
        "api": "running",
        "rag_pipeline": "initialized" if rag_pipeline else "not_initialized",
        "cache": "enabled" if rag_cache and rag_cache.enabled else "disabled",
    }

    # Check Qdrant connection
    if rag_pipeline and rag_pipeline.retriever:
        try:
            rag_pipeline.retriever.get_collection_info()
            services["qdrant"] = "connected"
        except Exception:
            services["qdrant"] = "disconnected"

    return HealthResponse(
        status="healthy",
        version="0.2.0",
        services=services,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a RAG query."""
    start_time = time.time()
    cached = False

    if rag_pipeline is None:
        REQUEST_COUNT.labels(endpoint="query", status="error").inc()
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Add documents first.",
        )

    # Check cache
    cache_key_params = {
        "top_k": request.top_k,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }

    if request.use_cache and rag_cache:
        cached_result = rag_cache.get(request.question, **cache_key_params)
        if cached_result:
            CACHE_HITS.inc()
            cached_result["cached"] = True
            cached_result["latency_ms"] = (time.time() - start_time) * 1000
            REQUEST_COUNT.labels(endpoint="query", status="success_cached").inc()
            return QueryResponse(**cached_result)

        CACHE_MISSES.inc()

    try:
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
            max_new_tokens=request.max_tokens,
        )

        latency_ms = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(endpoint="query").observe(latency_ms / 1000)
        REQUEST_COUNT.labels(endpoint="query", status="success").inc()

        # Record retrieval scores
        for doc in result.retrieved_docs:
            RETRIEVAL_SCORE.observe(doc.score)

        response_data = {
            "question": result.question,
            "answer": result.answer,
            "sources": [
                RetrievedDoc(
                    content=doc.content,
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for doc in result.retrieved_docs
            ],
            "latency_ms": latency_ms,
            "cached": cached,
            "metadata": result.metadata,
        }

        # Cache the result
        if request.use_cache and rag_cache:
            cache_data = response_data.copy()
            cache_data["sources"] = [s.dict() for s in cache_data["sources"]]
            rag_cache.set(request.question, cache_data, **cache_key_params)

        # Trace with LangFuse
        langfuse_tracer.trace_rag_query(
            question=result.question,
            answer=result.answer,
            sources=[s.dict() for s in response_data["sources"]],
            latency_ms=latency_ms,
            cached=cached,
            metadata={
                "top_k": request.top_k,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            },
        )

        return QueryResponse(**response_data)

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="query", status="error").inc()
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_query_generator(
    question: str,
    top_k: int,
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Generate streaming query response.

    Yields:
        Server-sent events with incremental response data.
    """
    try:
        # Retrieve documents
        retrieved = rag_pipeline.retriever.retrieve(
            query=question,
            top_k=top_k,
        )

        # Send retrieved documents first
        yield f"data: {json.dumps({'type': 'sources', 'data': [{'content': d.content, 'score': d.score} for d in retrieved]})}\n\n"

        # Generate answer (Note: streaming generation would require model.generate with callback)
        context_docs = [doc.content for doc in retrieved]
        generation_result = rag_pipeline.generator.generate(
            question=question,
            context_docs=context_docs,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )

        yield f"data: {json.dumps({'type': 'answer', 'data': generation_result.answer})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Process a RAG query with streaming response."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    return StreamingResponse(
        stream_query_generator(
            request.question,
            request.top_k,
            request.temperature,
            request.max_tokens,
        ),
        media_type="text/event-stream",
    )


@app.post("/documents")
async def add_documents(request: DocumentRequest):
    """Add documents to the knowledge base."""
    global rag_pipeline

    if rag_pipeline is None:
        # Initialize pipeline on first document addition
        logger.info("Initializing RAG pipeline...")
        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize(
            collection_name=config.retriever.collection_name if config else "vietnamese_docs",
            model_name=config.generator.model_name if config else "vinai/PhoGPT-4B-Chat",
            embedding_model=config.retriever.embedding_model if config else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        rag_pipeline.retriever.create_collection()
        logger.info("RAG pipeline initialized")

    try:
        rag_pipeline.add_documents(
            documents=request.documents,
            metadatas=request.metadatas,
        )
        REQUEST_COUNT.labels(endpoint="documents", status="success").inc()

        # Invalidate cache since knowledge base changed
        if rag_cache:
            invalidated = rag_cache.invalidate()
            logger.info(f"Invalidated {invalidated} cache entries")

        return {
            "status": "success",
            "documents_added": len(request.documents),
        }

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="documents", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/collection/info", response_model=CollectionInfo)
async def collection_info():
    """Get collection statistics."""
    if rag_pipeline is None or rag_pipeline.retriever is None:
        raise HTTPException(status_code=404, detail="No collection initialized")

    try:
        info = rag_pipeline.retriever.get_collection_info()
        return CollectionInfo(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.delete("/collection")
async def delete_collection():
    """Delete the entire collection."""
    if rag_pipeline is None or rag_pipeline.retriever is None:
        raise HTTPException(status_code=404, detail="No collection initialized")

    try:
        rag_pipeline.retriever.create_collection(recreate=True)

        # Invalidate all cache
        if rag_cache:
            rag_cache.invalidate()

        return {"status": "success", "message": "Collection deleted and recreated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics."""
    if not rag_cache:
        return {"enabled": False}

    return rag_cache.get_stats()


@app.delete("/cache")
async def clear_cache():
    """Clear all cached results."""
    if not rag_cache:
        return {"enabled": False, "cleared": 0}

    cleared = rag_cache.invalidate()
    return {"enabled": True, "cleared": cleared}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
