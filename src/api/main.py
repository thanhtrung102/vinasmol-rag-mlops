"""FastAPI application for Vietnamese RAG service."""

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

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


# Request/Response models
class QueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=1024)


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
    metadata: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    services: dict[str, str]


# Global RAG pipeline (initialized on startup)
rag_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup: Initialize RAG pipeline
    global rag_pipeline
    # Pipeline initialization is deferred to avoid loading models on import
    # In production, initialize here with: rag_pipeline = RAGPipeline()
    yield
    # Shutdown: Cleanup


app = FastAPI(
    title="VinaSmol RAG API",
    description="Vietnamese RAG (Retrieval-Augmented Generation) Service",
    version="0.1.0",
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
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={
            "api": "running",
            "rag_pipeline": "initialized" if rag_pipeline else "not_initialized",
        },
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a RAG query."""
    start_time = time.time()

    if rag_pipeline is None:
        REQUEST_COUNT.labels(endpoint="query", status="error").inc()
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Add documents first.",
        )

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

        return QueryResponse(
            question=result.question,
            answer=result.answer,
            sources=[
                RetrievedDoc(
                    content=doc.content,
                    score=doc.score,
                    metadata=doc.metadata,
                )
                for doc in result.retrieved_docs
            ],
            latency_ms=latency_ms,
            metadata=result.metadata,
        )

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="query", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents")
async def add_documents(request: DocumentRequest):
    """Add documents to the knowledge base."""
    global rag_pipeline

    if rag_pipeline is None:
        # Initialize pipeline on first document addition
        from src.rag import RAGPipeline

        rag_pipeline = RAGPipeline()
        rag_pipeline.initialize()
        rag_pipeline.retriever.create_collection()

    try:
        rag_pipeline.add_documents(
            documents=request.documents,
            metadatas=request.metadatas,
        )
        REQUEST_COUNT.labels(endpoint="documents", status="success").inc()

        return {
            "status": "success",
            "documents_added": len(request.documents),
        }

    except Exception as e:
        REQUEST_COUNT.labels(endpoint="documents", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collection/info")
async def collection_info():
    """Get collection statistics."""
    if rag_pipeline is None or rag_pipeline.retriever is None:
        raise HTTPException(status_code=404, detail="No collection initialized")

    return rag_pipeline.retriever.get_collection_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
