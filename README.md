# VinaSmol RAG MLOps

Vietnamese LLM (VinaSmol) and RAG System with Production-Grade MLOps Pipeline.

[![CI Pipeline](https://github.com/thanhtrung102/vinasmol-rag-mlops/actions/workflows/ci.yaml/badge.svg)](https://github.com/thanhtrung102/vinasmol-rag-mlops/actions)
![Project Status](https://img.shields.io/badge/Status-86%25_Complete-blue)
![Phase](https://img.shields.io/badge/Phase-6_of_7-green)

## Overview

This project implements a **production-ready MLOps pipeline** for Vietnamese language models and RAG systems:

- **VinaSmol**: Fine-tuning Vietnamese language models using LoRA (PhoGPT-4B-Chat)
- **OpenRAG**: Retrieval-Augmented Generation with hallucination detection and evaluation
- **MLOps**: Comprehensive experiment tracking, monitoring, CI/CD, and observability

### 🎯 Project Status: **86% Complete** (6 of 7 Phases)

| Phase | Status | Description |
|-------|--------|-------------|
| 1. Data Pipeline | ✅ Complete | Vietnamese text processing from Common Crawl |
| 2. Training Infrastructure | ✅ Complete | MLflow + LoRA fine-tuning |
| 3. RAG System | ✅ Complete | Qdrant + FastAPI + Reranking |
| 4. Evaluation Framework | ✅ Complete | Ragas + Hallucination detection |
| 5. Monitoring Stack | ✅ Complete | Prometheus + Grafana + LangFuse |
| 6. Infrastructure as Code | 🔲 Pending | Terraform modules (GCP/AWS) |
| 7. CI/CD Pipeline | ✅ Complete | GitHub Actions with quality gates |

📊 [**View Detailed Status**](PROJECT_STATUS_UPDATE.md) | 📋 [**Implementation Plan**](IMPLEMENTATION_PLAN.md)

## Quick Start (GitHub Codespaces)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/YOUR_USERNAME/vinasmol-rag-mlops)

**Requirements**:
- 4-core, 16GB RAM, 32GB storage Codespace
- **GPU required for training** (use GPU-enabled Codespace or cloud GPU instance)
- Standard Codespaces work for API/RAG features only

```bash
# Setup is automatic via postCreateCommand
# Or manually run:
make setup

# Start all services
make services-up

# Run tests (doesn't require GPU)
make test
```

### For Training (Requires GPU)

Use one of these GPU-enabled environments:
- **GitHub Codespaces**: GPU-enabled instance (select GPU machine type)
- **Google Colab**: Free T4 GPU available
- **Kaggle Notebooks**: Free GPU available
- **Local**: NVIDIA GPU with CUDA 11.8+

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                  │
│  Common Crawl → Vietnamese Filter → Text Processing → Embeddings    │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                                 │
│  MLflow Tracking │ W&B Prompts │ HuggingFace Hub │ LoRA Fine-tuning │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                  │
│  FastAPI Gateway │ Qdrant Vectors │ vLLM/Transformers │ Reranking   │
└─────────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                                 │
│  Prometheus │ Grafana │ LangFuse │ Ragas Evaluation │ Evidently     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
vinasmol-rag-mlops/
├── .devcontainer/          # Codespaces configuration
├── .github/workflows/      # CI/CD pipelines
├── configs/                # Configuration files
├── infrastructure/         # Terraform IaC
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── src/
│   ├── api/               # FastAPI application
│   ├── data_pipeline/     # Data processing
│   ├── evaluation/        # RAG & LLM evaluation
│   ├── monitoring/        # Observability
│   ├── rag/               # RAG components
│   └── training/          # Model training
├── tests/
│   ├── unit/              # Unit tests
│   └── integration/       # Integration tests
├── docker-compose.yaml     # Local services
├── Makefile               # Development commands
└── requirements.txt       # Python dependencies
```

## Services

All services run via Docker Compose with persistent volumes and health checks.

| Service | Port | Status | Description |
|---------|------|--------|-------------|
| **FastAPI** | 8000 | ✅ Ready | RAG API with streaming support |
| **Qdrant** | 6333 | ✅ Ready | Vector database (v1.11.3) |
| **MLflow** | 8080 | ✅ Ready | Experiment tracking UI |
| **Redis** | 6379 | ✅ Ready | Query result caching |
| **Prometheus** | 9090 | ✅ Ready | Metrics collection with alerts |
| **Grafana** | 3000 | ✅ Ready | 8-panel RAG dashboard (admin/admin) |
| **Postgres** | 5432 | ✅ Ready | MLflow backend store |
| **LangFuse** | Cloud | 🔧 Optional | LLM request tracing |

**Quick Access**:
```bash
# Health checks
curl http://localhost:8000/health
curl http://localhost:6333/health
curl http://localhost:9090/-/healthy

# Metrics
curl http://localhost:8000/metrics
```

## Usage

### 1. Data Pipeline

```bash
# Process Vietnamese text from Common Crawl
make data-process

# Generate embeddings
make data-embed
```

### 2. Training

```bash
# LoRA fine-tuning (optimized for 16GB RAM)
make train-lora
```

### 3. RAG API

```bash
# Start the API server
make api

# Query the RAG system
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Việt Nam nằm ở đâu?"}'
```

### 4. Evaluation

```bash
# Evaluate RAG system
make eval-rag

# Run hallucination detection
make eval-hallucination
```

### 5. Monitoring & Observability

**Full monitoring stack with Prometheus, Grafana, and LangFuse integration.**

```bash
# Start all services (includes monitoring)
make services-up

# Access monitoring dashboards
# Grafana:    http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# MLflow:     http://localhost:8080
# Metrics:    http://localhost:8000/metrics
```

**Features**:
- ✅ **8-panel Grafana dashboard** - Request rate, error rate, latency (P50/P95/P99), cache hits, retrieval scores
- ✅ **5 Prometheus alert rules** - Error rate, latency, retrieval quality, cache performance, uptime
- ✅ **LangFuse LLM tracing** - Request/response traces with latency and quality metrics
- ✅ **Real-time metrics** - Auto-refresh every 10 seconds

**Optional: Enable LangFuse tracing**:
```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-your-key
export LANGFUSE_SECRET_KEY=sk-lf-your-secret
# Restart API to enable tracing
```

**Generate test metrics**:
```bash
# Send test queries to populate dashboards
for i in {1..10}; do
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"Test query $i\", \"top_k\": 3}"
  sleep 1
done
```

📊 **Documentation**:
- [MONITORING_SETUP.md](MONITORING_SETUP.md) - Quick start guide
- [Grafana Dashboard Guide](configs/grafana/dashboards/) - Panel descriptions
- [Alert Rules](configs/alert.rules.yml) - Prometheus alerting configuration

## Development

```bash
# Install dev dependencies
make install-dev

# Run linting
make lint

# Run formatter
make format

# Run all tests
make test-all

# Pre-commit hooks
make pre-commit
```

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
- `MLFLOW_TRACKING_URI`: MLflow server URL
- `QDRANT_HOST`: Vector database host
- `OPENAI_API_KEY`: Required for Ragas evaluation

## Tech Stack

### Core ML/NLP
| Component | Technology | Version |
|-----------|-----------|---------|
| **Base Model** | PhoGPT-4B-Chat (VinAI) | - |
| **Framework** | PyTorch, Transformers | 2.1.0, 4.35.2 |
| **Fine-tuning** | PEFT (LoRA), bitsandbytes | 0.6.2 |
| **Embeddings** | Sentence-Transformers | paraphrase-multilingual-MiniLM-L12-v2 |
| **Text Processing** | FastText, underthesea | Vietnamese language detection |

### RAG System
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Qdrant | v1.11.3 - Semantic search |
| **Caching** | Redis | v7-alpine - Query result caching |
| **Reranking** | Cross-encoder | BGE reranker - Result reordering |
| **API** | FastAPI, Uvicorn | Async REST API with streaming |

### MLOps & Monitoring
| Category | Technologies |
|----------|-------------|
| **Experiment Tracking** | MLflow, Weights & Biases |
| **Orchestration** | Prefect |
| **Versioning** | DVC, HuggingFace Hub |
| **Evaluation** | Ragas, DeepEval |
| **Metrics** | Prometheus (5 alert rules) |
| **Dashboards** | Grafana (8-panel dashboard) |
| **Tracing** | LangFuse (LLM observability) |
| **Drift Detection** | Evidently |

### DevOps & Infrastructure
| Category | Technologies |
|----------|-------------|
| **CI/CD** | GitHub Actions (5-job pipeline) |
| **Containers** | Docker, Docker Compose |
| **IaC** | Terraform (pending) |
| **Testing** | pytest, ruff, mypy |
| **Package Build** | Python build, setuptools |

## Internship Alignment

This project directly addresses LINAGORA internship requirements:

| Requirement | Implementation |
|-------------|----------------|
| VinaSmol evaluation | Ragas metrics + custom Vietnamese benchmarks |
| Vietnamese dataset creation | Common Crawl pipeline with language detection |
| RAG hallucination detection | Faithfulness scoring + factual consistency |
| GraphRAG exploration | Knowledge graph extraction (roadmap) |
| Agentic systems | LangGraph workflows (roadmap) |

## License

MIT License - See [LICENSE](LICENSE) for details.
