# VinaSmol RAG MLOps

Vietnamese LLM (VinaSmol) and RAG System with Production-Grade MLOps Pipeline.

## Overview

This project implements an end-to-end MLOps pipeline for:
- **VinaSmol**: Fine-tuning Vietnamese language models using LoRA
- **OpenRAG**: Retrieval-Augmented Generation system with hallucination detection
- **MLOps**: Experiment tracking, model registry, monitoring, and CI/CD

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

| Service | Port | Description |
|---------|------|-------------|
| FastAPI | 8000 | RAG API endpoints |
| MLflow | 8080 | Experiment tracking |
| Qdrant | 6333 | Vector database |
| Grafana | 3000 | Dashboards |
| Prometheus | 9090 | Metrics |
| Prefect | 5000 | Orchestration |

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

| Category | Technologies |
|----------|-------------|
| **LLM** | Transformers, PEFT, bitsandbytes |
| **RAG** | LangChain, Qdrant, Sentence-Transformers |
| **MLOps** | MLflow, Prefect, DVC |
| **Evaluation** | Ragas, DeepEval |
| **API** | FastAPI, Uvicorn |
| **Monitoring** | Prometheus, Grafana, Evidently |
| **IaC** | Terraform, Docker |
| **CI/CD** | GitHub Actions |

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
