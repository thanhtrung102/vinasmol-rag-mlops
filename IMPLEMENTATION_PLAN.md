# VinaSmol RAG MLOps - Implementation Plan

> **Project Goal**: Build a production-grade Vietnamese LLM (VinaSmol) and RAG system with MLOps pipeline for LINAGORA AI Internship portfolio.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Internship Alignment](#internship-alignment)
3. [Architecture Overview](#architecture-overview)
4. [Technology Decisions](#technology-decisions)
5. [Implementation Phases](#implementation-phases)
6. [Phase Details](#phase-details)
7. [Success Metrics](#success-metrics)
8. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### Project Scope

| Component | Description |
|-----------|-------------|
| **VinaSmol** | Fine-tune Vietnamese language model using LoRA on curated Vietnamese corpus |
| **OpenRAG** | RAG system with hallucination detection and automatic evaluation |
| **MLOps** | End-to-end pipeline with experiment tracking, CI/CD, and monitoring |

### Dataset Choice: Common Crawl

After analyzing MLOps Zoomcamp datasets, **Common Crawl** was selected because:

- ✅ Contains Vietnamese web content (filter by `.vn` domains)
- ✅ Mirrors real LLM training data sourcing
- ✅ Demonstrates large-scale data engineering
- ✅ Can create RAG evaluation datasets from web Q&A
- ✅ Aligns with internship task: "Create dataset from Vietnamese media sources"

---

## Internship Alignment

| LINAGORA Requirement | Project Implementation |
|----------------------|------------------------|
| VinaSmol evaluation & benchmarking | Ragas metrics + custom Vietnamese benchmarks |
| Vietnamese dataset creation | Common Crawl pipeline with language detection + N8N-style automation |
| RAG hallucination detection | Faithfulness scoring via Ragas + factual consistency checks |
| Automatic RAG evaluation | CI/CD integrated evaluation with quality gates |
| GraphRAG exploration | Neo4j knowledge graph extraction (Phase 5) |
| Agentic systems | LangGraph multi-step workflows (Phase 6) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Common Crawl WARC    ──►    Vietnamese Filter    ──►    Text Processor    │
│         │                           │                           │           │
│         └───────────── Prefect Orchestration ───────────────────┘           │
│                                     │                                       │
│                    ┌────────────────┴────────────────┐                      │
│                    ▼                                 ▼                      │
│            ┌──────────────┐                 ┌──────────────┐                │
│            │ Training Set │                 │  RAG Corpus  │                │
│            │   (JSONL)    │                 │  (Chunked)   │                │
│            └──────────────┘                 └──────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            TRAINING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │   MLflow    │    │    W&B      │    │  HF Hub     │                    │
│   │  Tracking   │    │   Prompts   │    │  Registry   │                    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                    │
│          │                  │                  │                            │
│          └──────────────────┴──────────────────┘                            │
│                             │                                               │
│              VinaSmol Fine-tuning (LoRA/QLoRA)                              │
│              • 4-bit quantization for 16GB RAM                              │
│              • Gradient checkpointing                                       │
│              • paged_adamw_8bit optimizer                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SERVING LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │                     FastAPI Gateway                          │          │
│   │    /query  /documents  /health  /metrics                     │          │
│   └────────────────────────┬────────────────────────────────────┘          │
│                            │                                                │
│          ┌─────────────────┼─────────────────┐                              │
│          ▼                 ▼                 ▼                              │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐                       │
│   │    vLLM    │    │   Qdrant   │    │  Reranker  │                       │
│   │  VinaSmol  │    │  Vectors   │    │   (BGE)    │                       │
│   └────────────┘    └────────────┘    └────────────┘                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                          MONITORING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐              │
│   │Prometheus │  │  Grafana  │  │ LangFuse  │  │   Ragas   │              │
│   │  Metrics  │  │ Dashboards│  │  Traces   │  │ Eval Suite│              │
│   └───────────┘  └───────────┘  └───────────┘  └───────────┘              │
│                                                                             │
│   Alerts:                                                                   │
│   • Latency > 2s                                                            │
│   • Hallucination Rate > 10%                                                │
│   • Retrieval Score < 0.7                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        INFRASTRUCTURE LAYER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Terraform (GCP/AWS)  │  Docker Compose  │  GitHub Actions CI/CD          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────┐          │
│   │  Pre-commit: ruff, mypy, pytest, model-eval-check           │          │
│   └─────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technology Decisions

### Adopted from MLOps Zoomcamp (✅)

| Technology | Purpose | Rationale |
|------------|---------|-----------|
| **MLflow** | Experiment tracking | Industry standard, tracks params/metrics/artifacts |
| **Prefect** | Workflow orchestration | Python-native, simpler than Airflow |
| **Terraform** | Infrastructure as Code | Multi-cloud, reproducible |
| **GitHub Actions** | CI/CD | Free, integrates with everything |
| **pytest + ruff** | Testing & linting | Modern Python best practices |

### Adapted for LLM/RAG (⚠️)

| Zoomcamp Approach | Adaptation | Reason |
|-------------------|------------|--------|
| Flask deployment | **FastAPI + vLLM** | Async support, GPU-optimized LLM serving |
| Evidently monitoring | **Evidently + Ragas + LangFuse** | RAG-specific metrics |
| Batch processing | Embedding generation + evaluation | Not just inference |

### Replaced (❌)

| Zoomcamp Approach | Replacement | Reason |
|-------------------|-------------|--------|
| CSV data loading | **datatrove / Spark** | Common Crawl scale |
| RMSE/accuracy | **Ragas metrics** | Faithfulness, relevance, hallucination |
| Pickle model | **HF Hub + DVC** | LLM weights are GB-scale |

---

## Implementation Phases

```
Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5 ──► Phase 6 ──► Phase 7
  │           │           │           │           │           │           │
  ▼           ▼           ▼           ▼           ▼           ▼           ▼
 Data      Training      RAG       Evaluation  Monitoring    IaC        CI/CD
Pipeline   Infra       System     Framework    Stack      Terraform   Pipeline
```

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase 1** | Data Pipeline | ✅ Implemented |
| **Phase 2** | Training Infrastructure | ✅ Implemented |
| **Phase 3** | RAG System | ✅ Implemented |
| **Phase 4** | Evaluation Framework | 🔲 Pending |
| **Phase 5** | Monitoring Stack | 🔲 Pending |
| **Phase 6** | Infrastructure as Code | 🔲 Pending |
| **Phase 7** | CI/CD & Best Practices | 🔲 Pending |

---

## Phase Details

### Phase 1: Data Pipeline ✅ COMPLETE

**Goal**: Vietnamese text extraction and processing from Common Crawl

**Components Implemented**:

```
src/data_pipeline/
├── __init__.py
├── vietnamese_detector.py    # FastText language detection
└── text_processor.py         # Cleaning, chunking, quality scoring
```

**Features**:

| Feature | Implementation | File |
|---------|---------------|------|
| Vietnamese detection | FastText + regex diacritics check | `vietnamese_detector.py` |
| Text cleaning | URL/email removal, whitespace normalization | `text_processor.py` |
| Quality scoring | Repetition penalty, length checks | `text_processor.py` |
| RAG chunking | Sentence-boundary aware splitting | `text_processor.py` |

**Demo Command**:
```bash
python scripts/demo_phase1.py
```

**Tests**:
```bash
make test  # 17 passed, 2 fixed
```

---

### Phase 2: Training Infrastructure ✅ COMPLETE

**Goal**: Configure MLflow + W&B tracking, implement LoRA fine-tuning

**Tasks**:

- [x] Set up MLflow server with artifact storage
- [x] Integrate Weights & Biases for prompt tracking
- [x] Implement LoRA training script (memory-optimized for 16GB)
- [x] Create training config YAML schema
- [x] Add model checkpointing and resumption
- [x] Push trained adapters to HuggingFace Hub

**Key Files**:
```
src/training/
├── __init__.py          # Module exports
├── train_lora.py        # LoRA fine-tuning with full MLOps integration
└── trainer_config.py    # Configuration dataclass with YAML loader
```

**Features Implemented**:

| Feature | Implementation | File |
|---------|---------------|------|
| YAML Config Loading | TrainerConfig.from_yaml() with validation | `trainer_config.py` |
| MLflow Tracking | Experiment, params, metrics, artifacts | `train_lora.py` |
| W&B Integration | Optional logging with WandbMetricsCallback | `train_lora.py` |
| Checkpoint Resumption | --resume CLI flag + config option | `train_lora.py` |
| HuggingFace Hub Push | push_to_hub() with --push-to-hub flag | `train_lora.py` |
| Memory Optimization | 4-bit quantization, gradient checkpointing | `training_config.yaml` |

**Memory Optimization** (for Codespaces 16GB):
```python
# configs/training_config.yaml
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_quant_type: "nf4"

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  gradient_checkpointing: true
  optim: "paged_adamw_8bit"
```

**Demo Commands**:
```bash
# Basic training
make train-lora

# Resume from checkpoint
make train-lora-resume CHECKPOINT=./outputs/vinasmol-lora/checkpoint-100

# Train and push to HuggingFace Hub
make train-lora-push
```

**Acceptance Criteria**:
- [x] MLflow UI accessible at `localhost:8080`
- [x] Training run logged with params/metrics
- [x] LoRA adapter saved and loadable
- [x] Training completes without OOM on 16GB RAM

---

### Phase 3: RAG System ✅ COMPLETE

**Goal**: Build Qdrant vector store, FastAPI gateway, integrate LLM serving

**Tasks**:

- [x] Initialize Qdrant collection with Vietnamese embeddings
- [x] Implement document ingestion endpoint
- [x] Create RAG query pipeline (retrieve → rerank → generate)
- [x] Add streaming response support
- [x] Implement caching layer (Redis)

**Key Files**:
```
src/rag/
├── __init__.py       # Module exports
├── retriever.py      # Qdrant integration
├── generator.py      # LLM generation
├── pipeline.py       # End-to-end RAG
├── reranker.py       # Document reranking (cross-encoder + hybrid)
├── cache.py          # Redis caching layer
└── config.py         # Configuration management

src/api/
└── main.py           # FastAPI endpoints with streaming

configs/
└── rag_config.yaml   # RAG system configuration
```

**Features Implemented**:

| Feature | Implementation | File |
|---------|---------------|------|
| Vector Retrieval | Qdrant semantic search | `retriever.py` |
| Document Reranking | Cross-encoder + hybrid scoring | `reranker.py` |
| LLM Generation | Vietnamese prompt templates | `generator.py` |
| Redis Caching | Query result caching with TTL | `cache.py` |
| Configuration | YAML-based config management | `config.py` |
| Streaming | Server-sent events | `main.py` |

**API Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/query` | POST | RAG query with caching |
| `/query/stream` | POST | Streaming RAG query |
| `/documents` | POST | Document ingestion |
| `/collection/info` | GET | Collection statistics |
| `/collection` | DELETE | Delete collection |
| `/cache/stats` | GET | Cache statistics |
| `/cache` | DELETE | Clear cache |
| `/metrics` | GET | Prometheus metrics |

**Demo Commands**:
```bash
# Start services
make services-up

# Start API server
make api

# Run Phase 3 demo
python scripts/demo_phase3.py

# Test API
curl http://localhost:8000/health
```

**Acceptance Criteria**:
- [x] Qdrant running via Docker Compose
- [x] Documents indexed with Vietnamese embeddings
- [x] Query returns answer + source citations
- [x] Latency < 2s for typical queries (with caching)

---

### Phase 4: Evaluation Framework 🔲

**Goal**: Implement Ragas metrics, hallucination detection, Vietnamese benchmarks

**Tasks**:

- [ ] Integrate Ragas evaluation suite
- [ ] Implement hallucination detection module
- [ ] Create Vietnamese Q&A benchmark dataset
- [ ] Add evaluation to CI pipeline
- [ ] Generate evaluation reports

**Key Files**:
```
src/evaluation/
├── evaluate_rag.py           # Ragas integration (implemented)
├── hallucination_detector.py # Factual consistency
└── vietnamese_benchmark.py   # Custom Vietnamese eval
```

**Metrics**:
| Metric | Source | Threshold |
|--------|--------|-----------|
| Faithfulness | Ragas | > 0.7 |
| Answer Relevance | Ragas | > 0.7 |
| Context Precision | Ragas | > 0.6 |
| Hallucination Rate | Custom | < 10% |

**Acceptance Criteria**:
- [ ] Ragas evaluation runs on test dataset
- [ ] Hallucination detection flags unfaithful responses
- [ ] Metrics logged to MLflow
- [ ] CI fails if metrics below threshold

---

### Phase 5: Monitoring Stack 🔲

**Goal**: Deploy Prometheus + Grafana + LangFuse with alerting

**Tasks**:

- [ ] Configure Prometheus scraping for FastAPI
- [ ] Create Grafana dashboards for RAG metrics
- [ ] Integrate LangFuse for LLM tracing
- [ ] Set up alerting rules
- [ ] Implement log aggregation

**Docker Services**:
```yaml
# docker-compose.yaml (implemented)
services:
  prometheus:    # Port 9090
  grafana:       # Port 3000
  qdrant:        # Port 6333
  mlflow:        # Port 8080
  redis:         # Port 6379
```

**Dashboards**:
- Request latency histogram
- Retrieval score distribution
- Hallucination rate over time
- Token usage and costs

**Acceptance Criteria**:
- [ ] Grafana dashboard shows live metrics
- [ ] Alerts fire when thresholds exceeded
- [ ] LangFuse traces individual requests

---

### Phase 6: Infrastructure as Code 🔲

**Goal**: Terraform modules for full stack deployment

**Tasks**:

- [ ] Create GCP/AWS Terraform modules
- [ ] Define compute resources (GPU instances)
- [ ] Set up networking and security
- [ ] Configure managed services (Cloud Storage, etc.)
- [ ] Add remote state management

**Terraform Structure**:
```
infrastructure/terraform/
├── main.tf
├── variables.tf
├── outputs.tf
├── modules/
│   ├── compute/
│   ├── storage/
│   └── networking/
└── environments/
    ├── dev.tfvars
    └── prod.tfvars
```

**Acceptance Criteria**:
- [ ] `terraform plan` succeeds
- [ ] Infrastructure deployable with single command
- [ ] State stored remotely

---

### Phase 7: CI/CD & Best Practices 🔲

**Goal**: GitHub Actions pipeline with testing, linting, model evaluation gates

**Tasks**:

- [ ] Lint and format checks
- [ ] Unit test execution
- [ ] Integration tests with services
- [ ] Model evaluation gate
- [ ] Docker image build and push
- [ ] Deployment automation

**Pipeline Stages**:
```yaml
# .github/workflows/ci.yaml (implemented)
jobs:
  lint:        # ruff check
  test:        # pytest unit tests
  build:       # Python package build
  docker:      # Docker image build
  integration: # Tests with Qdrant service
```

**Quality Gates**:
| Gate | Threshold | Action |
|------|-----------|--------|
| Test coverage | > 30% | Block merge |
| Lint errors | 0 | Block merge |
| Hallucination rate | < 10% | Block deploy |
| Faithfulness | > 0.7 | Block deploy |

**Acceptance Criteria**:
- [ ] PR blocked if tests fail
- [ ] Automatic deployment on main merge
- [ ] Model evaluation runs before deploy

---

## Success Metrics

### Technical Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| RAG Faithfulness | > 0.7 | Ragas evaluation |
| Query Latency (P95) | < 2s | Prometheus |
| Hallucination Rate | < 10% | Custom detector |
| Test Coverage | > 50% | pytest-cov |
| CI Pipeline Time | < 10min | GitHub Actions |

### Portfolio Demonstration

| Skill | Evidence |
|-------|----------|
| MLOps | MLflow tracking, Prefect pipelines, monitoring |
| NLP/LLM | Vietnamese text processing, LoRA fine-tuning |
| RAG | Retrieval, generation, evaluation |
| Data Engineering | Common Crawl processing at scale |
| DevOps | Docker, Terraform, CI/CD |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| OOM during training | 4-bit quantization, gradient checkpointing |
| Common Crawl too large | Start with sample, scale incrementally |
| OpenAI API costs (Ragas) | Use local models for evaluation where possible |
| Codespace limitations | Optimize for 16GB RAM, 4 cores |

---

## Quick Reference Commands

```bash
# Development
make setup          # Install dependencies
make test           # Run unit tests
make lint           # Check code quality
make format         # Auto-format code

# Services
make services-up    # Start Docker services
make services-down  # Stop services
make mlflow         # Start MLflow UI
make api            # Start FastAPI server

# Demo
python scripts/demo_phase1.py  # Phase 1 demo

# Training
make train-lora     # LoRA fine-tuning

# Evaluation
make eval-rag       # RAG evaluation
```

---

## Repository Structure

```
vinasmol-rag-mlops/
├── .devcontainer/           # Codespaces config (4 cores, 16GB RAM)
├── .github/workflows/       # CI/CD pipelines
├── configs/                 # YAML configurations
├── data/                    # Training/eval data
├── infrastructure/          # Terraform modules
├── notebooks/               # Jupyter experiments
├── scripts/                 # Utility scripts
├── src/
│   ├── api/                # FastAPI application
│   ├── data_pipeline/      # Vietnamese text processing ✅
│   ├── evaluation/         # RAG evaluation
│   ├── monitoring/         # Observability
│   ├── rag/                # RAG components
│   └── training/           # Model fine-tuning ✅
├── tests/                   # Unit & integration tests
├── docker-compose.yaml      # Local services
├── Makefile                 # Development commands
├── requirements.txt         # Production deps
└── requirements-dev.txt     # Development deps
```

---

## Next Steps

1. ~~**Immediate**: Complete Phase 2 (Training Infrastructure)~~ ✅ Done
2. ~~**Immediate**: Complete Phase 3 (RAG System)~~ ✅ Done
3. **Immediate**: Complete Phase 4 (Evaluation Framework)
4. **Next**: Phase 5-7 (Monitoring, IaC, CI/CD)
5. **Portfolio Ready**: All phases complete with documentation

---

*Generated from project planning conversation - Feb 2026*
