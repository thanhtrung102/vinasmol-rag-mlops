.PHONY: setup install install-dev test lint format pre-commit clean services-up services-down mlflow prefect api

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

setup: install-dev pre-commit-install
	@echo "✅ Development environment setup complete"

install:
	pip3 install --user --upgrade pip
	pip3 install --user -r requirements.txt

install-dev: install
	pip3 install --user -r requirements-dev.txt

pre-commit-install:
	pre-commit install || true

# =============================================================================
# CODE QUALITY
# =============================================================================

lint:
	ruff check src tests
	mypy src --ignore-missing-imports

format:
	ruff format src tests
	ruff check --fix src tests

pre-commit:
	pre-commit run --all-files

# =============================================================================
# TESTING
# =============================================================================

test:
	pytest tests/unit -v --cov=src --cov-report=term-missing

test-integration:
	pytest tests/integration -v

test-all:
	pytest tests -v --cov=src --cov-report=html

test-eval:
	python -m pytest tests/evaluation -v --tb=short

# =============================================================================
# DOCKER SERVICES
# =============================================================================

services-up:
	docker compose up -d

services-down:
	docker compose down

services-logs:
	docker compose logs -f

services-clean:
	docker compose down -v --remove-orphans

# =============================================================================
# INDIVIDUAL SERVICES
# =============================================================================

mlflow:
	mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns \
		--host 0.0.0.0 \
		--port 8080

prefect:
	prefect server start --host 0.0.0.0

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# =============================================================================
# DATA PIPELINE
# =============================================================================

data-download:
	python scripts/download_sample_data.py

data-process:
	python -m src.data_pipeline.process_vietnamese

data-embed:
	python -m src.data_pipeline.generate_embeddings

# =============================================================================
# TRAINING
# =============================================================================

train:
	python -m src.training.train --config configs/training_config.yaml

train-lora:
	python -m src.training.train_lora --config configs/training_config.yaml

train-lora-resume:
	python -m src.training.train_lora --config configs/training_config.yaml --resume $(CHECKPOINT)

train-lora-push:
	python -m src.training.train_lora --config configs/training_config.yaml --push-to-hub

# =============================================================================
# EVALUATION
# =============================================================================

eval-rag:
	python -m src.evaluation.evaluate_rag --config configs/eval_config.yaml

eval-hallucination:
	python -m src.evaluation.hallucination_detection

benchmark:
	python -m src.evaluation.benchmark_vietnamese

# =============================================================================
# INFRASTRUCTURE
# =============================================================================

infra-init:
	cd infrastructure/terraform && terraform init

infra-plan:
	cd infrastructure/terraform && terraform plan

infra-apply:
	cd infrastructure/terraform && terraform apply

infra-destroy:
	cd infrastructure/terraform && terraform destroy

# =============================================================================
# UTILITIES
# =============================================================================

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".ruff_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf htmlcov .coverage

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

docs:
	mkdocs serve -a 0.0.0.0:8001

help:
	@echo "Available commands:"
	@echo "  setup          - Complete development environment setup"
	@echo "  install        - Install production dependencies"
	@echo "  install-dev    - Install development dependencies"
	@echo "  lint           - Run linting checks"
	@echo "  format         - Format code"
	@echo "  test           - Run unit tests"
	@echo "  test-all       - Run all tests with coverage"
	@echo "  services-up    - Start Docker services"
	@echo "  services-down  - Stop Docker services"
	@echo "  mlflow         - Start MLflow server"
	@echo "  api            - Start FastAPI server"
	@echo "  train          - Run training pipeline"
	@echo "  eval-rag       - Evaluate RAG system"
	@echo "  clean          - Clean temporary files"
