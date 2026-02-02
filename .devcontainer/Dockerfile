FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    vim \
    wget \
    unzip \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies globally for development
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    pip install -r /tmp/requirements-dev.txt

# Install pre-commit hooks
RUN pip install pre-commit

# Set working directory
WORKDIR /workspaces/vinasmol-rag-mlops

# Create directories for data and models (within storage limits)
RUN mkdir -p /workspaces/data /workspaces/models /workspaces/mlruns

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
