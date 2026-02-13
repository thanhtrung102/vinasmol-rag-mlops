#!/bin/bash
set -e

# Install CUDA drivers (already in deep learning image)
# Install additional dependencies
pip install --upgrade pip
pip install transformers peft bitsandbytes mlflow wandb

# Clone repository
cd /home/jupyter
git clone https://github.com/thanhtrung102/vinasmol-rag-mlops.git

echo "Training server setup complete"
