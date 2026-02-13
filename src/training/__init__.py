"""Training module for VinaSmol fine-tuning.

This module provides LoRA fine-tuning capabilities with MLOps integration:
- Memory-optimized training for 16GB RAM environments
- MLflow experiment tracking
- Weights & Biases integration
- HuggingFace Hub publishing
"""

from src.training.train_lora import VinaSmolLoRATrainer
from src.training.trainer_config import (
    DataConfig,
    HuggingFaceHubConfig,
    LoRAConfig,
    MLflowConfig,
    ModelConfig,
    OutputConfig,
    QuantizationConfig,
    TrainerConfig,
    TrainingParams,
    WandbConfig,
)

__all__ = [
    "VinaSmolLoRATrainer",
    "TrainerConfig",
    "ModelConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "TrainingParams",
    "DataConfig",
    "OutputConfig",
    "MLflowConfig",
    "WandbConfig",
    "HuggingFaceHubConfig",
]
