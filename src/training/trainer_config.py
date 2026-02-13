"""Training configuration management for VinaSmol.

Provides YAML config loading, validation, and structured configuration
for LoRA fine-tuning with MLflow and W&B integration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str = "vinai/PhoGPT-4B-Chat"
    revision: str = "main"
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"


@dataclass
class QuantizationConfig:
    """Quantization settings for memory efficiency."""

    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_double_quant: bool = True


@dataclass
class TrainingParams:
    """Training hyperparameters."""

    num_epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    max_grad_norm: float = 0.3


@dataclass
class DataConfig:
    """Dataset configuration."""

    train_file: str = "data/train.jsonl"
    eval_file: str | None = "data/eval.jsonl"
    text_column: str = "text"


@dataclass
class OutputConfig:
    """Output and saving configuration."""

    dir: str = "./outputs/vinasmol-lora"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    logging_steps: int = 10
    resume_from_checkpoint: str | None = None


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""

    experiment_name: str = "vinasmol-lora-training"
    tracking_uri: str = "http://localhost:8080"
    log_model: bool = True
    run_name: str | None = None


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""

    enabled: bool = False
    project: str = "vinasmol-rag-mlops"
    entity: str | None = None
    run_name: str | None = None
    log_model: str = "checkpoint"
    tags: list[str] = field(default_factory=list)


@dataclass
class HuggingFaceHubConfig:
    """HuggingFace Hub push configuration."""

    enabled: bool = False
    repo_id: str | None = None
    private: bool = True
    commit_message: str = "Upload VinaSmol LoRA adapter"


@dataclass
class TrainerConfig:
    """Complete training configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    training: TrainingParams = field(default_factory=TrainingParams)
    data: DataConfig = field(default_factory=DataConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hub: HuggingFaceHubConfig = field(default_factory=HuggingFaceHubConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainerConfig":
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Populated TrainerConfig instance.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValueError: If the YAML is malformed.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            raw_config = {}

        return cls._from_dict(raw_config)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> "TrainerConfig":
        """Create config from dictionary."""
        return cls(
            model=cls._parse_section(data.get("model", {}), ModelConfig),
            lora=cls._parse_section(data.get("lora", {}), LoRAConfig),
            quantization=cls._parse_section(
                data.get("quantization", {}), QuantizationConfig
            ),
            training=cls._parse_section(data.get("training", {}), TrainingParams),
            data=cls._parse_section(data.get("data", {}), DataConfig),
            output=cls._parse_section(data.get("output", {}), OutputConfig),
            mlflow=cls._parse_section(data.get("mlflow", {}), MLflowConfig),
            wandb=cls._parse_section(data.get("wandb", {}), WandbConfig),
            hub=cls._parse_section(data.get("hub", {}), HuggingFaceHubConfig),
        )

    @staticmethod
    def _parse_section(data: dict[str, Any], dataclass_type: type) -> Any:
        """Parse a config section into a dataclass.

        Args:
            data: Dictionary of config values.
            dataclass_type: The dataclass type to instantiate.

        Returns:
            Instantiated dataclass with provided values.
        """
        if not data:
            return dataclass_type()

        # Filter to only valid fields for the dataclass
        valid_fields = {f.name for f in dataclass_type.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return dataclass_type(**filtered_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for logging."""
        return {
            "model": {
                "name": self.model.name,
                "revision": self.model.revision,
            },
            "lora": {
                "r": self.lora.r,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
            },
            "quantization": {
                "load_in_4bit": self.quantization.load_in_4bit,
                "compute_dtype": self.quantization.bnb_4bit_compute_dtype,
                "quant_type": self.quantization.bnb_4bit_quant_type,
            },
            "training": {
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "gradient_accumulation_steps": self.training.gradient_accumulation_steps,
                "max_seq_length": self.training.max_seq_length,
            },
        }

    def validate(self) -> list[str]:
        """Validate configuration and return list of issues.

        Returns:
            List of validation error messages (empty if valid).
        """
        issues = []

        if self.training.batch_size < 1:
            issues.append("Batch size must be at least 1")

        if self.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")

        if self.lora.r < 1:
            issues.append("LoRA rank (r) must be at least 1")

        if not self.lora.target_modules:
            issues.append("At least one target module must be specified for LoRA")

        if self.hub.enabled and not self.hub.repo_id:
            issues.append("HuggingFace Hub repo_id is required when hub.enabled is True")

        return issues
