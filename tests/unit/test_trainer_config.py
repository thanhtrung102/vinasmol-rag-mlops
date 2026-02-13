"""Unit tests for trainer configuration."""

import tempfile
from pathlib import Path

import pytest

from src.training.trainer_config import (
    HuggingFaceHubConfig,
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    TrainerConfig,
    TrainingParams,
    WandbConfig,
)


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_default_values(self):
        """Test default model configuration."""
        config = ModelConfig()
        assert config.name == "vinai/PhoGPT-4B-Chat"
        assert config.revision == "main"
        assert config.trust_remote_code is True

    def test_custom_values(self):
        """Test custom model configuration."""
        config = ModelConfig(
            name="custom/model",
            revision="v1.0",
            trust_remote_code=False,
        )
        assert config.name == "custom/model"
        assert config.revision == "v1.0"
        assert config.trust_remote_code is False


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()
        assert config.r == 16
        assert config.alpha == 32
        assert config.dropout == 0.05
        assert "q_proj" in config.target_modules
        assert config.bias == "none"

    def test_custom_values(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            r=8,
            alpha=16,
            dropout=0.1,
            target_modules=["q_proj", "v_proj"],
        )
        assert config.r == 8
        assert config.alpha == 16
        assert config.dropout == 0.1
        assert len(config.target_modules) == 2


class TestQuantizationConfig:
    """Tests for QuantizationConfig dataclass."""

    def test_default_values(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.use_double_quant is True


class TestTrainingParams:
    """Tests for TrainingParams dataclass."""

    def test_default_values(self):
        """Test default training parameters."""
        config = TrainingParams()
        assert config.num_epochs == 3
        assert config.batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.learning_rate == 2e-4
        assert config.gradient_checkpointing is True
        assert config.optim == "paged_adamw_8bit"

    def test_memory_optimized_defaults(self):
        """Test that defaults are optimized for 16GB RAM."""
        config = TrainingParams()
        # Small batch size for memory efficiency
        assert config.batch_size <= 2
        # Gradient checkpointing enabled
        assert config.gradient_checkpointing is True
        # 8-bit optimizer
        assert "8bit" in config.optim


class TestWandbConfig:
    """Tests for WandbConfig dataclass."""

    def test_default_disabled(self):
        """Test W&B is disabled by default."""
        config = WandbConfig()
        assert config.enabled is False
        assert config.project == "vinasmol-rag-mlops"

    def test_enabled_with_project(self):
        """Test W&B enabled configuration."""
        config = WandbConfig(
            enabled=True,
            project="my-project",
            entity="my-team",
        )
        assert config.enabled is True
        assert config.project == "my-project"
        assert config.entity == "my-team"


class TestHuggingFaceHubConfig:
    """Tests for HuggingFaceHubConfig dataclass."""

    def test_default_disabled(self):
        """Test HF Hub push is disabled by default."""
        config = HuggingFaceHubConfig()
        assert config.enabled is False
        assert config.repo_id is None
        assert config.private is True

    def test_enabled_configuration(self):
        """Test HF Hub enabled configuration."""
        config = HuggingFaceHubConfig(
            enabled=True,
            repo_id="username/vinasmol-lora",
            private=False,
        )
        assert config.enabled is True
        assert config.repo_id == "username/vinasmol-lora"
        assert config.private is False


class TestTrainerConfig:
    """Tests for TrainerConfig main class."""

    def test_default_config(self):
        """Test default trainer configuration."""
        config = TrainerConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.training, TrainingParams)
        assert isinstance(config.wandb, WandbConfig)
        assert isinstance(config.hub, HuggingFaceHubConfig)

    def test_from_yaml_minimal(self):
        """Test loading minimal YAML config."""
        yaml_content = """
model:
  name: "test/model"
training:
  num_epochs: 5
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = TrainerConfig.from_yaml(f.name)

        assert config.model.name == "test/model"
        assert config.training.num_epochs == 5
        # Defaults should be preserved
        assert config.lora.r == 16
        assert config.quantization.load_in_4bit is True

        Path(f.name).unlink()

    def test_from_yaml_full(self):
        """Test loading full YAML config."""
        yaml_content = """
model:
  name: "vinai/PhoGPT-4B-Chat"
  revision: "v2.0"
  trust_remote_code: true

lora:
  r: 32
  alpha: 64
  dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_quant_type: "nf4"

training:
  num_epochs: 5
  batch_size: 2
  learning_rate: 1.0e-4

wandb:
  enabled: true
  project: "test-project"

hub:
  enabled: true
  repo_id: "user/model"
  private: false
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = TrainerConfig.from_yaml(f.name)

        assert config.model.revision == "v2.0"
        assert config.lora.r == 32
        assert config.lora.alpha == 64
        assert config.quantization.bnb_4bit_compute_dtype == "bfloat16"
        assert config.training.num_epochs == 5
        assert config.wandb.enabled is True
        assert config.hub.enabled is True
        assert config.hub.repo_id == "user/model"

        Path(f.name).unlink()

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            TrainerConfig.from_yaml("nonexistent.yaml")

    def test_to_dict(self):
        """Test config serialization to dictionary."""
        config = TrainerConfig()
        config_dict = config.to_dict()

        assert "model" in config_dict
        assert "lora" in config_dict
        assert "training" in config_dict
        assert config_dict["model"]["name"] == "vinai/PhoGPT-4B-Chat"
        assert config_dict["lora"]["r"] == 16

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = TrainerConfig()
        issues = config.validate()
        assert len(issues) == 0

    def test_validate_invalid_batch_size(self):
        """Test validation catches invalid batch size."""
        config = TrainerConfig()
        config.training.batch_size = 0
        issues = config.validate()
        assert any("batch size" in issue.lower() for issue in issues)

    def test_validate_invalid_learning_rate(self):
        """Test validation catches invalid learning rate."""
        config = TrainerConfig()
        config.training.learning_rate = -0.001
        issues = config.validate()
        assert any("learning rate" in issue.lower() for issue in issues)

    def test_validate_invalid_lora_rank(self):
        """Test validation catches invalid LoRA rank."""
        config = TrainerConfig()
        config.lora.r = 0
        issues = config.validate()
        assert any("lora" in issue.lower() for issue in issues)

    def test_validate_empty_target_modules(self):
        """Test validation catches empty target modules."""
        config = TrainerConfig()
        config.lora.target_modules = []
        issues = config.validate()
        assert any("target module" in issue.lower() for issue in issues)

    def test_validate_hub_enabled_without_repo(self):
        """Test validation catches hub enabled without repo_id."""
        config = TrainerConfig()
        config.hub.enabled = True
        config.hub.repo_id = None
        issues = config.validate()
        assert any("repo_id" in issue.lower() for issue in issues)


class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_load_project_training_config(self):
        """Test loading the project's training config file."""
        config_path = Path("configs/training_config.yaml")
        if config_path.exists():
            config = TrainerConfig.from_yaml(config_path)
            assert config.model.name is not None
            assert config.training.num_epochs > 0
            # Validate the loaded config
            issues = config.validate()
            assert len(issues) == 0
