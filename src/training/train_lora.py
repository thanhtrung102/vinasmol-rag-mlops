"""LoRA fine-tuning for VinaSmol on Vietnamese data."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


@dataclass
class LoRATrainingConfig:
    """Configuration for LoRA training."""

    # Model
    model_name: str = "vinai/PhoGPT-4B-Chat"
    output_dir: str = "./outputs/vinasmol-lora"

    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    # Training parameters (optimized for 16GB RAM)
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 512

    # Quantization (required for Codespaces)
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # MLflow
    mlflow_experiment: str = "vinasmol-lora-training"


class VinaSmolLoRATrainer:
    """LoRA trainer for VinaSmol Vietnamese language model."""

    def __init__(self, config: LoRATrainingConfig):
        """Initialize the trainer.

        Args:
            config: Training configuration.
        """
        self.config = config
        self.model = None
        self.tokenizer = None

    def setup_mlflow(self) -> None:
        """Set up MLflow experiment tracking."""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:8080"))
        mlflow.set_experiment(self.config.mlflow_experiment)

    def load_model(self) -> None:
        """Load the base model with quantization."""
        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
            bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def prepare_dataset(
        self,
        dataset_path: str | None = None,
        dataset: Dataset | None = None,
    ) -> Dataset:
        """Prepare training dataset.

        Args:
            dataset_path: Path to dataset file or HuggingFace dataset name.
            dataset: Pre-loaded dataset.

        Returns:
            Tokenized dataset ready for training.
        """
        if dataset is None:
            if dataset_path:
                dataset = load_dataset("json", data_files=dataset_path, split="train")
            else:
                raise ValueError("Either dataset_path or dataset must be provided")

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            # Assuming dataset has 'text' field
            texts = examples.get("text", examples.get("content", []))

            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return tokenized

    def train(self, train_dataset: Dataset, eval_dataset: Dataset | None = None) -> None:
        """Run LoRA fine-tuning.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Training arguments optimized for Codespaces
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch" if eval_dataset else "no",
            fp16=True,
            optim="paged_adamw_8bit",
            report_to=["mlflow"],
            gradient_checkpointing=True,
            max_grad_norm=0.3,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train with MLflow tracking
        with mlflow.start_run():
            # Log config
            mlflow.log_params({
                "model_name": self.config.model_name,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "learning_rate": self.config.learning_rate,
                "epochs": self.config.num_train_epochs,
            })

            # Train
            trainer.train()

            # Save model
            trainer.save_model()

            # Log model artifact
            mlflow.log_artifacts(self.config.output_dir, "model")

    def save_merged_model(self, output_path: str) -> None:
        """Save the merged (base + LoRA) model.

        Args:
            output_path: Path to save the merged model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)


def main():
    """Main training entrypoint."""
    import argparse

    parser = argparse.ArgumentParser(description="LoRA fine-tuning for VinaSmol")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to training data")
    args = parser.parse_args()

    # Initialize with default config (can be loaded from YAML)
    config = LoRATrainingConfig()

    # Initialize trainer
    trainer = VinaSmolLoRATrainer(config)
    trainer.setup_mlflow()
    trainer.load_model()

    # Prepare and train
    train_dataset = trainer.prepare_dataset(dataset_path=args.dataset)
    trainer.train(train_dataset)

    print(f"Training complete. Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
