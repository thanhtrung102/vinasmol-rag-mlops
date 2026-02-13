"""LoRA fine-tuning for VinaSmol on Vietnamese data.

This module provides memory-optimized LoRA fine-tuning with:
- MLflow experiment tracking
- Weights & Biases integration
- Checkpoint resumption
- HuggingFace Hub publishing
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import mlflow
import torch
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.training.trainer_config import TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class WandbMetricsCallback(TrainerCallback):
    """Callback to log additional metrics to W&B."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B on each logging step."""
        if logs is None:
            return

        try:
            import wandb

            if wandb.run is not None:
                wandb.log(logs, step=state.global_step)
        except ImportError:
            pass


class VinaSmolLoRATrainer:
    """LoRA trainer for VinaSmol Vietnamese language model.

    Provides memory-optimized LoRA fine-tuning with full MLOps integration:
    - 4-bit quantization for 16GB RAM environments
    - MLflow experiment tracking
    - Optional Weights & Biases logging
    - Checkpoint resumption support
    - HuggingFace Hub publishing
    """

    def __init__(self, config: TrainerConfig):
        """Initialize the trainer.

        Args:
            config: Training configuration loaded from YAML.
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._wandb_initialized = False

    def setup_tracking(self) -> None:
        """Set up experiment tracking (MLflow and optionally W&B)."""
        # MLflow setup
        mlflow_uri = os.getenv(
            "MLFLOW_TRACKING_URI", self.config.mlflow.tracking_uri
        )
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)
        logger.info(f"MLflow tracking URI: {mlflow_uri}")
        logger.info(f"MLflow experiment: {self.config.mlflow.experiment_name}")

        # W&B setup (optional)
        if self.config.wandb.enabled:
            self._setup_wandb()

    def _setup_wandb(self) -> None:
        """Initialize Weights & Biases tracking."""
        try:
            import wandb

            wandb.init(
                project=self.config.wandb.project,
                entity=self.config.wandb.entity,
                name=self.config.wandb.run_name,
                config=self.config.to_dict(),
                tags=self.config.wandb.tags,
                resume="allow",
            )
            self._wandb_initialized = True
            logger.info(f"W&B initialized: {self.config.wandb.project}")
        except ImportError:
            logger.warning("wandb not installed. Skipping W&B integration.")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def load_model(self) -> None:
        """Load the base model with quantization and apply LoRA."""
        logger.info(f"Loading model: {self.config.model.name}")

        # Quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(
                torch, self.config.quantization.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.quantization.use_double_quant,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            revision=self.config.model.revision,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Prepare for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)

        # Apply LoRA
        lora_config = LoraConfig(
            r=self.config.lora.r,
            lora_alpha=self.config.lora.alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.dropout,
            bias=self.config.lora.bias,
            task_type=TaskType.CAUSAL_LM,
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("Model loaded and LoRA applied successfully")

    def load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from a training checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint directory.
        """
        logger.info(f"Loading from checkpoint: {checkpoint_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.quantization.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(
                torch, self.config.quantization.bnb_4bit_compute_dtype
            ),
            bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.quantization.use_double_quant,
        )

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=self.config.model.trust_remote_code,
        )

        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            checkpoint_path,
            is_trainable=True,
        )
        self.model = prepare_model_for_kbit_training(self.model)

        logger.info("Model loaded from checkpoint successfully")

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
                logger.info(f"Loading dataset from: {dataset_path}")
                path = Path(dataset_path)
                if path.suffix == ".jsonl":
                    dataset = load_dataset(
                        "json", data_files=str(path), split="train"
                    )
                elif path.suffix == ".json":
                    dataset = load_dataset(
                        "json", data_files=str(path), split="train"
                    )
                else:
                    # Assume HuggingFace dataset
                    dataset = load_dataset(dataset_path, split="train")
            else:
                raise ValueError("Either dataset_path or dataset must be provided")

        text_column = self.config.data.text_column

        def tokenize_function(examples: dict[str, Any]) -> dict[str, Any]:
            texts = examples.get(text_column, [])

            return self.tokenizer(
                texts,
                truncation=True,
                max_length=self.config.training.max_seq_length,
                padding="max_length",
            )

        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Dataset prepared: {len(tokenized)} samples")
        return tokenized

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
    ) -> None:
        """Run LoRA fine-tuning.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Determine report_to based on configuration
        report_to = ["mlflow"]
        if self._wandb_initialized:
            report_to.append("wandb")

        # Check for checkpoint resumption
        resume_checkpoint = self.config.output.resume_from_checkpoint
        if resume_checkpoint and not Path(resume_checkpoint).exists():
            logger.warning(
                f"Checkpoint path {resume_checkpoint} not found. Starting fresh."
            )
            resume_checkpoint = None

        # Training arguments optimized for Codespaces (16GB RAM)
        training_args = TrainingArguments(
            output_dir=self.config.output.dir,
            num_train_epochs=self.config.training.num_epochs,
            per_device_train_batch_size=self.config.training.batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            learning_rate=self.config.training.learning_rate,
            warmup_ratio=self.config.training.warmup_ratio,
            logging_steps=self.config.output.logging_steps,
            save_strategy=self.config.output.save_strategy,
            save_total_limit=self.config.output.save_total_limit,
            evaluation_strategy="epoch" if eval_dataset else "no",
            fp16=self.config.training.fp16,
            optim=self.config.training.optim,
            report_to=report_to,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            max_grad_norm=self.config.training.max_grad_norm,
            load_best_model_at_end=bool(eval_dataset),
            logging_first_step=True,
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Callbacks
        callbacks = []
        if self._wandb_initialized:
            callbacks.append(WandbMetricsCallback())

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks,
        )

        # Train with MLflow tracking
        with mlflow.start_run(run_name=self.config.mlflow.run_name):
            # Log configuration
            mlflow.log_params(self.config.to_dict())

            # Log additional info
            mlflow.set_tag("model_name", self.config.model.name)
            mlflow.set_tag("framework", "transformers+peft")

            # Train (with optional checkpoint resumption)
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

            # Log training metrics
            metrics = train_result.metrics
            mlflow.log_metrics(metrics)

            # Save final model
            trainer.save_model()
            logger.info(f"Model saved to {self.config.output.dir}")

            # Log model artifacts to MLflow
            if self.config.mlflow.log_model:
                mlflow.log_artifacts(self.config.output.dir, "model")

        logger.info("Training complete!")

    def push_to_hub(self, commit_message: str | None = None) -> str | None:
        """Push the trained adapter to HuggingFace Hub.

        Args:
            commit_message: Optional custom commit message.

        Returns:
            URL of the pushed model, or None if push failed.
        """
        if not self.config.hub.enabled:
            logger.info("HuggingFace Hub push disabled in config")
            return None

        if not self.config.hub.repo_id:
            logger.error("No repo_id specified for HuggingFace Hub push")
            return None

        try:
            from huggingface_hub import HfApi

            logger.info(f"Pushing to HuggingFace Hub: {self.config.hub.repo_id}")

            # Push adapter
            self.model.push_to_hub(
                self.config.hub.repo_id,
                private=self.config.hub.private,
                commit_message=commit_message or self.config.hub.commit_message,
            )

            # Push tokenizer
            self.tokenizer.push_to_hub(
                self.config.hub.repo_id,
                private=self.config.hub.private,
            )

            repo_url = f"https://huggingface.co/{self.config.hub.repo_id}"
            logger.info(f"Successfully pushed to: {repo_url}")

            # Log to MLflow
            mlflow.set_tag("hf_hub_repo", self.config.hub.repo_id)

            return repo_url

        except Exception as e:
            logger.error(f"Failed to push to HuggingFace Hub: {e}")
            return None

    def save_merged_model(self, output_path: str) -> None:
        """Save the merged (base + LoRA) model.

        Args:
            output_path: Path to save the merged model.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        logger.info(f"Merging and saving model to: {output_path}")
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info("Merged model saved successfully")

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._wandb_initialized:
            try:
                import wandb

                wandb.finish()
            except Exception:
                pass


def main():
    """Main training entrypoint."""
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for VinaSmol Vietnamese LLM"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to training data (overrides config)",
    )
    parser.add_argument(
        "--eval-dataset",
        type=str,
        help="Path to evaluation data (overrides config)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push trained adapter to HuggingFace Hub",
    )
    parser.add_argument(
        "--merge-and-save",
        type=str,
        help="Path to save merged model (base + LoRA)",
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading config from: {args.config}")
    config = TrainerConfig.from_yaml(args.config)

    # Override with CLI arguments
    if args.resume:
        config.output.resume_from_checkpoint = args.resume

    # Validate config
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.error(f"Config validation error: {issue}")
        raise ValueError("Configuration validation failed")

    # Initialize trainer
    trainer = VinaSmolLoRATrainer(config)
    trainer.setup_tracking()

    # Load model (or from checkpoint)
    if args.resume and Path(args.resume).exists():
        trainer.load_from_checkpoint(args.resume)
    else:
        trainer.load_model()

    # Prepare datasets
    train_path = args.dataset or config.data.train_file
    train_dataset = trainer.prepare_dataset(dataset_path=train_path)

    eval_dataset = None
    eval_path = args.eval_dataset or config.data.eval_file
    if eval_path and Path(eval_path).exists():
        eval_dataset = trainer.prepare_dataset(dataset_path=eval_path)

    # Train
    trainer.train(train_dataset, eval_dataset)

    # Push to Hub if requested
    if args.push_to_hub or config.hub.enabled:
        trainer.push_to_hub()

    # Merge and save if requested
    if args.merge_and_save:
        trainer.save_merged_model(args.merge_and_save)

    # Cleanup
    trainer.cleanup()

    logger.info(f"Training complete. Model saved to {config.output.dir}")


if __name__ == "__main__":
    main()
