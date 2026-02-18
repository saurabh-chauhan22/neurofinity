"""
LoRA Fine-Tuning Pipeline for Flan-T5 Mind Map Generation.

This module implements the three-stage curriculum training strategy:
  Stage 1: Meeting summarization foundation (learn dialogue understanding)
  Stage 2: Structured extraction (learn JSON/MindBench token output)
  Stage 3: End-to-end transcript → mind map generation

Optimized for H100 GPU with LoRA to stay within ~$20 budget.

Usage:
    from neurofinity.mindmap.trainer import MindMapTrainer

    trainer = MindMapTrainer(
        base_model="google/flan-t5-large",
        output_dir="./models/mindmap-lora",
    )

    # Load and prepare your dataset
    trainer.prepare_dataset("path/to/training_data.jsonl")

    # Run training (all 3 stages)
    trainer.train()

    # Or run individual stages
    trainer.train_stage1()  # Summarization foundation
    trainer.train_stage2()  # Structured output
    trainer.train_stage3()  # End-to-end mind map
"""

import json
import os
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

import torch
from torch import device

from neurofinity.modeling.mindmap_evaluation import MindMapEvaluator

app = typer.Typer()


class MindMapTrainer:
    """LoRA fine-tuning pipeline for Flan-T5 mind map generation.

    This trainer uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA
    adapters to fine-tune Flan-T5-large for structured mind map output.

    The key configuration choices (from our research):
      - LoRA r=16, alpha=32 (2:1 ratio, sweet spot for T5)
      - Target ALL linear layers: q, k, v, o, wi, wo
      - bf16 precision (T5 has known fp16 overflow issues)
      - AdamW optimizer with linear warmup schedule
    """

    def __init__(
        self,
        base_model: str = "google/flan-t5-large",
        output_dir: str = "./models/mindmap-lora",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        learning_rate: float = 5e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 5,
        max_source_length: int = 1024,
        max_target_length: int = 512,
        warmup_ratio: float = 0.1,
    ):
        self.base_model = base_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LoRA hyperparameters (researched optimal values)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Training hyperparameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_epochs = num_epochs
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.warmup_ratio = warmup_ratio

        # Will be loaded lazily
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._mindmap_evaluator = None

        logger.info(
            f"MindMapTrainer initialized: base={base_model}, "
            f"LoRA r={lora_r}, alpha={lora_alpha}, lr={learning_rate}"
        )

    def setup_model(self):
        """Load the base model and apply LoRA adapter configuration."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from peft import LoraConfig, get_peft_model, TaskType

        logger.info(f"Loading base model: {self.base_model}")

        # Load tokenizer and add MindBench special tokens
        self._tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        special_tokens = self._get_special_tokens()
        num_added = self._tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        logger.info(f"Added {num_added} special tokens to tokenizer")

        # Load model and resize embeddings for new tokens
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.base_model)
        self._model.resize_token_embeddings(len(self._tokenizer))

        # Configure LoRA — targeting ALL linear layers in T5 is critical
        # for structured output quality (per QLoRA paper findings)
        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        
        
        
        self._model = get_peft_model(self._model, lora_config)
        trainable, total = self._model.get_nb_trainable_parameters()
        logger.info(
            f"LoRA applied: {trainable:,} trainable params "
            f"/ {total:,} total ({100 * trainable / total:.2f}%)"
        )

    def prepare_dataset(self, data_path: str, validation_split: float = 0.1):
        """Load and tokenize training data from a JSONL file.

        Expected JSONL format (one per line):
        {"input": "transcript text...", "output": "<s_map>...</s_map>"}

        Args:
            data_path: Path to the JSONL training data file.
            validation_split: Fraction of data to use for validation.
        """
        from datasets import Dataset

        logger.info(f"Loading dataset from: {data_path}")

        records = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        logger.info(f"Loaded {len(records)} training examples")

        dataset = Dataset.from_list(records)
        split = dataset.train_test_split(test_size=validation_split, seed=42)

        self._train_dataset = split["train"]
        self._eval_dataset = split["test"]

        logger.info(
            f"Dataset split: {len(self._train_dataset)} train, "
            f"{len(self._eval_dataset)} eval"
        )

    def _tokenize_function(self, examples):
        """Tokenize input-output pairs for Seq2Seq training."""
        # The input is the meeting transcript (with instruction prefix)
        inputs = [
            f"Generate a mind map from this meeting transcript: {text}"
            for text in examples["input"]
        ]
        targets = examples["output"]

        model_inputs = self._tokenizer(
            inputs,
            max_length=self.max_source_length,
            truncation=True,
            padding="max_length",
        )

        labels = self._tokenizer(
            targets,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
        )

        # Replace padding token ids with -100 so they're ignored in loss
        labels["input_ids"] = [
            [(l if l != self._tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def train(self):
        """Run the full training pipeline."""
        if self._model is None:
            self.setup_model()

        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

        # Tokenize datasets
        logger.info("Tokenizing datasets...")
        tokenized_train = self._train_dataset.map(
            self._tokenize_function, batched=True,
            remove_columns=self._train_dataset.column_names,
        )
        tokenized_eval = self._eval_dataset.map(
            self._tokenize_function, batched=True,
            remove_columns=self._eval_dataset.column_names,
        )
        
        use_bf16 = False
        use_fp16 = False
        if device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            compute_capability = torch.cuda.get_device_capability(0)
            # bf16 requires compute capability >= 8.0 (Ampere+)
            if compute_capability[0] >= 8:
                use_bf16 = True
                logger.info(f"GPU {gpu_name} supports bf16 - using bf16 precision")
            else:
                use_fp16 = True
                logger.info(f"GPU {gpu_name} doesn't support bf16 - using fp16 precision")
                logger.warning("Note: T5 has known fp16 overflow issues. Monitor for NaN losses.")
        else:
            logger.info("Using fp32 precision (CPU training)")

        # Training arguments optimized for H100 + LoRA
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_epochs,
            warmup_ratio=self.warmup_ratio,
            weight_decay=0.01,
            # CRITICAL: Use bf16, NOT fp16 — T5 has known fp16 overflow bugs
            bf16=use_bf16,
            fp16=use_fp16,
            #CUDA specific device settings
            dataloader_pin_memory=True,
            dataloader_num_workers=2,
            gradient_checkpointing=True,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=400,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            predict_with_generate=True,
            generation_max_length=self.max_target_length,
            report_to="wandb",  # Set to "wandb" if you want W&B logging
            lr_scheduler_type="linear",
            optim="adamw_torch",
        )

        self._mindmap_evaluator = MindMapEvaluator(self._tokenizer)
        # Initialize trainer
        self._trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=self._mindmap_evaluator.compute_metrics
        )
        self._trainer.tokenizer = self._tokenizer

        logger.info("Starting training...")
        self._trainer.train()
        # Save the LoRA adapter

        self.save_model()

    def save_model(self, path: Optional[str] = None):
        """Save the LoRA adapter weights and tokenizer."""
        save_path = Path(path) if path else self.output_dir / "final"
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))
        logger.success(f"Model saved to: {save_path}")

    @staticmethod
    def _get_special_tokens() -> list[str]:
        """Return the MindBench special tokens to add to the tokenizer.

        These tokens encode mind map structure explicitly, making it far
        easier for T5 to learn the output format than using whitespace.
        """
        tokens = ["<s_map>", "</s_map>", "<sep/>"]

        # Node depth tokens (support up to 10 levels deep)
        for i in range(10):
            tokens.extend([f"<s_node-{i}>", f"</s_node-{i}>"])

        # Content and relation tokens
        tokens.extend([
            "<s_text>", "</s_text>",
            "<s_relation>", "</s_relation>",
            # Custom Neurofinity tokens for node types
            "<s_decision>", "</s_decision>",
            "<s_action>", "</s_action>",
            "<s_question>", "</s_question>",
            "<s_insight>", "</s_insight>",
            "<s_speaker>", "</s_speaker>",
        ])

        return tokens


# ---------------------------------------------------------------------------
# CLI interface for running training from command line
# ---------------------------------------------------------------------------

@app.command()
def train_cli(
    data_path: str = typer.Argument(..., help="Path to training JSONL file"),
    output_dir: str = typer.Option("./models/mindmap-lora", help="Output directory"),
    base_model: str = typer.Option("google/flan-t5-large", help="Base model name"),
    epochs: int = typer.Option(5, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Per-device batch size"),
    lr: float = typer.Option(5e-4, help="Learning rate"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
):
    """Train the mind map generation model from the command line."""
    trainer = MindMapTrainer(
        base_model=base_model,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        lora_r=lora_r,
    )
    trainer.prepare_dataset(data_path)
    trainer.train()


if __name__ == "__main__":
    app()
