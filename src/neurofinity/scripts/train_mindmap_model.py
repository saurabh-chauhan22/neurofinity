#!/usr/bin/env python3
"""
Train the Neurofinity Mind Map Generation Model.

This script provides a complete workflow for fine-tuning Flan-T5 with LoRA
to generate mind maps from meeting transcripts.

Usage Examples:
    # Step 1: Prepare training data from MindBench
    python scripts/train_mindmap_model.py prepare-data \
        --mindbench-path data/raw/crawled_data_train.jsonl \
        --output-path data/processed/mindmap_training.jsonl \
        --max-samples 200 \
        --backend claude

    # Step 2: Prepare data from QMSum (Apache 2.0 licensed)
    python scripts/train_mindmap_model.py prepare-qmsum \
        --qmsum-path data/raw/qmsum_train.json \
        --output-path data/processed/qmsum_training.jsonl \
        --max-samples 100

    # Step 3: Train the model
    python scripts/train_mindmap_model.py train \
        --data-path data/processed/mindmap_training.jsonl \
        --output-dir models/mindmap-lora \
        --epochs 5 \
        --batch-size 4

    # Step 4: Test the trained model
    python scripts/train_mindmap_model.py test \
        --model-path models/mindmap-lora/final \
        --transcript "John: Let's discuss the Q3 roadmap..."
"""

import sys
from pathlib import Path

# Add project root to path so imports work
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "neurofinity"))

import typer
from loguru import logger

app = typer.Typer(
    name="neurofinity-train",
    help="Train and evaluate Neurofinity mind map generation models.",
)


@app.command()
def prepare_data(
    mindbench_path: str = typer.Option(..., help="Path to MindBench JSONL"),
    output_path: str = typer.Option(
        "data/processed/mindmap_training.jsonl", help="Output JSONL path"
    ),
    backend: str = typer.Option("claude", help="LLM backend: claude or openai"),
    max_samples: int = typer.Option(200, help="Max training pairs to generate"),
    min_nodes: int = typer.Option(3, help="Min mind map nodes"),
    max_nodes: int = typer.Option(40, help="Max mind map nodes"),
):
    """Prepare training data from MindBench dataset (reverse engineering approach)."""
    from neurofinity.mindmap.data_prep import build_training_dataset

    pairs = build_training_dataset(
        mindbench_path=mindbench_path,
        output_path=output_path,
        backend=backend,
        max_samples=max_samples,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
    )
    logger.success(f"Generated {len(pairs)} training pairs")


@app.command()
def prepare_qmsum(
    qmsum_path: str = typer.Option(..., help="Path to QMSum JSON file"),
    output_path: str = typer.Option(
        "data/processed/qmsum_training.jsonl", help="Output JSONL path"
    ),
    backend: str = typer.Option("claude", help="LLM backend: claude or openai"),
    max_samples: int = typer.Option(100, help="Max training pairs"),
):
    """Prepare training data from QMSum dataset (Apache 2.0 licensed)."""
    from neurofinity.mindmap.data_prep import build_from_qmsum

    pairs = build_from_qmsum(
        qmsum_path=qmsum_path,
        output_path=output_path,
        backend=backend,
        max_samples=max_samples,
    )
    logger.success(f"Generated {len(pairs)} training pairs from QMSum")


@app.command()
def train(
    data_path: str = typer.Option(..., help="Path to training JSONL"),
    output_dir: str = typer.Option("models/mindmap-lora", help="Output directory"),
    base_model: str = typer.Option("google/flan-t5-large", help="Base model"),
    epochs: int = typer.Option(5, help="Training epochs"),
    batch_size: int = typer.Option(4, help="Per-device batch size"),
    lr: float = typer.Option(5e-4, help="Learning rate"),
    lora_r: int = typer.Option(16, help="LoRA rank"),
    lora_alpha: int = typer.Option(32, help="LoRA alpha"),
    max_source_len: int = typer.Option(1024, help="Max input token length"),
    max_target_len: int = typer.Option(512, help="Max output token length"),
):
    """Train the Flan-T5 + LoRA model for mind map generation."""
    from neurofinity.mindmap.trainer import MindMapTrainer

    trainer = MindMapTrainer(
        base_model=base_model,
        output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        learning_rate=lr,
        batch_size=batch_size,
        num_epochs=epochs,
        max_source_length=max_source_len,
        max_target_length=max_target_len,
    )
    trainer.prepare_dataset(data_path)
    trainer.train()


@app.command()
def test(
    model_path: str = typer.Option(
        "models/mindmap-lora/final", help="Path to trained model"
    ),
    transcript: str = typer.Option(
        None, help="Transcript text to test with"
    ),
    transcript_file: str = typer.Option(
        None, help="Path to a file containing the transcript"
    ),
):
    """Test the trained model on a transcript."""
    from neurofinity.mindmap.generator import MindMapGenerator

    if transcript_file:
        with open(transcript_file, "r") as f:
            transcript = f.read()
    elif transcript is None:
        # Use a default test transcript
        transcript = (
            "John: Good morning everyone. Let's start with the Q3 roadmap. "
            "Sarah: I think we should prioritize the mobile app redesign. "
            "Our user research shows 60% of users prefer mobile. "
            "John: Good point. What about the API performance issues? "
            "Mike: I've been looking into that. We need to optimize the "
            "database queries. I estimate it'll take two sprints. "
            "Sarah: Can we run both in parallel? "
            "John: Yes, let's do that. Mike, you own the API optimization. "
            "Sarah, you lead the mobile redesign. "
            "Mike: Sounds good. I'll have a plan by Friday. "
            "John: Great. Let's reconvene next week."
        )

    gen = MindMapGenerator(backend="local", model_path=model_path)
    result = gen.generate(transcript)

    logger.info(f"Generated mind map with {len(result.to_flat_topics())} nodes")
    logger.info(f"Mermaid code:\n{result.mermaid_code}")
    print("\n--- Generated Mind Map (JSON) ---")
    print(result.root.model_dump_json(indent=2))
    print("\n--- Mermaid Code ---")
    print(result.mermaid_code)


if __name__ == "__main__":
    app()
    
       