# Neurofinity - Mind Map Integration Guide

## What's New

This build adds the complete AI-powered mind map generation pipeline to your Neurofinity project. Here's what was added:

**New Files:**
- `neurofinity/mindmap/__init__.py` — Package exports
- `neurofinity/mindmap/generator.py` — Multi-backend MindMapGenerator (Claude, OpenAI, local Flan-T5)
- `neurofinity/mindmap/schemas.py` — Pydantic data models (MindMapNode, MindMapOutput, AccessibilityConfig)
- `neurofinity/mindmap/converter.py` — Format converters (JSON ↔ Mermaid ↔ MindBench tokens ↔ Graphviz)
- `neurofinity/mindmap/trainer.py` — LoRA fine-tuning pipeline for Flan-T5
- `neurofinity/mindmap/data_prep.py` — MindBench dataset preparation + synthetic data generation
- `neurofinity/api.py` — FastAPI REST endpoints for mind map generation
- `scripts/train_mindmap_model.py` — CLI training script
- `CURSOR_GUIDE.md` — Cursor Pro IDE configuration

**Modified Files:**
- `requirements.txt` — Added anthropic, openai, peft, accelerate, datasets, etc.

## Quick Start (5 Minutes)

### Step 1: Install dependencies

```bash
cd src/neurofinity
pip install -r requirements.txt
```

### Step 2: Set your API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
# OR
export OPENAI_API_KEY="your-key-here"
```

### Step 3: Start the API server

```bash
uvicorn neurofinity.api:app --reload --port 8000
```

### Step 4: Test it

```bash
curl -X POST http://localhost:8000/mindmap/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "John: Lets discuss the Q3 roadmap. Sarah: I think we should prioritize mobile. Mike: The API needs optimization too.",
    "backend": "claude",
    "theme": "high_contrast"
  }'
```

## Training Your Own Model

### Step 1: Download MindBench data

```bash
# From HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('MiaSanLei/MindBench', local_dir='data/raw/mindbench', allow_patterns='*.jsonl')
"
```

### Step 2: Generate training data (reverse engineering from MindBench)

```bash
python scripts/train_mindmap_model.py prepare-data \
  --mindbench-path data/raw/mindbench/crawled_data_train.jsonl \
  --output-path data/processed/mindmap_training.jsonl \
  --max-samples 200 \
  --backend claude
```

### Step 3: Train with LoRA on H100

```bash
python scripts/train_mindmap_model.py train \
  --data-path data/processed/mindmap_training.jsonl \
  --output-dir models/mindmap-lora \
  --epochs 5 \
  --batch-size 4 \
  --lr 5e-4
```

### Step 4: Test

```bash
python scripts/train_mindmap_model.py test \
  --model-path models/mindmap-lora/final
```

## Architecture

```
neurofinity/
├── mindmap/
│   ├── generator.py     # MindMapGenerator (3 backends)
│   ├── schemas.py       # Pydantic models
│   ├── converter.py     # JSON ↔ Mermaid ↔ MindBench ↔ Graphviz
│   ├── trainer.py       # LoRA fine-tuning pipeline
│   └── data_prep.py     # MindBench data preparation
├── api.py               # FastAPI endpoints
├── nlp.py               # Existing NLP (summarization, topics)
├── audio_to_text.py     # Existing audio transcription
├── neuroui.py           # Existing Streamlit frontend
└── config.py            # Configuration
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/mindmap/generate` | Generate mind map from transcript text |
| POST | `/mindmap/from-audio` | Upload audio → transcribe → mind map |
| GET | `/mindmap/backends` | List available backends and their status |
| GET | `/health` | Health check |
