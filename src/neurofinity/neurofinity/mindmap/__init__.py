"""
Neurofinity Mind Map Generation Module.

This module provides the core mind map generation pipeline that converts
meeting transcripts into structured, hierarchical mind maps optimized
for neurodivergent accessibility.

Supports three backends:
  - Claude API (Anthropic) — highest quality, recommended for production
  - OpenAI API — good alternative with GPT-4o
  - Local Flan-T5 + LoRA — cost-free inference after fine-tuning
"""

from neurofinity.mindmap.generator import MindMapGenerator
from neurofinity.mindmap.schemas import MindMapNode, MindMapOutput
from neurofinity.mindmap.converter import (
    json_to_mermaid,
    json_to_mindbench_tokens,
    mindbench_tokens_to_json,
)

__all__ = [
    "MindMapGenerator",
    "MindMapNode",
    "MindMapOutput",
    "json_to_mermaid",
    "json_to_mindbench_tokens",
    "mindbench_tokens_to_json",
]
