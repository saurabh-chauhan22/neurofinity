"""
MindBench Dataset Preparation Utilities.

This module handles downloading, parsing, and converting the MindBench
dataset into training pairs for the Flan-T5 LoRA fine-tuning pipeline.

The key workflow:
  1. Download MindBench crawled dataset labels from HuggingFace
  2. Parse the MindBench token format into our MindMapNode schema
  3. Use an LLM (Claude/OpenAI) to generate synthetic meeting transcripts
     that would produce each mind map as output (reverse engineering)
  4. Save as JSONL training pairs: {"input": transcript, "output": mindmap_tokens}

This implements the "reverse engineering" approach — using real human-created
mind maps from MindBench as targets, then generating transcripts to match.
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

from loguru import logger


def load_mindbench_jsonl(path: str) -> list[dict]:
    """Load a MindBench JSONL file and extract the mind map structures.

    MindBench stores data in a conversation format. We extract the
    assistant's response which contains the <s_map>...</s_map> tokens.

    Args:
        path: Path to a MindBench JSONL file (e.g., crawled_data_train.jsonl)

    Returns:
        List of dicts with 'mindmap_tokens' and 'image_path' fields.
    """
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed JSON at line {line_num}")
                continue

            # Extract the assistant's mind map output from conversations
            conversations = data.get("conversations", [])
            mindmap_tokens = None

            for turn in conversations:
                if turn.get("from") == "assistant":
                    value = turn.get("value", "")
                    if "<s_map>" in value:
                        mindmap_tokens = value
                        break

            if mindmap_tokens:
                records.append({
                    "mindmap_tokens": mindmap_tokens,
                    "image_path": data.get("image", [""])[0] if data.get("image") else "",
                })

    logger.info(f"Loaded {len(records)} mind map structures from {path}")
    return records


def extract_topic_from_tokens(tokens: str) -> str:
    """Extract the root topic text from MindBench token sequence.

    Useful for understanding what each mind map is about before
    generating synthetic transcripts.
    """
    match = re.search(r"<s_node-0><s_text>(.*?)</s_text>", tokens)
    return match.group(1) if match else "Unknown Topic"


def count_nodes(tokens: str) -> int:
    """Count the total number of nodes in a MindBench token sequence."""
    return len(re.findall(r"<s_text>", tokens))


def filter_by_complexity(
    records: list[dict],
    min_nodes: int = 3,
    max_nodes: int = 50,
) -> list[dict]:
    """Filter mind maps by node count to match training difficulty.

    For curriculum learning, we train on simple maps first (3-10 nodes)
    then progress to complex ones (10-50 nodes).

    Args:
        records: List of MindBench records.
        min_nodes: Minimum number of nodes to include.
        max_nodes: Maximum number of nodes to include.

    Returns:
        Filtered list of records.
    """
    filtered = []
    for r in records:
        n = count_nodes(r["mindmap_tokens"])
        if min_nodes <= n <= max_nodes:
            filtered.append({**r, "node_count": n})

    logger.info(
        f"Filtered to {len(filtered)} records "
        f"({min_nodes}-{max_nodes} nodes, from {len(records)} total)"
    )
    return filtered


def generate_synthetic_transcript(
    mindmap_tokens: str,
    topic: str,
    backend: str = "claude",
    api_key: Optional[str] = None,
) -> str:
    """Generate a synthetic meeting transcript for a given mind map.

    This is the "reverse engineering" step: given a mind map structure,
    generate a realistic meeting transcript that would produce it.

    Args:
        mindmap_tokens: The MindBench token representation of the mind map.
        topic: The root topic of the mind map.
        backend: "claude" or "openai" for transcript generation.
        api_key: API key (falls back to environment variables).

    Returns:
        A synthetic meeting transcript string.
    """
    prompt = f"""You are generating a realistic meeting transcript that would result in this mind map structure.

MIND MAP TOPIC: {topic}

MIND MAP STRUCTURE (in MindBench token format):
{mindmap_tokens[:3000]}

Generate a realistic 3-5 minute meeting transcript with 2-4 speakers discussing this topic.
The transcript should naturally cover all the points in the mind map.

Rules:
1. Use speaker labels like "Speaker 1:", "Sarah:", "John:", etc.
2. Include natural conversation flow (questions, responses, decisions)
3. Cover all major branches of the mind map through discussion
4. Include filler words occasionally for realism (um, yeah, so)
5. Keep it under 800 words

Output ONLY the transcript, nothing else."""

    if backend == "claude":
        return _generate_with_claude(prompt, api_key)
    elif backend == "openai":
        return _generate_with_openai(prompt, api_key)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _generate_with_claude(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate text using Claude API."""
    import anthropic

    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _generate_with_openai(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate text using OpenAI API."""
    import openai

    key = api_key or os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


def build_training_dataset(
    mindbench_path: str,
    output_path: str,
    backend: str = "claude",
    api_key: Optional[str] = None,
    max_samples: int = 100,
    min_nodes: int = 3,
    max_nodes: int = 40,
):
    """Build the full training dataset from MindBench data.

    This is the main entry point for data preparation. It:
    1. Loads MindBench mind map structures
    2. Filters by complexity
    3. Generates synthetic transcripts for each
    4. Saves as JSONL training pairs

    Args:
        mindbench_path: Path to MindBench JSONL file.
        output_path: Where to save the training JSONL.
        backend: "claude" or "openai" for transcript generation.
        api_key: API key for the LLM backend.
        max_samples: Maximum number of training pairs to generate.
        min_nodes: Minimum mind map complexity.
        max_nodes: Maximum mind map complexity.
    """
    logger.info(f"Building training dataset from: {mindbench_path}")

    # Load and filter MindBench data
    records = load_mindbench_jsonl(mindbench_path)
    filtered = filter_by_complexity(records, min_nodes, max_nodes)

    # Limit to max_samples
    if len(filtered) > max_samples:
        import random
        random.seed(42)
        filtered = random.sample(filtered, max_samples)
        logger.info(f"Sampled {max_samples} records for generation")

    # Generate synthetic transcripts
    training_pairs = []
    for i, record in enumerate(filtered):
        topic = extract_topic_from_tokens(record["mindmap_tokens"])
        logger.info(f"[{i + 1}/{len(filtered)}] Generating transcript for: {topic}")

        try:
            transcript = generate_synthetic_transcript(
                record["mindmap_tokens"],
                topic,
                backend=backend,
                api_key=api_key,
            )

            training_pairs.append({
                "input": transcript,
                "output": record["mindmap_tokens"],
                "topic": topic,
                "node_count": record.get("node_count", count_nodes(record["mindmap_tokens"])),
            })
        except Exception as e:
            logger.error(f"Failed to generate transcript for '{topic}': {e}")
            continue

    # Save the training data
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.success(
        f"Training dataset saved: {len(training_pairs)} pairs → {output_path}"
    )
    return training_pairs


def build_from_qmsum(
    qmsum_path: str,
    output_path: str,
    backend: str = "claude",
    api_key: Optional[str] = None,
    max_samples: int = 50,
):
    """Build training pairs from QMSum dataset (Apache 2.0 licensed).

    QMSum provides meeting transcripts with topic segmentations — we
    use the transcripts as inputs and generate mind map targets.

    Args:
        qmsum_path: Path to QMSum JSON file (train/val/test).
        output_path: Where to save the training JSONL.
        backend: "claude" or "openai" for mind map generation.
        api_key: API key for the LLM backend.
        max_samples: Maximum samples to generate.
    """
    logger.info(f"Building training data from QMSum: {qmsum_path}")

    with open(qmsum_path, "r", encoding="utf-8") as f:
        qmsum_data = json.load(f)

    # QMSum format: list of meetings, each with 'meeting_transcripts' and 'topic_list'
    training_pairs = []

    for i, meeting in enumerate(qmsum_data[:max_samples]):
        # Build transcript from turns
        turns = meeting.get("meeting_transcripts", [])
        transcript_lines = []
        for turn in turns:
            speaker = turn.get("speaker", "Unknown")
            content = turn.get("content", "")
            transcript_lines.append(f"{speaker}: {content}")

        transcript = "\n".join(transcript_lines[:100])  # Limit to ~100 turns

        if len(transcript) < 100:
            continue

        logger.info(f"[{i + 1}/{min(len(qmsum_data), max_samples)}] Generating mind map...")

        # Use LLM to generate mind map in MindBench format
        mindmap_prompt = f"""Convert this meeting transcript into a mind map using MindBench token format.

Use these tokens: <s_map>, </s_map>, <s_node-N>, </s_node-N>, <s_text>, </s_text>, <sep/>, <s_relation>, </s_relation>

Where N is the depth level (0=root, 1=main topics, 2=subtopics, etc.)
Siblings at the same level are separated by <sep/>

TRANSCRIPT:
{transcript[:2000]}

Output ONLY the MindBench tokens, nothing else."""

        try:
            if backend == "claude":
                mindmap_tokens = _generate_with_claude(mindmap_prompt, api_key)
            else:
                mindmap_tokens = _generate_with_openai(mindmap_prompt, api_key)

            if "<s_map>" in mindmap_tokens:
                training_pairs.append({
                    "input": transcript,
                    "output": mindmap_tokens,
                    "source": "qmsum",
                })
        except Exception as e:
            logger.error(f"Failed for meeting {i}: {e}")
            continue

    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for pair in training_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.success(f"QMSum training data: {len(training_pairs)} pairs → {output_path}")
    return training_pairs
