"""
MindMapGenerator: Multi-backend mind map generation from meeting transcripts.

This is the core class that your FastAPI endpoints and Streamlit UI will call.
It supports three generation backends that can be switched at runtime:

  1. Claude API (Anthropic) — Best quality. Uses claude-sonnet-4-20250514 for
     cost-effective structured generation with explicit JSON output.
  2. OpenAI API — Alternative using GPT-4o-mini for structured mind maps.
  3. Local Flan-T5 + LoRA — Free inference after fine-tuning. Uses your
     custom LoRA adapter trained on MindBench + meeting data.

The generator always returns a MindMapOutput schema regardless of backend,
making the frontend completely backend-agnostic.
"""

import json
import os
from typing import Optional

from loguru import logger

from neurofinity.mindmap.schemas import MindMapNode, MindMapOutput, AccessibilityConfig
from neurofinity.mindmap.converter import json_to_mermaid


# ---------------------------------------------------------------------------
# System prompt shared across API backends
# ---------------------------------------------------------------------------
MINDMAP_SYSTEM_PROMPT = """You are an expert meeting analyst that creates structured mind maps from meeting transcripts.

Given a meeting transcript, extract and organize the content into a hierarchical mind map structure.

OUTPUT FORMAT: Return ONLY valid JSON (no markdown, no backticks) matching this exact schema:
{
  "text": "Meeting Title / Central Topic",
  "node_type": "topic",
  "children": [
    {
      "text": "Main Topic 1",
      "node_type": "topic",
      "speaker": "Name or null",
      "children": [
        {
          "text": "Subtopic or detail",
          "node_type": "topic|decision|action_item|insight|question",
          "speaker": "Name or null",
          "children": []
        }
      ]
    }
  ]
}

RULES:
1. The root node should capture the meeting's central theme or title.
2. First-level children = major discussion topics.
3. Second-level children = specific points, details, arguments under each topic.
4. Use node_type to classify: "decision" for agreed outcomes, "action_item" for tasks
   assigned (include owner in speaker field), "question" for unresolved questions,
   "insight" for key observations.
5. Keep text concise (under 20 words per node).
6. Preserve speaker attribution when identifiable.
7. Aim for 3-6 top-level branches, with 2-5 children each.
8. Return ONLY the JSON object, nothing else."""


class MindMapGenerator:
    """Generate mind maps from meeting transcripts using multiple backends.

    Usage:
        # Using Claude API (default, highest quality):
        gen = MindMapGenerator(backend="claude", api_key="sk-ant-...")
        result = gen.generate("John: Let's discuss the roadmap...")

        # Using OpenAI:
        gen = MindMapGenerator(backend="openai", api_key="sk-...")
        result = gen.generate(transcript_text)

        # Using local fine-tuned Flan-T5 (no API key needed):
        gen = MindMapGenerator(backend="local", model_path="./models/mindmap-lora")
        result = gen.generate(transcript_text)
    """

    SUPPORTED_BACKENDS = ("claude", "openai", "local")

    def __init__(
        self,
        backend: str = "claude",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize the generator with the specified backend.

        Args:
            backend: One of "claude", "openai", or "local".
            api_key: API key for Claude or OpenAI. Falls back to env vars
                     ANTHROPIC_API_KEY or OPENAI_API_KEY if not provided.
            model_path: Path to local LoRA adapter directory (for "local" backend).
        """
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend '{backend}'. Choose from: {self.SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.api_key = api_key
        self.model_path = model_path

        # Lazy-loaded resources (initialized on first generate call)
        self._client = None
        self._local_model = None
        self._local_tokenizer = None

        logger.info(f"MindMapGenerator initialized with backend='{backend}'")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        transcript: str,
        title: str = "Meeting Mind Map",
        accessibility: Optional[AccessibilityConfig] = None,
    ) -> MindMapOutput:
        """Generate a mind map from a meeting transcript.

        Args:
            transcript: The meeting transcript text.
            title: Title for the mind map.
            accessibility: Optional accessibility configuration.

        Returns:
            MindMapOutput with the hierarchical tree, Mermaid code, and metadata.
        """
        if not transcript or not transcript.strip():
            logger.warning("Empty transcript received, returning minimal mind map")
            return self._empty_mindmap(title, accessibility)

        logger.info(
            f"Generating mind map via '{self.backend}' backend "
            f"({len(transcript)} chars input)"
        )

        # Route to the appropriate backend
        if self.backend == "claude":
            root_dict = self._generate_claude(transcript)
        elif self.backend == "openai":
            root_dict = self._generate_openai(transcript)
        else:
            root_dict = self._generate_local(transcript)

        # Parse the JSON dict into our Pydantic schema
        root_node = self._parse_node(root_dict)

        # Generate Mermaid code for frontend rendering
        mermaid = json_to_mermaid(root_node)

        acc = accessibility or AccessibilityConfig()

        output = MindMapOutput(
            root=root_node,
            title=title,
            backend_used=self.backend,
            mermaid_code=mermaid,
            accessibility=acc,
        )
        logger.success(
            f"Mind map generated: {len(output.to_flat_topics())} nodes, "
            f"backend={self.backend}"
        )
        return output

    # ------------------------------------------------------------------
    # Claude API Backend
    # ------------------------------------------------------------------

    def _generate_claude(self, transcript: str) -> dict:
        """Generate mind map JSON using Anthropic Claude API."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Install anthropic SDK: pip install anthropic --break-system-packages"
            )

        if self._client is None:
            key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Set ANTHROPIC_API_KEY env var or pass api_key to constructor"
                )
            self._client = anthropic.Anthropic(api_key=key)

        # Truncate to ~3000 words to stay within cost budget
        words = transcript.split()
        if len(words) > 3000:
            logger.warning(
                f"Transcript truncated from {len(words)} to 3000 words for API call"
            )
            transcript = " ".join(words[:3000])

        response = self._client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=MINDMAP_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Generate a mind map from this meeting transcript:\n\n{transcript}",
                }
            ],
        )

        raw_text = response.content[0].text.strip()
        return self._safe_parse_json(raw_text)

    # ------------------------------------------------------------------
    # OpenAI API Backend
    # ------------------------------------------------------------------

    def _generate_openai(self, transcript: str) -> dict:
        """Generate mind map JSON using OpenAI API."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "Install openai SDK: pip install openai --break-system-packages"
            )

        key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Set OPENAI_API_KEY env var or pass api_key to constructor"
            )

        client = openai.OpenAI(api_key=key)

        words = transcript.split()
        if len(words) > 3000:
            transcript = " ".join(words[:3000])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": MINDMAP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Generate a mind map from this meeting transcript:\n\n{transcript}",
                },
            ],
            max_tokens=2048,
            temperature=0.3,
        )

        raw_text = response.choices[0].message.content.strip()
        return self._safe_parse_json(raw_text)

    # ------------------------------------------------------------------
    # Local Flan-T5 + LoRA Backend
    # ------------------------------------------------------------------

    def _generate_local(self, transcript: str) -> dict:
        """Generate mind map JSON using local fine-tuned Flan-T5 + LoRA."""
        if self._local_model is None:
            self._load_local_model()

        # Prepare input prompt matching our training format
        prompt = (
            "Generate a structured mind map from this meeting transcript. "
            "Output JSON with text, node_type, speaker, and children fields.\n\n"
            f"Transcript: {transcript[:2048]}"
        )

        inputs = self._local_tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        )

        # Move inputs to the same device as the model
        device = next(self._local_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = self._local_model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        raw_text = self._local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._safe_parse_json(raw_text)

    def _load_local_model(self):
        """Load the fine-tuned Flan-T5 model with LoRA adapter."""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            from peft import PeftModel
        except ImportError:
            raise ImportError(
                "Install dependencies: pip install transformers peft "
                "accelerate --break-system-packages"
            )

        model_path = self.model_path or os.getenv(
            "MINDMAP_MODEL_PATH", "./models/mindmap-lora"
        )

        logger.info(f"Loading local Flan-T5 + LoRA model from: {model_path}")

        # Check if it's a LoRA adapter or a full model
        base_model_name = "google/flan-t5-large"

        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # It's a LoRA adapter — load base model then attach adapter
            logger.info("Detected LoRA adapter, loading base model + adapter")
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            self._local_model = PeftModel.from_pretrained(base_model, model_path)
            self._local_tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        elif os.path.exists(os.path.join(model_path, "config.json")):
            # It's a full merged model
            logger.info("Detected merged model, loading directly")
            self._local_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self._local_tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            # Fall back to base Flan-T5 without fine-tuning
            logger.warning(
                f"No model found at {model_path}, falling back to base flan-t5-large. "
                "Results will be poor without fine-tuning!"
            )
            self._local_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            self._local_tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        self._local_model.eval()
        logger.success("Local model loaded successfully")

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_parse_json(raw_text: str) -> dict:
        """Parse JSON from model output, handling common formatting issues."""
        # Strip markdown code fences if present
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]  # remove first line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # Remove common JSON-breaking prefixes
        for prefix in ("json", "JSON"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw output: {text[:500]}")
            # Return a fallback structure so the pipeline doesn't crash
            return {
                "text": "Mind Map (parse error)",
                "node_type": "topic",
                "children": [
                    {
                        "text": text[:200] if text else "Generation failed",
                        "node_type": "topic",
                        "children": [],
                    }
                ],
            }

    @staticmethod
    def _parse_node(data: dict) -> MindMapNode:
        """Recursively parse a dict into a MindMapNode tree."""
        children = [
            MindMapGenerator._parse_node(c) for c in data.get("children", [])
        ]
        return MindMapNode(
            text=data.get("text", "Untitled"),
            children=children,
            node_type=data.get("node_type", "topic"),
            speaker=data.get("speaker"),
            relation=data.get("relation"),
        )

    @staticmethod
    def _empty_mindmap(
        title: str, accessibility: Optional[AccessibilityConfig] = None
    ) -> MindMapOutput:
        """Return a minimal mind map for empty inputs."""
        root = MindMapNode(
            text="No content available",
            node_type="topic",
            children=[
                MindMapNode(
                    text="Please provide a meeting transcript",
                    node_type="insight",
                )
            ],
        )
        return MindMapOutput(
            root=root,
            title=title,
            backend_used="none",
            mermaid_code="mindmap\n  No content available\n    Please provide a transcript",
            accessibility=accessibility or AccessibilityConfig(),
        )
