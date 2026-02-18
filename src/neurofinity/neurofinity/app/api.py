"""
Neurofinity FastAPI Application.

This module provides REST API endpoints for the Neurofinity mind map
generation pipeline. It exposes:

  POST /mindmap/generate       — Generate mind map from transcript text
  POST /mindmap/from-audio     — Upload audio → transcribe → mind map
  GET  /mindmap/backends       — List available generation backends
  GET  /health                 — Health check

The API returns structured JSON including the mind map tree, Mermaid
diagram code, and accessibility configuration — everything the frontend
needs to render the visualization.
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

from neurofinity.mindmap.generator import MindMapGenerator
from neurofinity.mindmap.schemas import AccessibilityConfig
from neurofinity.audio import AudioToText

# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neurofinity API",
    description="AI-powered mind map generation for neurodivergent accessibility",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Schemas
# ---------------------------------------------------------------------------

class MindMapRequest(BaseModel):
    """Request body for mind map generation."""
    text: str = Field(..., description="Meeting transcript text")
    title: str = Field(default="Meeting Mind Map", description="Title for the mind map")
    backend: str = Field(
        default="claude",
        description="Generation backend: claude, openai, or local",
    )
    theme: str = Field(
        default="high_contrast",
        description="Accessibility theme: high_contrast, pastel, dark, minimal",
    )
    font_size: int = Field(default=14, description="Base font size in pixels")
    node_spacing: float = Field(default=1.5, description="Node spacing multiplier")


class MindMapResponse(BaseModel):
    """Response containing the generated mind map."""
    title: str
    backend_used: str
    mermaid_code: Optional[str]
    tree: dict  # The full MindMapNode tree as a dict
    total_nodes: int
    accessibility: dict


# ---------------------------------------------------------------------------
# Lazy-loaded generator instances (one per backend)
# ---------------------------------------------------------------------------

_generators: dict[str, MindMapGenerator] = {}


def get_generator(backend: str) -> MindMapGenerator:
    """Get or create a MindMapGenerator for the specified backend."""
    if backend not in _generators:
        _generators[backend] = MindMapGenerator(backend=backend)
    return _generators[backend]


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "neurofinity-api"}


@app.get("/mindmap/backends")
def list_backends():
    """List available mind map generation backends and their status."""
    backends = {
        "claude": {
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
            "description": "Anthropic Claude API — highest quality output",
        },
        "openai": {
            "available": bool(os.getenv("OPENAI_API_KEY")),
            "description": "OpenAI GPT-4o-mini — good alternative",
        },
        "local": {
            "available": os.path.exists("./models/mindmap-lora/adapter_config.json"),
            "description": "Local Flan-T5 + LoRA — free, requires fine-tuning",
        },
    }
    return {"backends": backends}


@app.post("/mindmap/generate", response_model=MindMapResponse)
def generate_mindmap(request: MindMapRequest):
    """Generate a mind map from a meeting transcript.

    This is the primary endpoint. Send a transcript and get back a
    structured mind map with Mermaid rendering code.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Transcript text cannot be empty")

    try:
        generator = get_generator(request.backend)
        accessibility = AccessibilityConfig(
            theme=request.theme,
            font_size=request.font_size,
            node_spacing=request.node_spacing,
        )

        result = generator.generate(
            transcript=request.text,
            title=request.title,
            accessibility=accessibility,
        )

        return MindMapResponse(
            title=result.title,
            backend_used=result.backend_used,
            mermaid_code=result.mermaid_code,
            tree=result.root.model_dump(),
            total_nodes=len(result.to_flat_topics()),
            accessibility=result.accessibility.model_dump(),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Mind map generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/mindmap/from-audio")
async def mindmap_from_audio(
    file: UploadFile = File(...),
    backend: str = "claude",
    theme: str = "high_contrast",
):
    """Upload an audio file → transcribe → generate mind map.

    This endpoint chains the audio-to-text and mind map generation
    pipelines together for a single-step workflow.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save uploaded file temporarily
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Step 1: Transcribe audio
        
        transcriber = AudioToText(audio_path=temp_path)
        transcript_response = transcriber.output_text()

        if hasattr(transcript_response, "json"):
            transcript_data = transcript_response.json()
            transcript = transcript_data.get("text", str(transcript_data))
        else:
            transcript = str(transcript_response)

        # Step 2: Generate mind map
        generator = get_generator(backend)
        result = generator.generate(
            transcript=transcript,
            title=f"Meeting: {file.filename}",
            accessibility=AccessibilityConfig(theme=theme),
        )

        return {
            "transcript": transcript,
            "mindmap": MindMapResponse(
                title=result.title,
                backend_used=result.backend_used,
                mermaid_code=result.mermaid_code,
                tree=result.root.model_dump(),
                total_nodes=len(result.to_flat_topics()),
                accessibility=result.accessibility.model_dump(),
            ).model_dump(),
        }

    except Exception as e:
        logger.error(f"Audio → mind map pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
