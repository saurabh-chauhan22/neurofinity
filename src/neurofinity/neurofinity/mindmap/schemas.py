"""
Pydantic schemas for mind map data structures.

These schemas define the canonical representation of a mind map used
throughout the Neurofinity pipeline. The structure is designed to be:
  - Serializable to JSON for API responses
  - Convertible to Mermaid syntax for rendering
  - Convertible to MindBench token format for model training
  - Compatible with accessibility customizations (themes, spacing)
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class MindMapNode(BaseModel):
    """A single node in the mind map tree.

    Each node has a text label and can contain child nodes (subtopics)
    and optional metadata like the speaker who raised the topic or
    a relationship to another node.
    """

    text: str = Field(..., description="The text content of this node")
    children: list[MindMapNode] = Field(
        default_factory=list,
        description="Child nodes (subtopics, action items, details)",
    )
    node_type: str = Field(
        default="topic",
        description="Type: topic, decision, action_item, insight, question",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker who raised this point (if identifiable)",
    )
    relation: Optional[str] = Field(
        default=None,
        description="Cross-branch relationship label (MindBench-style)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Q3 Budget Review",
                "node_type": "topic",
                "children": [
                    {
                        "text": "Marketing: increase by 15%",
                        "node_type": "decision",
                        "speaker": "Sarah",
                        "children": [],
                    },
                    {
                        "text": "John to prepare revised forecast by Friday",
                        "node_type": "action_item",
                        "speaker": "John",
                        "children": [],
                    },
                ],
            }
        }


class MindMapOutput(BaseModel):
    """Complete mind map output from the generation pipeline.

    Contains the hierarchical tree structure plus metadata about
    the generation process and accessibility configuration.
    """

    root: MindMapNode = Field(..., description="Root node of the mind map tree")
    title: str = Field(
        default="Meeting Mind Map",
        description="Title displayed above the mind map",
    )
    backend_used: str = Field(
        default="unknown",
        description="Which backend generated this: claude, openai, local_flan_t5",
    )
    mermaid_code: Optional[str] = Field(
        default=None,
        description="Pre-rendered Mermaid diagram code",
    )
    accessibility: AccessibilityConfig = Field(
        default_factory=lambda: AccessibilityConfig(),
        description="Accessibility customization settings",
    )

    def to_flat_topics(self) -> list[str]:
        """Flatten the tree into a list of all topic strings."""
        topics = []

        def _walk(node: MindMapNode):
            topics.append(node.text)
            for child in node.children:
                _walk(child)

        _walk(self.root)
        return topics


class AccessibilityConfig(BaseModel):
    """Accessibility settings for neurodivergent-friendly rendering.

    These settings control visual presentation of the mind map to
    support different cognitive processing styles.
    """

    theme: str = Field(
        default="high_contrast",
        description="Visual theme: high_contrast, pastel, dark, minimal",
    )
    font_size: int = Field(default=14, description="Base font size in pixels")
    node_spacing: float = Field(
        default=1.5, description="Multiplier for space between nodes"
    )
    max_depth_visible: int = Field(
        default=4, description="Max tree depth shown initially (rest collapsed)"
    )
    color_by_type: bool = Field(
        default=True,
        description="Color-code nodes by type (decision=green, action=orange, etc.)",
    )
