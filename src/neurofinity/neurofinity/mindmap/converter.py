"""
Format converters for mind map data.

This module handles conversion between three representations:
  1. JSON (Pydantic MindMapNode) — canonical internal format
  2. Mermaid syntax — for frontend diagram rendering
  3. MindBench tokens — for training Flan-T5 with LoRA

The key insight: train on MindBench tokens (explicit structural markers
like <s_node-0> make depth unambiguous for the model), then convert
to Mermaid or JSON for rendering. This sidesteps T5's weakness with
whitespace-sensitive formats.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger


def json_to_mermaid(node, indent: int = 0) -> str:
    """Convert a MindMapNode tree to Mermaid mindmap syntax.

    Args:
        node: A MindMapNode (or dict with text/children fields).
        indent: Current indentation level (used for recursion).

    Returns:
        A string of valid Mermaid mindmap code.
    """
    # Handle both MindMapNode objects and raw dicts
    text = node.text if hasattr(node, "text") else node.get("text", "Untitled")
    children = (
        node.children if hasattr(node, "children") else node.get("children", [])
    )
    node_type = (
        node.node_type if hasattr(node, "node_type") else node.get("node_type", "topic")
    )

    # Clean text for Mermaid (remove special chars that break syntax)
    clean_text = text.replace('"', "'").replace("\n", " ").strip()

    # Add type emoji prefix for accessibility color-coding
    type_prefix = {
        "decision": "✅ ",
        "action_item": "📋 ",
        "question": "❓ ",
        "insight": "💡 ",
    }.get(node_type, "")

    lines = []
    if indent == 0:
        # Root node - start the Mermaid mindmap block
        lines.append("mindmap")
        lines.append(f"  {type_prefix}{clean_text}")
        child_indent = 2
    else:
        spaces = "  " * (indent + 1)
        lines.append(f"{spaces}{type_prefix}{clean_text}")
        child_indent = indent + 1

    for child in children:
        lines.append(json_to_mermaid(child, child_indent))

    return "\n".join(lines)


def json_to_graphviz(node, theme: str = "high_contrast") -> str:
    """Convert a MindMapNode tree to Graphviz DOT syntax.

    This generates the same style of Graphviz output your existing
    neuroui.py uses, but from the structured mind map data.

    Args:
        node: A MindMapNode (or dict with text/children).
        theme: Color theme — high_contrast, pastel, dark, or minimal.

    Returns:
        A valid Graphviz DOT string for st.graphviz_chart().
    """
    # Theme color palettes
    themes = {
        "high_contrast": {
            "bg": "#1a1a2e",
            "root": "#e94560",
            "topic": "#0f3460",
            "decision": "#16a085",
            "action_item": "#e67e22",
            "question": "#8e44ad",
            "insight": "#2980b9",
            "edge": "#e94560",
            "font": "white",
        },
        "pastel": {
            "bg": "#fef9ef",
            "root": "#f4a261",
            "topic": "#8ecae6",
            "decision": "#95d5b2",
            "action_item": "#ffb4a2",
            "question": "#cdb4db",
            "insight": "#a2d2ff",
            "edge": "#264653",
            "font": "#264653",
        },
        "dark": {
            "bg": "#0d1117",
            "root": "#58a6ff",
            "topic": "#388bfd",
            "decision": "#3fb950",
            "action_item": "#d29922",
            "question": "#bc8cff",
            "insight": "#79c0ff",
            "edge": "#58a6ff",
            "font": "#c9d1d9",
        },
        "minimal": {
            "bg": "#ffffff",
            "root": "#333333",
            "topic": "#666666",
            "decision": "#2d6a4f",
            "action_item": "#b56727",
            "question": "#7b2cbf",
            "insight": "#1d3557",
            "edge": "#999999",
            "font": "#333333",
        },
    }

    colors = themes.get(theme, themes["high_contrast"])
    node_counter = [0]  # mutable counter for unique IDs

    def _get_node_id():
        node_counter[0] += 1
        return f"node_{node_counter[0]}"

    def _get_color(ntype: str) -> str:
        return colors.get(ntype, colors["topic"])

    def _walk(n, parent_id: Optional[str] = None) -> list[str]:
        """Recursively build DOT statements."""
        text = n.text if hasattr(n, "text") else n.get("text", "")
        children = n.children if hasattr(n, "children") else n.get("children", [])
        ntype = n.node_type if hasattr(n, "node_type") else n.get("node_type", "topic")

        nid = _get_node_id()
        fill = _get_color(ntype)
        label = text.replace('"', "'")[:60]  # Truncate long labels

        stmts = []
        stmts.append(
            f'  "{nid}" [label="{label}", shape=box, style="filled,rounded", '
            f'fillcolor="{fill}", fontcolor="{colors["font"]}", '
            f'fontsize="14", fontname="Arial"];'
        )

        if parent_id:
            stmts.append(
                f'  "{parent_id}" -> "{nid}" '
                f'[color="{colors["edge"]}", penwidth=1.5];'
            )

        for child in children:
            stmts.extend(_walk(child, nid))

        return stmts

    root_text = node.text if hasattr(node, "text") else node.get("text", "Mind Map")
    header = f"""digraph G {{
    bgcolor="{colors['bg']}";
    fontname="Arial";
    node [fontname="Arial"];
    edge [arrowhead=normal];
    rankdir=LR;
"""
    body_lines = _walk(node)
    return header + "\n".join(body_lines) + "\n}"


# ---------------------------------------------------------------------------
# MindBench Token Format Converters
# ---------------------------------------------------------------------------
# These are used for training data preparation. MindBench uses a custom
# token format with explicit depth markers that Flan-T5 learns efficiently.

def json_to_mindbench_tokens(node, depth: int = 0) -> str:
    """Convert a MindMapNode tree to MindBench token sequence.

    The MindBench format uses tags like <s_node-0>, <s_text>, <sep/>,
    <s_relation> to encode the tree structure explicitly. This is far
    more learnable for T5 than whitespace-based Mermaid syntax.

    Example output:
        <s_map><s_node-0><s_text>Budget Review</s_text>
        <s_node-1><s_text>Marketing</s_text><sep/>
        <s_text>Engineering</s_text></s_node-1></s_node-0></s_map>
    """
    text = node.text if hasattr(node, "text") else node.get("text", "")
    children = node.children if hasattr(node, "children") else node.get("children", [])
    relation = node.relation if hasattr(node, "relation") else node.get("relation")

    parts = []

    if depth == 0:
        parts.append("<s_map>")

    parts.append(f"<s_node-{depth}>")
    parts.append(f"<s_text>{text}</s_text>")

    if relation:
        parts.append(f"<s_relation>{relation}</s_relation>")

    # Process children
    if children:
        child_parts = []
        for child in children:
            child_parts.append(json_to_mindbench_tokens(child, depth + 1))
        # Join siblings with <sep/> separator
        child_str = "<sep/>".join(child_parts)
        # But the first child doesn't need a leading separator
        parts.append(child_str)

    parts.append(f"</s_node-{depth}>")

    if depth == 0:
        parts.append("</s_map>")

    return "".join(parts)


def mindbench_tokens_to_json(token_str: str) -> dict:
    """Parse MindBench token sequence back into a JSON dict.

    This is used for:
    - Evaluating model outputs during training
    - Converting model predictions to the canonical format
    - Validating training data integrity

    Args:
        token_str: MindBench format string (e.g., "<s_map><s_node-0>...")

    Returns:
        A nested dict with text, children, and optional relation fields.
    """
    # Strip the outer <s_map>...</s_map> wrapper
    text = token_str.strip()
    text = re.sub(r"^<s_map>", "", text)
    text = re.sub(r"</s_map>$", "", text)

    try:
        return _parse_mindbench_node(text)
    except Exception as e:
        logger.error(f"MindBench token parsing failed: {e}")
        return {"text": "Parse Error", "children": []}


def _parse_mindbench_node(text: str) -> dict:
    """Recursively parse a single node from MindBench tokens."""
    # Extract the node depth level
    depth_match = re.match(r"<s_node-(\d+)>", text)
    if not depth_match:
        # Might be raw text content
        return {"text": text.strip(), "children": []}

    depth = int(depth_match.group(1))

    # Remove the outer node tags
    inner = re.sub(r"^<s_node-\d+>", "", text)
    inner = re.sub(r"</s_node-\d+>$", "", inner)

    # Extract the text content
    text_match = re.search(r"<s_text>(.*?)</s_text>", inner)
    node_text = text_match.group(1) if text_match else ""

    # Extract optional relation
    rel_match = re.search(r"<s_relation>(.*?)</s_relation>", inner)
    relation = rel_match.group(1) if rel_match else None

    # Find child nodes (next depth level)
    children = []
    child_depth = depth + 1
    child_pattern = f"<s_node-{child_depth}>"

    if child_pattern in inner:
        # Split by <sep/> at the same level to get sibling groups
        # This is a simplified parser; a full implementation would use
        # a proper recursive descent parser for nested structures
        child_sections = _split_siblings(inner, child_depth)
        for section in child_sections:
            child = _parse_mindbench_node(section)
            children.append(child)

    result = {"text": node_text, "children": children, "node_type": "topic"}
    if relation:
        result["relation"] = relation

    return result


def _split_siblings(text: str, depth: int) -> list[str]:
    """Split text into sibling node sections at the given depth.

    Handles the <sep/> delimiter between sibling nodes while respecting
    nesting (doesn't split on <sep/> inside deeper nodes).
    """
    tag_open = f"<s_node-{depth}>"
    tag_close = f"</s_node-{depth}>"

    sections = []
    pos = 0

    while True:
        start = text.find(tag_open, pos)
        if start == -1:
            break

        # Find the matching close tag (accounting for nesting)
        nesting = 0
        i = start
        while i < len(text):
            if text[i:].startswith(tag_open):
                nesting += 1
                i += len(tag_open)
            elif text[i:].startswith(tag_close):
                nesting -= 1
                if nesting == 0:
                    end = i + len(tag_close)
                    sections.append(text[start:end])
                    pos = end
                    break
                i += len(tag_close)
            else:
                i += 1
        else:
            break

    return sections
