"""
Mind Map Renderer Component (Strategy Pattern).

This component converts a mind map JSON tree into a Graphviz DOT string
for rendering with st.graphviz_chart(). The rendering strategy (colors,
fonts, layout direction) changes based on the selected accessibility theme.

Design Pattern: Strategy Pattern
    Why: The rendering algorithm is the same regardless of theme (recursively
    walk the tree, emit DOT nodes and edges), but the VISUAL STRATEGY differs
    (which colors, which font sizes, which background). The Strategy Pattern
    lets us swap theme "strategies" without changing the traversal logic.

    In classic OOP, you'd define a ThemeStrategy interface with subclasses.
    Here, we use the simpler data-driven variant: ThemeColors dataclasses
    (defined in config.py) act as strategy objects, and the renderer selects
    the right one at runtime. The effect is identical — the rendering code
    is decoupled from the color decisions — but with less boilerplate.

    This approach also makes adding new themes trivial: just add a new
    ThemeColors entry in config.py. No renderer code changes needed.

Usage:
    from components import MindMapRenderer

    renderer = MindMapRenderer()
    dot_code = renderer.to_graphviz(tree_dict, theme="pastel")
    st.graphviz_chart(dot_code)
"""

from __future__ import annotations

from config import AppConfig, ThemeColors


class MindMapRenderer:
    """Converts mind map tree dicts to Graphviz DOT strings.

    This class is stateless — each call to to_graphviz() is independent.
    The class structure exists to group related rendering methods and to
    make future extensions (to_mermaid, to_svg, etc.) natural.

    The rendering process has two phases:
    1. Tree Traversal — recursively walk the JSON tree, assign unique IDs
    2. DOT Generation — emit Graphviz statements for each node and edge

    The theme (a ThemeColors instance from config.py) controls all visual
    decisions. The traversal logic never changes.
    """

    def __init__(self) -> None:
        self._config = AppConfig.get_instance()

    def to_graphviz(self, tree: dict, theme: str = "high_contrast") -> str:
        """Convert a mind map tree dict to a complete Graphviz DOT string.

        Args:
            tree: The mind map tree from the API response. Expected shape:
                  {"text": "Root Topic", "node_type": "topic", "children": [...]}
            theme: Name of the accessibility theme to apply. Must be a key
                   in config.THEMES (high_contrast, pastel, dark, minimal).

        Returns:
            A complete DOT language string that st.graphviz_chart() can render.
            The string defines a directed graph with styled nodes and edges.

        Example output:
            digraph G {
                bgcolor="#1a1a2e";
                ...
                "n1" [label="Budget Review", shape=box, ...];
                "n1" -> "n2" [color="#e94560", ...];
            }
        """
        colors = self._config.get_theme(theme)
        # Counter is wrapped in a list so the nested _walk function can
        # mutate it (Python closures can read enclosing variables but
        # can't rebind them without `nonlocal`, and a list avoids that).
        counter = [0]

        def _make_id() -> str:
            counter[0] += 1
            return f"n{counter[0]}"

        def _walk(node: dict, parent_id: str | None = None) -> list[str]:
            """Recursively emit DOT statements for a node and its children.

            Each call produces:
              - One node declaration (with label, shape, colors)
              - One edge from parent → this node (unless root)
              - Recursive calls for all children

            The node_type field drives the fill color via the theme strategy.
            """
            text = node.get("text", "")
            children = node.get("children", [])
            node_type = node.get("node_type", "topic")

            nid = _make_id()
            fill = colors.get_fill(node_type)
            # Truncate long labels to keep the graph readable
            label = text.replace('"', "'")[: self._config.MAX_LABEL_LENGTH]

            stmts = []

            # Node declaration with rounded box styling
            stmts.append(
                f'  "{nid}" [label="{label}", shape=box, '
                f'style="filled,rounded", fillcolor="{fill}", '
                f'fontcolor="{colors.font}", fontsize="14", '
                f'fontname="Arial"];'
            )

            # Edge from parent to this node (skip for root)
            if parent_id:
                stmts.append(
                    f'  "{parent_id}" -> "{nid}" '
                    f'[color="{colors.edge}", penwidth=1.5];'
                )

            # Recurse into children
            for child in children:
                stmts.extend(_walk(child, nid))

            return stmts

        # Assemble the complete DOT graph
        header = (
            f'digraph G {{\n'
            f'    bgcolor="{colors.bg}";\n'
            f'    fontname="Arial";\n'
            f'    node [fontname="Arial"];\n'
            f'    edge [arrowhead=normal];\n'
            f'    rankdir=LR;\n'
        )
        body = "\n".join(_walk(tree))
        return header + body + "\n}"

    @staticmethod
    def count_nodes(tree: dict) -> int:
        """Count total nodes in a mind map tree (useful for metrics display).

        Performs a simple recursive traversal. This is used by the mind map
        page to show a "Total Nodes" metric even when the API response
        doesn't include it.
        """
        count = 1  # Count this node
        for child in tree.get("children", []):
            count += MindMapRenderer.count_nodes(child)
        return count
