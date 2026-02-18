"""
Reusable UI Components.

This package contains self-contained UI components that can be composed
into pages. Each component is responsible for rendering a specific piece
of the interface (sidebar, mind map visualization, etc.) but does NOT
manage page routing or application state directly.

Components receive data as arguments and emit results through return values
or session state updates. This makes them reusable — the same MindMapRenderer
could be used on the mind map page, an evaluation page, or a comparison page.
"""

from .sidebar import SidebarComponent
from .mindmap_renderer import MindMapRenderer

__all__ = ["SidebarComponent", "MindMapRenderer"]
