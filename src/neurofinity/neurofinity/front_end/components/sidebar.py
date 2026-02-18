"""
Sidebar Component (Composite Pattern).

The sidebar is a composite of several widget groups (API status, backend
selector, theme picker) that together form the application's settings panel.
Each group is a private method that renders its own section of the sidebar.

Design Pattern: Composite
    Why: The sidebar is logically a single "settings panel," but it's built
    from independent widget groups. Each group (status indicator, backend
    dropdown, theme dropdown) can be developed, tested, and modified
    independently. The render() method composes them into the final sidebar.

    This is preferable to having one giant sidebar function because:
    - Adding a new setting (e.g., font size slider) only means adding
      one small method and one line in render()
    - Reordering settings is just moving lines in render()
    - Each widget group can be conditionally shown/hidden

The sidebar writes user selections to SessionState, which pages then read.
This avoids passing settings through function arguments across the app.

Usage:
    from components import SidebarComponent

    sidebar = SidebarComponent(api_client)
    sidebar.render()  # Call in app.py; populates the sidebar
"""

from __future__ import annotations

import streamlit as st

from neurofinity.front_end.config import AppConfig
from neurofinity.front_end.state import SessionState
from neurofinity.front_end.services.api_client import MindMapAPIClient


class SidebarComponent:
    """Renders the application sidebar with settings and status indicators.

    The sidebar serves three purposes:
    1. Shows whether the FastAPI backend is reachable (health indicator)
    2. Lets the user choose the AI backend (Claude, OpenAI, local)
    3. Lets the user pick an accessibility theme

    All selections are persisted to SessionState so they survive page
    navigations and Streamlit reruns.
    """

    def __init__(self, api_client: MindMapAPIClient) -> None:
        """Initialize with dependencies.

        Args:
            api_client: The API client singleton, used to check health
                        and discover available backends.
        """
        self._client = api_client
        self._config = AppConfig.get_instance()

    def render(self) -> None:
        """Render the complete sidebar. Call this once from app.py.

        The method composes all sidebar sections in order. Each section
        is a private method that renders its own widgets. This makes it
        easy to reorder sections or conditionally show/hide them.
        """
        with st.sidebar:
            st.header("Settings")
            self._render_api_status()
            st.divider()
            self._render_backend_selector()
            self._render_theme_selector()

    def _render_api_status(self) -> None:
        """Show a green/yellow indicator for the API connection status.

        This gives users immediate feedback about whether the backend
        is running, without them having to submit a transcript first
        and wait for an error.
        """
        is_healthy = self._client.is_healthy()
        if is_healthy:
            st.success("API connected", icon="✅")
        else:
            st.warning(
                "API not detected. Start it with:\n\n"
                "`uvicorn neurofinity.api:app --reload --port 8000`",
                icon="⚠️",
            )

    def _render_backend_selector(self) -> None:
        """Dropdown to choose the AI generation backend.

        Only shows backends that are actually available (i.e., have their
        API keys configured). If the API is unreachable and we can't
        discover backends, falls back to showing just "claude".
        """
        backends = self._client.get_available_backends()
        available = [name for name, info in backends.items() if info.get("available")]

        if not available:
            available = [self._config.DEFAULT_BACKEND]

        # Build human-readable labels with descriptions
        descriptions = {
            "claude": "Claude — highest quality",
            "openai": "OpenAI — good alternative",
            "local": "Local Flan-T5 — free, needs training",
        }

        current_backend = SessionState.get_backend()
        # Ensure the current selection is in the available list
        current_index = (
            available.index(current_backend) if current_backend in available else 0
        )

        selected = st.selectbox(
            "AI Backend",
            options=available,
            index=current_index,
            format_func=lambda x: descriptions.get(x, x),
            help="Which AI model generates the mind map structure.",
        )
        SessionState.set_backend(selected)

    def _render_theme_selector(self) -> None:
        """Dropdown to choose the accessibility color theme.

        Each theme is designed for a specific neurodivergent need.
        The help text explains what each theme is optimized for.
        """
        descriptions = {
            "high_contrast": "High Contrast — maximum readability",
            "pastel": "Pastel — gentle on eyes",
            "dark": "Dark — low-light comfort",
            "minimal": "Minimal — clean focus",
        }

        themes = self._config.theme_names
        current_theme = SessionState.get_theme()
        current_index = themes.index(current_theme) if current_theme in themes else 0

        selected = st.selectbox(
            "Accessibility Theme",
            options=themes,
            index=current_index,
            format_func=lambda x: descriptions.get(x, x),
            help=(
                "Themes are designed for neurodivergent accessibility. "
                "High contrast for visual processing, pastel for light "
                "sensitivity, dark for low-light, minimal for focus."
            ),
        )
        SessionState.set_theme(selected)
