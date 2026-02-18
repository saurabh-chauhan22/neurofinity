"""
Mind Map Page — Visualization and Export.

This is the application's main output page. It ties together all three
architectural layers:

  Service Layer  → MindMapAPIClient fetches the mind map from FastAPI
  Component      → MindMapRenderer converts the tree to Graphviz DOT
  State          → SessionState caches the result across Streamlit reruns

The page handles a critical Streamlit gotcha: every user interaction
(clicking a download button, toggling an expander) triggers a full script
rerun. Without caching, each rerun would re-call the API and regenerate
the mind map. We avoid this by storing the API result in SessionState
after the first successful generation, and checking for it on subsequent
reruns.

Page Flow:
  1. Read transcript from SessionState (set by home/processing page)
  2. Check if we already have a cached result (from a previous rerun)
  3. If not, call the API → cache the result
  4. Render the mind map using MindMapRenderer
  5. Show metrics (node count, backend, theme)
  6. Provide export buttons (Mermaid, JSON, DOT)
  7. Navigation button to generate another
"""

from __future__ import annotations

import json

import streamlit as st

from pages.base_page import BasePage
from state import SessionState, PageName
from components.mindmap_renderer import MindMapRenderer


class MindMapPage(BasePage):
    """Displays the generated mind map with export and navigation controls.

    This page demonstrates the benefit of the Service Layer separation:
    the page doesn't know or care HOW the mind map was generated (API call,
    local model, mock data). It just receives a tree dict and renders it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._renderer = MindMapRenderer()

    @property
    def page_title(self) -> str:
        return "Your Mind Map"

    def _render_content(self) -> None:
        """Main rendering logic with API result caching.

        The caching strategy here is important for Streamlit:

        First rerun (after navigating from home page):
          - No cached result → call API → store result → render

        Subsequent reruns (user clicks download button, expander, etc.):
          - Cached result found → skip API call → render from cache

        This means the API is called exactly ONCE per mind map generation,
        not on every rerun. The cache is cleared when the user clicks
        "Generate Another" (via SessionState.reset_for_new_session).
        """
        # Guard: need a transcript to proceed
        if not SessionState.has_transcript():
            st.warning("No transcript available. Please upload audio or paste text.")
            if st.button("Back to Home"):
                SessionState.navigate_to(PageName.HOME)
            return

        # Show transcript in a collapsible section for reference
        with st.expander("View Transcript", expanded=False):
            st.text(SessionState.get_transcript())

        # Check for cached result (avoids re-calling API on every rerun)
        result = SessionState.get_mindmap_result()

        if result is None:
            # First time on this page — call the API
            result = self._fetch_mindmap()
            if result is None:
                # API call failed — error already shown by _fetch_mindmap
                return
            # Cache the result for subsequent reruns
            SessionState.set_mindmap_result(result)

        # Render the mind map visualization
        tree = result.get("tree", {})
        theme = SessionState.get_theme()

        dot_code = self._renderer.to_graphviz(tree, theme=theme)
        st.graphviz_chart(dot_code)

        # Show metadata metrics
        self._render_metrics(result)

        # Export controls
        st.divider()
        self._render_exports(result, tree, dot_code)

    def _render_footer(self) -> None:
        """Navigation controls at the bottom of the page."""
        st.divider()
        if st.button("Generate Another", type="primary"):
            SessionState.reset_for_new_session()

    # --- Private Helpers ---

    def _fetch_mindmap(self) -> dict | None:
        """Call the API to generate a mind map from the current transcript.

        Uses the service layer (MindMapAPIClient) so this page never
        touches HTTP directly. The spinner provides feedback during the
        potentially slow API call (10-30 seconds for Claude backend).

        Returns:
            The API response dict on success, None on failure.
        """
        backend = SessionState.get_backend()
        theme = SessionState.get_theme()

        with st.spinner(f"Generating mind map via {backend} backend..."):
            api_result = self._client.generate_mindmap(
                transcript=SessionState.get_transcript(),
                title="Meeting Mind Map",
                backend=backend,
                theme=theme,
            )

        if api_result.success:
            return api_result.data

        # Show the error and provide recovery options
        st.error(api_result.error)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retry"):
                st.rerun()
        with col2:
            if st.button("Back to Home"):
                SessionState.navigate_to(PageName.HOME)
        return None

    @staticmethod
    def _render_metrics(result: dict) -> None:
        """Display mind map metadata as metrics cards.

        Shows three key stats: total nodes (complexity indicator),
        which backend was used, and which theme is active. These help
        users understand what they're looking at and how it was generated.
        """
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Nodes", result.get("total_nodes", 0))
        with col2:
            st.metric("Backend", result.get("backend_used", "unknown"))
        with col3:
            theme = SessionState.get_theme()
            st.metric("Theme", theme.replace("_", " ").title())

    @staticmethod
    def _render_exports(result: dict, tree: dict, dot_code: str) -> None:
        """Render download buttons for different export formats.

        Three export formats serve different downstream uses:
          - Mermaid (.mmd)  → Obsidian, documentation, web embeds
          - JSON (.json)    → Programmatic access, further processing
          - DOT (.dot)      → Graphviz tools, academic papers

        Each button uses Streamlit's download_button, which generates a
        download link without triggering a page navigation.
        """
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            mermaid = result.get("mermaid_code", "")
            if mermaid:
                st.download_button(
                    "Download Mermaid",
                    data=mermaid,
                    file_name="mindmap.mmd",
                    mime="text/plain",
                )

        with col_b:
            st.download_button(
                "Download JSON",
                data=json.dumps(tree, indent=2),
                file_name="mindmap.json",
                mime="application/json",
            )

        with col_c:
            st.download_button(
                "Download DOT",
                data=dot_code,
                file_name="mindmap.dot",
                mime="text/plain",
            )
