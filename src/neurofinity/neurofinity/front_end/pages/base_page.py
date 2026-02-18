"""
Base Page (Template Method Pattern).

This abstract base class defines the skeleton of a page's rendering lifecycle.
Every page in the app follows the same three-step structure:

    1. _render_header()  — Page title, breadcrumbs, status indicators
    2. _render_content() — The main page body (MUST be overridden)
    3. _render_footer()  — Navigation buttons, help text

Subclasses override one or more of these "hook" methods to customize their
appearance, while the overall rendering order stays fixed. This is the
Template Method pattern — the base class defines the algorithm structure,
and subclasses fill in the steps.

Design Pattern: Template Method
    Why: All pages share common behavior (they all need consistent header
    styling, they all should handle missing state gracefully, they all
    need navigation controls). Without a base class, each page would
    duplicate this boilerplate. The Template Method lets us define this
    shared structure once and let each page focus only on what's unique.

    The alternative — using plain functions — would require copying the
    header/footer logic into every page function. That violates DRY and
    makes it easy for pages to drift out of sync visually.

Usage:
    class MyPage(BasePage):
        @property
        def page_title(self) -> str:
            return "My Page"

        def _render_content(self) -> None:
            st.write("Hello from my page!")

    page = MyPage(api_client)
    page.render()  # Calls header → content → footer in order
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import streamlit as st

from services.api_client import MindMapAPIClient


class BasePage(ABC):
    """Abstract base class for all application pages.

    Each page receives the API client via dependency injection (constructor),
    rather than creating its own. This makes pages testable — in tests, you
    can pass a mock client that returns canned responses.

    The render() method is final (not meant to be overridden). It defines
    the fixed rendering order. Subclasses customize behavior by overriding
    the protected hook methods (_render_header, _render_content, _render_footer).
    """

    def __init__(self, api_client: MindMapAPIClient) -> None:
        """Initialize the page with its dependencies.

        Args:
            api_client: The singleton API client for backend communication.
        """
        self._client = api_client

    # --- Template Method (the fixed algorithm) ---

    def render(self) -> None:
        """Render the complete page. DO NOT OVERRIDE.

        This method defines the rendering lifecycle. It calls the three
        hook methods in order. Subclasses customize by overriding the hooks,
        not this method.

        The try/except ensures that even if a page crashes during rendering,
        the user sees a helpful error message instead of a raw traceback.
        """
        try:
            self._render_header()
            self._render_content()
            self._render_footer()
        except Exception as e:
            st.error(f"Page rendering error: {e}")
            st.info("Try refreshing the page or going back to Home.")

    # --- Hook Methods (override these in subclasses) ---

    def _render_header(self) -> None:
        """Render the page header. Override for custom headers.

        Default implementation shows the page_title as a Markdown heading.
        Override this to add breadcrumbs, status indicators, or custom
        styling to the top of specific pages.
        """
        st.markdown(f"## {self.page_title}")

    @abstractmethod
    def _render_content(self) -> None:
        """Render the main page content. MUST BE OVERRIDDEN.

        This is where the page's unique UI lives — forms, visualizations,
        file uploaders, etc. This is the only method that every subclass
        MUST implement.
        """
        ...

    def _render_footer(self) -> None:
        """Render the page footer. Override for custom navigation.

        Default implementation is empty. Override to add "Back" buttons,
        help text, or navigation links at the bottom of specific pages.
        """
        pass

    # --- Abstract Properties ---

    @property
    @abstractmethod
    def page_title(self) -> str:
        """Human-readable title for this page. Shown in the header."""
        ...
