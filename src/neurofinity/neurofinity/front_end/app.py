"""
Neurofinity Streamlit Application — Entry Point.

This is the file Streamlit runs directly:
    streamlit run app.py

It wires together all the architectural layers:

    config.py          → Singleton configuration (AppConfig)
    state.py           → Session state facade (SessionState)
    services/          → API communication (MindMapAPIClient)
    components/        → Reusable UI widgets (Sidebar, MindMapRenderer)
    pages/             → Page classes (Home, Processing, MindMap)

Architecture Overview:
    ┌─────────────────────────────────────────────────┐
    │  app.py (this file)                             │
    │  ┌─────────────┐   ┌────────────────────────┐   │
    │  │ PageRouter   │──▶│ HomePage               │   │
    │  │ (Registry)   │──▶│ ProcessingPage         │   │
    │  │              │──▶│ MindMapPage             │   │
    │  └──────┬───────┘   └───────────┬────────────┘   │
    │         │                       │                │
    │  ┌──────▼───────┐   ┌──────────▼────────────┐   │
    │  │ SessionState  │   │ SidebarComponent      │   │
    │  │ (Facade)      │   │ (Composite)           │   │
    │  └───────────────┘   └──────────┬────────────┘   │
    │                                 │                │
    │                      ┌──────────▼────────────┐   │
    │                      │ MindMapAPIClient       │   │
    │                      │ (Service Layer)        │   │
    │                      └───────────────────────┘   │
    └─────────────────────────────────────────────────┘

Design Pattern: Registry (Page Router)
    Why: The app needs to map page names (from SessionState) to page
    class instances. A dict-based registry is simpler and more extensible
    than a chain of if/elif statements. Adding a new page means adding
    one line to the registry dict — no conditional logic changes needed.

    The registry also enables dependency injection: each page receives
    the API client through its constructor, making pages testable with
    mock clients.

Streamlit Execution Model:
    Streamlit re-executes this entire script on EVERY user interaction
    (button click, slider move, text input). This means:
    - All imports run every time (fast, because Python caches modules)
    - SessionState.initialize() uses setdefault (safe across reruns)
    - AppConfig and MindMapAPIClient use @st.cache_resource (created once)
    - The page registry dict is rebuilt (cheap — just dict creation)
    - Only the current page's render() method produces visible output

Usage:
    # Terminal 1: Start the FastAPI backend
    uvicorn neurofinity.api:app --reload --port 8000

    # Terminal 2: Start this Streamlit frontend
    streamlit run app.py
"""

import streamlit as st

from config import AppConfig
from state import SessionState, PageName
from services.api_client import MindMapAPIClient
from components.sidebar import SidebarComponent
from pages.home_page import HomePage
from pages.processing_page import ProcessingPage
from pages.mindmap_page import MindMapPage


# ---------------------------------------------------------------------------
# Page Configuration (must be the FIRST Streamlit call)
# ---------------------------------------------------------------------------
# st.set_page_config() must be called before any other st.* function.
# This is a Streamlit requirement — violating it causes a runtime error.

st.set_page_config(
    page_title="Neuroinfy — Mind Map Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Application entry point.

    Execution order:
    1. Initialize session state (safe across reruns via setdefault)
    2. Create singleton services (cached — only constructed once)
    3. Build the page registry (cheap dict creation)
    4. Render the sidebar (always visible, regardless of current page)
    5. Dispatch to the current page's render() method
    """

    # Step 1: Ensure all session state keys exist with defaults.
    # This is idempotent — setdefault() preserves existing values.
    SessionState.initialize()

    # Step 2: Get singleton instances of services.
    # @st.cache_resource ensures these are created once and reused.
    api_client = MindMapAPIClient.get_instance()

    # Step 3: Build the page registry.
    # Maps PageName enum values to instantiated page objects.
    # Each page receives the api_client via constructor injection.
    #
    # To add a new page:
    #   1. Create the class in pages/
    #   2. Add the route to PageName enum in state.py
    #   3. Add one line here in the registry
    page_registry: dict[str, object] = {
        PageName.HOME.value: HomePage(api_client),
        PageName.PROCESSING.value: ProcessingPage(api_client),
        PageName.MIND_MAP.value: MindMapPage(api_client),
    }

    # Step 4: Render the sidebar (always visible).
    sidebar = SidebarComponent(api_client)
    sidebar.render()

    # Step 5: Route to the current page.
    # SessionState.get_page() returns the current route string.
    # The registry lookup finds the right page instance.
    # Fallback to HomePage if the route is somehow invalid.
    current_page_name = SessionState.get_page()
    current_page = page_registry.get(
        current_page_name,
        page_registry[PageName.HOME.value],
    )

    # Step 6: Render the page.
    # This calls BasePage.render(), which invokes the Template Method:
    #   _render_header() → _render_content() → _render_footer()
    current_page.render()


# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------
# Streamlit runs this file as a script (not imported as a module),
# so __name__ is always "__main__". The guard is here for clarity
# and to prevent issues if someone accidentally imports this file.

if __name__ == "__main__":
    main()
