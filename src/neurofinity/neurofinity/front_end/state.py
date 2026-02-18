"""
Session State Manager (Facade Pattern).

Streamlit reruns the entire script on every user interaction. Between reruns,
the only persistence mechanism is st.session_state — a dict-like object that
survives across reruns for a single user session. Raw access to session_state
is error-prone: typos in key names cause silent bugs, there's no type safety,
and state initialization is scattered across files.

This module wraps st.session_state behind a typed, centralized interface.
Every piece of state the app uses is accessed through SessionState methods,
so you get autocomplete, type hints, and a single place to see all state.

Design Pattern: Facade
    Why: st.session_state is a low-level dict API. The Facade provides a
    high-level, domain-specific interface (get_transcript(), set_page())
    that hides the underlying key-value mechanics. This makes the rest
    of the codebase cleaner and prevents the "magic string" problem
    where key names are duplicated across files.

Usage:
    from state import SessionState

    SessionState.initialize()           # Call once at app startup
    SessionState.set_page("mind_map")   # Navigate to mind map page
    transcript = SessionState.get_transcript()  # Read current transcript
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import streamlit as st


class PageName(str, Enum):
    """All possible page routes in the application.

    Using an Enum instead of raw strings prevents typo bugs like
    set_page("hom") silently routing nowhere. The IDE will autocomplete
    PageName.HOME and the type checker will catch invalid values.
    """
    HOME = "home"
    PROCESSING = "process_audio"
    MIND_MAP = "mind_map"


class SessionState:
    """Typed facade over st.session_state.

    All session state keys are defined as private class constants (_KEY_*)
    in one place. Public methods provide typed get/set access. This ensures:

    1. No duplicate key strings scattered across files
    2. Type safety — get_transcript() returns str, not Any
    3. Default values are defined once, not repeated at every access point
    4. Easy to audit: "what state does the app track?" → look at _KEY_* constants

    Note: This is a static class (all methods are @staticmethod) because
    Streamlit's session_state is a global singleton — there's no benefit
    to instantiating a SessionState object.
    """

    # --- Key Constants (private) ---
    # Centralizing keys here means a rename only touches one line.
    _KEY_PAGE = "page"
    _KEY_RECORDING = "recording"
    _KEY_TRANSCRIPT = "transcript"
    _KEY_BACKEND = "backend"
    _KEY_THEME = "theme"
    _KEY_MINDMAP_RESULT = "mindmap_result"

    @staticmethod
    def initialize() -> None:
        """Set up all session state keys with their default values.

        This must be called once at app startup (in app.py), before any
        page tries to read state. It uses setdefault() so that existing
        values (from a previous rerun) are preserved — only missing keys
        get initialized.
        """
        defaults = {
            SessionState._KEY_PAGE: PageName.HOME.value,
            SessionState._KEY_RECORDING: False,
            SessionState._KEY_TRANSCRIPT: "",
            SessionState._KEY_BACKEND: "claude",
            SessionState._KEY_THEME: "high_contrast",
            SessionState._KEY_MINDMAP_RESULT: None,
        }
        for key, default in defaults.items():
            st.session_state.setdefault(key, default)

    # --- Page Navigation ---

    @staticmethod
    def get_page() -> str:
        """Return the current page route name."""
        return st.session_state.get(SessionState._KEY_PAGE, PageName.HOME.value)

    @staticmethod
    def set_page(page: PageName) -> None:
        """Navigate to a new page. Accepts a PageName enum for type safety."""
        st.session_state[SessionState._KEY_PAGE] = page.value

    @staticmethod
    def navigate_to(page: PageName) -> None:
        """Navigate to a page AND trigger a rerun.

        This is the standard Streamlit navigation pattern: update state,
        then call st.rerun() so the script re-executes and renders the
        new page. Combining both steps in one method prevents the common
        bug of forgetting the rerun after a state change.
        """
        SessionState.set_page(page)
        st.rerun()

    # --- Recording State ---

    @staticmethod
    def is_recording() -> bool:
        return st.session_state.get(SessionState._KEY_RECORDING, False)

    @staticmethod
    def set_recording(value: bool) -> None:
        st.session_state[SessionState._KEY_RECORDING] = value

    # --- Transcript ---

    @staticmethod
    def get_transcript() -> str:
        return st.session_state.get(SessionState._KEY_TRANSCRIPT, "")

    @staticmethod
    def set_transcript(text: str) -> None:
        st.session_state[SessionState._KEY_TRANSCRIPT] = text

    @staticmethod
    def has_transcript() -> bool:
        return bool(SessionState.get_transcript().strip())

    # --- Backend & Theme Preferences ---

    @staticmethod
    def get_backend() -> str:
        return st.session_state.get(SessionState._KEY_BACKEND, "claude")

    @staticmethod
    def set_backend(backend: str) -> None:
        st.session_state[SessionState._KEY_BACKEND] = backend

    @staticmethod
    def get_theme() -> str:
        return st.session_state.get(SessionState._KEY_THEME, "high_contrast")

    @staticmethod
    def set_theme(theme: str) -> None:
        st.session_state[SessionState._KEY_THEME] = theme

    # --- Mind Map Result Cache ---
    # Stores the API response so we don't re-call the API on every rerun
    # of the mind_map page (e.g., when the user clicks a download button).

    @staticmethod
    def get_mindmap_result() -> dict | None:
        return st.session_state.get(SessionState._KEY_MINDMAP_RESULT)

    @staticmethod
    def set_mindmap_result(result: dict | None) -> None:
        st.session_state[SessionState._KEY_MINDMAP_RESULT] = result

    # --- Bulk Reset ---

    @staticmethod
    def reset_for_new_session() -> None:
        """Clear all data and return to home. Used by 'Generate Another' button."""
        SessionState.set_transcript("")
        SessionState.set_mindmap_result(None)
        SessionState.set_recording(False)
        SessionState.navigate_to(PageName.HOME)
