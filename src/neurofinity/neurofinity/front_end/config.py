"""
Application Configuration (Singleton Pattern).

This module centralizes all configuration constants and environment-driven
settings into a single, cached object. In Streamlit, every user interaction
triggers a full script rerun — so we use @st.cache_resource to ensure
the config is only constructed once and then reused on every rerun.

Design Pattern: Singleton
    Why: Config should be immutable and shared. Multiple instances would
    waste memory and risk inconsistency if env vars change mid-session.
    Streamlit's @st.cache_resource decorator gives us thread-safe singleton
    behavior without writing our own __new__ override.

Usage:
    from config import AppConfig
    cfg = AppConfig.get_instance()
    print(cfg.API_BASE_URL)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import streamlit as st

from loguru import logger


@dataclass(frozen=True)
class ThemeColors:
    """Color palette for a single accessibility theme.

    frozen=True makes instances immutable — themes should never change
    at runtime, and immutability prevents accidental mutation bugs.

    Each color maps to a node_type in the mind map tree. The 'bg', 'edge',
    and 'font' keys control the graph-level appearance.
    """
    bg: str
    root: str
    topic: str
    decision: str
    action_item: str
    question: str
    insight: str
    edge: str
    font: str

    def get_fill(self, node_type: str) -> str:
        """Look up the fill color for a given node_type, defaulting to 'topic'."""
        return getattr(self, node_type, self.topic)


# ---------------------------------------------------------------------------
# Pre-defined Accessibility Themes
# ---------------------------------------------------------------------------
# Each theme is designed for a specific neurodivergent accessibility need:
#
#   high_contrast — Maximum readability. Dark background with saturated
#                   foreground colors. Best for users with visual processing
#                   difficulties or low vision.
#
#   pastel        — Gentle, warm tones on a light background. Reduces
#                   visual stimulation. Good for users with light sensitivity
#                   or sensory overload concerns.
#
#   dark          — GitHub-style dark mode. Familiar to developers and
#                   comfortable for extended screen time. Reduces eye strain
#                   in low-light environments.
#
#   minimal       — Clean white background with muted colors. Maximum focus
#                   with minimal visual distraction. Good for users who
#                   prefer simplicity and clarity.

THEMES: dict[str, ThemeColors] = {
    "high_contrast": ThemeColors(
        bg="#1a1a2e", root="#e94560", topic="#0f3460",
        decision="#16a085", action_item="#e67e22",
        question="#8e44ad", insight="#2980b9",
        edge="#e94560", font="white",
    ),
    "pastel": ThemeColors(
        bg="#fef9ef", root="#f4a261", topic="#8ecae6",
        decision="#95d5b2", action_item="#ffb4a2",
        question="#cdb4db", insight="#a2d2ff",
        edge="#264653", font="#264653",
    ),
    "dark": ThemeColors(
        bg="#0d1117", root="#58a6ff", topic="#388bfd",
        decision="#3fb950", action_item="#d29922",
        question="#bc8cff", insight="#79c0ff",
        edge="#58a6ff", font="#c9d1d9",
    ),
    "minimal": ThemeColors(
        bg="#ffffff", root="#333333", topic="#666666",
        decision="#2d6a4f", action_item="#b56727",
        question="#7b2cbf", insight="#1d3557",
        edge="#999999", font="#333333",
    ),
}


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration.

    All settings that might vary between environments (local dev, staging,
    production) are read from environment variables with sensible defaults.
    The frozen=True ensures no code can accidentally modify config at runtime.
    """

    # --- Network ---
    API_BASE_URL: str =os.getenv("NEUROFINITY_API_URL", "http://localhost:8000")
    logger.info(f"API_BASE_URL: {API_BASE_URL}")
    API_TIMEOUT_SECONDS: int = 60  # Mind map generation can take 10-30s
    HEALTH_CHECK_TIMEOUT: int = 3

    # --- UI Defaults ---
    DEFAULT_BACKEND: str = "claude"
    DEFAULT_THEME: str = "high_contrast"
    DEFAULT_FONT_SIZE: int = 14
    DEFAULT_NODE_SPACING: float = 1.5

    # --- File Handling ---
    UPLOAD_DIR: str = "uploads"
    ALLOWED_AUDIO_TYPES: tuple = ("mp3", "wav")
    MAX_LABEL_LENGTH: int = 60  # Truncate node labels beyond this

    # --- App Metadata ---
    APP_TITLE: str = "Neuroinfy"
    APP_DESCRIPTION: str = "AI-powered mind maps for neurodivergent accessibility"

    @staticmethod
    @st.cache_resource
    def get_instance() -> "AppConfig":
        """Return the singleton AppConfig instance.

        @st.cache_resource ensures this is called exactly once across
        all Streamlit reruns. The returned object lives in Streamlit's
        resource cache until the app restarts.
        """
        return AppConfig()

    def get_theme(self, name: str) -> ThemeColors:
        """Look up a ThemeColors by name, falling back to high_contrast."""
        return THEMES.get(name, THEMES[self.DEFAULT_THEME])

    @property
    def theme_names(self) -> list[str]:
        """List all available theme names for the sidebar selector."""
        return list(THEMES.keys())
