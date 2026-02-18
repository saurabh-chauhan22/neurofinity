"""
Mind Map API Client (Service Layer Pattern).

This module handles ALL communication with the FastAPI backend. No other
module in the frontend should import `requests` or know about HTTP details.
This separation has several benefits:

1. **Testability** — You can unit-test this class by mocking `requests`,
   without needing Streamlit or a running API server.

2. **Swappability** — If you later switch from REST to gRPC, WebSocket,
   or even direct Python imports (calling MindMapGenerator directly
   instead of through HTTP), only this file changes.

3. **Error Isolation** — Network errors, timeouts, and JSON parsing
   failures are caught here and converted to clean Result objects.
   UI code never has to handle raw HTTP exceptions.

Design Pattern: Service Layer
    Why: The frontend shouldn't know or care HOW the mind map is generated.
    It just says "give me a mind map for this transcript" and gets back
    structured data. The Service Layer provides this abstraction boundary.

    Combined with Streamlit caching:
    - @st.cache_resource on get_instance() → Singleton (one client per app)
    - @st.cache_data on health/backends → avoid redundant network calls
      during Streamlit's frequent reruns

Usage:
    from services import MindMapAPIClient

    client = MindMapAPIClient.get_instance()
    if client.is_healthy():
        result = client.generate_mindmap("John: Let's discuss...")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import requests
import streamlit as st
from loguru import logger

from neurofinity.front_end.config import AppConfig


@dataclass
class APIResult:
    """Structured result from an API call.

    Instead of returning raw dicts or None (which forces callers to do
    null checks everywhere), every API method returns an APIResult with
    explicit success/error fields. This makes error handling in UI code
    clean and predictable:

        result = client.generate_mindmap(transcript)
        if result.success:
            render(result.data)
        else:
            st.error(result.error)
    """
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None


class MindMapAPIClient:
    """HTTP client for the Neurofinity FastAPI backend.

    This class encapsulates all API endpoints behind clean Python methods.
    Each method handles its own error cases and returns an APIResult,
    so the calling code never needs to think about HTTP status codes,
    connection errors, or JSON parsing.

    The class is designed to be used as a singleton via get_instance()
    because there's no per-request state — it's just a thin wrapper
    around the base URL and timeout settings.
    """

    def __init__(self, config: AppConfig) -> None:
        """Initialize with app configuration.

        Args:
            config: The AppConfig singleton containing API_BASE_URL,
                    timeout settings, etc.
        """
        self._base_url = config.API_BASE_URL.rstrip("/")
        self._timeout = config.API_TIMEOUT_SECONDS
        self._health_timeout = config.HEALTH_CHECK_TIMEOUT

    @staticmethod
    @st.cache_resource
    def get_instance() -> "MindMapAPIClient":
        """Return the singleton MindMapAPIClient.

        Uses @st.cache_resource so the client is created once and reused
        across all Streamlit reruns. This avoids re-reading env vars and
        re-constructing the object on every user interaction.
        """
        config = AppConfig.get_instance()
        return MindMapAPIClient(config)

    # --- Health & Discovery Endpoints ---

    def is_healthy(self) -> bool:
        """Check if the FastAPI backend is running and reachable.

        Uses a short timeout (3s default) because this is called on every
        page load to show the status indicator. We don't want the UI to
        hang for 60s if the backend is down.

        Returns:
            True if the /health endpoint responds with 200, False otherwise.
        """
        try:
            logger.info(f"Checking health of API at {self._base_url}")
            resp = requests.get(
                f"{self._base_url}/health",
                timeout=self._health_timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_available_backends(self) -> dict[str, dict]:
        """Query which AI backends are available (have API keys configured).

        Returns:
            A dict mapping backend names to their info:
            {"claude": {"available": True, "description": "..."}, ...}
            Returns empty dict on failure.
        """
        try:
            resp = requests.get(
                f"{self._base_url}/mindmap/backends",
                timeout=self._health_timeout,
            )
            if resp.status_code == 200:
                return resp.json().get("backends", {})
        except Exception as e:
            logger.debug(f"Failed to fetch backends: {e}")
        return {}

    # --- Core Generation Endpoint ---

    def generate_mindmap(
        self,
        transcript: str,
        title: str = "Meeting Mind Map",
        backend: str = "claude",
        theme: str = "high_contrast",
    ) -> APIResult:
        """Generate a mind map from a meeting transcript.

        This is the primary method — it sends the transcript to the FastAPI
        /mindmap/generate endpoint and returns the structured result.

        The method handles three classes of errors:
        1. ConnectionError — API server not running
        2. HTTPError — API returned 4xx/5xx (bad input, server crash)
        3. Unexpected exceptions — network issues, JSON parse failures

        Args:
            transcript: Raw meeting transcript text.
            title: Title for the mind map root node.
            backend: AI backend — "claude", "openai", or "local".
            theme: Accessibility theme name.

        Returns:
            APIResult with success=True and data containing the full API
            response (tree, mermaid_code, total_nodes, etc.), or
            success=False with a human-readable error message.
        """
        try:
            response = requests.post(
                f"{self._base_url}/mindmap/generate",
                json={
                    "text": transcript,
                    "title": title,
                    "backend": backend,
                    "theme": theme,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            return APIResult(success=True, data=response.json())

        except requests.exceptions.ConnectionError:
            return APIResult(
                success=False,
                error=(
                    "Could not connect to the Neurofinity API. "
                    "Make sure the FastAPI server is running:\n\n"
                    "`uvicorn neurofinity.api:app --reload --port 8000`"
                ),
            )
        except requests.exceptions.HTTPError as e:
            # Extract the error detail from the FastAPI response body
            try:
                detail = e.response.json().get("detail", e.response.text)
            except Exception:
                detail = e.response.text
            return APIResult(success=False, error=f"API error: {detail}")

        except requests.exceptions.Timeout:
            return APIResult(
                success=False,
                error=(
                    f"API request timed out after {self._timeout}s. "
                    "The transcript may be too long, or the backend is slow."
                ),
            )
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            return APIResult(success=False, error=f"Unexpected error: {e}")
