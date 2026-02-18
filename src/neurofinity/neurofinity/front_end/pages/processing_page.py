"""
Processing Page — Intermediate Audio Processing.

This page is a brief transition state shown after voice recording stops.
Its job is to:
  1. Check if a transcript already exists (from audio upload) → redirect
  2. Process voice recording into text (placeholder) → redirect
  3. Show a loading indicator while processing happens

In production, this page would wire into the actual recording pipeline
(PyAudio buffer → wav file → AudioToText). Currently it acts as a
pass-through that redirects to the mind map page.

This page exists as a separate state (rather than inline logic in the
home page) because audio processing can be slow, and separating it into
its own page gives the user clear visual feedback that something is
happening. It also keeps the home page's render method clean.
"""

from __future__ import annotations

import time

import streamlit as st

from pages.base_page import BasePage
from state import SessionState, PageName


class ProcessingPage(BasePage):
    """Brief loading/processing page between input and mind map display."""

    @property
    def page_title(self) -> str:
        return "Processing Audio"

    def _render_content(self) -> None:
        """Check for existing transcript or process new recording.

        Two scenarios:
        1. Transcript already exists (from file upload path) — skip
           processing and go straight to mind map rendering.
        2. No transcript yet (from voice recording path) — process the
           recording into text, then navigate.
        """
        # Fast path: transcript already available from upload
        if SessionState.has_transcript():
            st.info("Transcript ready. Generating mind map...")
            SessionState.navigate_to(PageName.MIND_MAP)
            return

        # Slow path: process voice recording (placeholder implementation)
        with st.spinner("Processing audio and extracting transcript..."):
            transcript = self._process_recording()

        if transcript:
            SessionState.set_transcript(transcript)
            SessionState.navigate_to(PageName.MIND_MAP)
        else:
            st.error("Could not process recording. Please try again.")
            if st.button("Back to Home"):
                SessionState.navigate_to(PageName.HOME)

    @staticmethod
    def _process_recording() -> str | None:
        """Process a voice recording into transcript text.

        TODO: Wire this into the actual recording pipeline. The current
        implementation is a placeholder that simulates processing time.

        In production, this would:
        1. Read the audio buffer from the recording session
        2. Save it as a temporary WAV file
        3. Run AudioToText on the file
        4. Return the transcript text

        Returns:
            Transcript text, or None if processing failed.
        """
        # Simulated processing delay
        time.sleep(2)
        return "Placeholder transcript from voice recording."
