"""
Home Page — Landing and Input Collection.

This page provides three ways for users to input meeting content:
  1. Upload an audio file (MP3/WAV) → transcribed via AudioToText
  2. Paste a transcript directly → useful for testing and pre-transcribed content
  3. Record voice live → placeholder, to be wired to actual recording pipeline

Each input method is implemented as a separate private method, making it
easy to add, remove, or reorder input options. When any input method
successfully produces a transcript, it stores the text in SessionState
and navigates to the MindMap page.

This page also renders the hero image and app title at the top.
"""

from __future__ import annotations

import os

import streamlit as st

from pages.base_page import BasePage
from state import SessionState, PageName
from services.api_client import MindMapAPIClient


class HomePage(BasePage):
    """The application's entry point and primary input collection page.

    This page follows the "progressive disclosure" UX pattern: the most
    common action (uploading a file) is shown first, followed by alternatives
    (paste text, record voice). Users see all options at a glance but are
    naturally guided toward the primary flow.
    """

    @property
    def page_title(self) -> str:
        return ""  # Title is rendered as HTML in _render_header

    def _render_header(self) -> None:
        """Custom header with hero image and centered title.

        Overrides the base class header to show the app's branding
        instead of a plain text heading.
        """
        # Show background image if it exists (from the original project)
        if os.path.exists("background.jpg"):
            st.image("background.jpg", use_container_width=True)

        st.markdown(
            "<h1 style='text-align: center;'>Neuroinfy</h1>",
            unsafe_allow_html=True,
        )

    def _render_content(self) -> None:
        """Render all three input methods in order.

        Each method is self-contained — it handles its own UI, validation,
        and state transitions. This makes it easy to add a fourth input
        method (e.g., "Import from Google Meet") by just adding a new
        private method and calling it here.
        """
        self._render_audio_upload()
        self._render_text_paste()
        self._render_voice_recorder()

    # --- Input Method 1: Audio File Upload ---

    def _render_audio_upload(self) -> None:
        """File uploader for MP3/WAV audio files.

        Flow: User uploads file → saved to disk → AudioToText transcribes
        → transcript stored in SessionState → navigate to mind map page.

        The audio player widget lets users verify they uploaded the right
        file before the (potentially slow) transcription begins.
        """
        st.subheader("Upload an Audio File")
        uploaded_file = st.file_uploader(
            "Choose an MP3 or WAV file",
            type=["mp3", "wav"],
        )

        if uploaded_file is not None:
            # Save to disk because AudioToText expects a file path
            file_path = os.path.join("uploads", uploaded_file.name)
            os.makedirs("uploads", exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Let the user hear the audio before committing to transcription
            audio_format = (
                "audio/mp3" if uploaded_file.type == "audio/mpeg" else "audio/wav"
            )
            st.audio(file_path, format=audio_format)

            # Transcribe using the existing AudioToText module
            with st.spinner("Transcribing audio..."):
                transcript = self._transcribe_audio(file_path)

            if transcript:
                SessionState.set_transcript(transcript)
                st.success("File uploaded and transcribed!")
                SessionState.navigate_to(PageName.MIND_MAP)

    @staticmethod
    def _transcribe_audio(file_path: str) -> str | None:
        """Run audio-to-text transcription.

        Wrapped in a static method to isolate the import and error handling.
        The AudioToText class may return different response types depending
        on the HuggingFace API version, so we handle multiple formats.

        Returns:
            The transcript text, or None if transcription failed.
        """
        try:
            from audio_to_text import AudioToText

            transcriber = AudioToText(audio_path=file_path)
            response = transcriber.output_text()

            # The HuggingFace API can return either a Response object
            # or a plain string, depending on the wrapper implementation
            if hasattr(response, "json"):
                data = response.json()
                return data.get("text", str(data))
            return str(response)
        except ImportError:
            st.error(
                "AudioToText module not found. Make sure `audio_to_text.py` "
                "is in the same directory."
            )
            return None
        except Exception as e:
            st.error(f"Transcription failed: {e}")
            return None

    # --- Input Method 2: Direct Text Paste ---

    def _render_text_paste(self) -> None:
        """Text area for pasting pre-existing transcripts.

        This is the fastest way to test the pipeline — paste any text and
        immediately generate a mind map. It's also useful for users who
        get transcripts from external tools (Otter.ai, Google Meet, etc.).
        """
        st.subheader("Or Paste a Transcript")
        pasted_text = st.text_area(
            "Paste meeting transcript here",
            height=200,
            placeholder=(
                "John: Let's discuss the Q3 roadmap.\n"
                "Sarah: I think we should prioritize mobile.\n"
                "Mike: The API also needs optimization..."
            ),
        )

        if st.button(
            "Generate Mind Map",
            type="primary",
            disabled=not pasted_text.strip(),
        ):
            SessionState.set_transcript(pasted_text.strip())
            SessionState.navigate_to(PageName.MIND_MAP)

    # --- Input Method 3: Live Voice Recording ---

    def _render_voice_recorder(self) -> None:
        """Start/Stop buttons for live voice recording.

        NOTE: This is currently a placeholder. The actual recording pipeline
        (using PyAudio or streamlit-webrtc) needs to be wired in. The UI
        skeleton is here so the interaction flow is complete.

        The circular button styling matches the original neuroui.py design.
        """
        st.subheader("Record Your Voice")

        # Custom CSS for circular recording buttons
        st.markdown(
            """
            <style>
            .stButton button {
                border-radius: 50%;
                width: 150px;
                height: 150px;
                font-size: 24px;
                margin: 10px;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🎤 Start", key="start_recording"):
                SessionState.set_recording(True)
                st.write("Recording started... Speak now!")

        with col2:
            if st.button("⏹️ Stop", key="stop_recording"):
                if SessionState.is_recording():
                    SessionState.set_recording(False)
                    st.success("Recording stopped and saved!")
                    SessionState.navigate_to(PageName.PROCESSING)
                else:
                    st.warning("Recording is not in progress.")
