"""
Services Layer.

This package contains service classes that handle external communication
(API calls, file I/O, transcription). Services are stateless — they don't
hold UI state or render anything. They just take input, talk to an external
system, and return structured results.

Keeping services separate from UI code means:
  - API logic can be tested without Streamlit
  - Swapping backends (e.g., REST → gRPC) only changes this layer
  - UI components stay focused on rendering, not networking
"""

from .api_client import MindMapAPIClient

__all__ = ["MindMapAPIClient"]
