"""
Page Classes.

Each page in the application is a class with a render() method. Pages
inherit from BasePage, which provides the Template Method pattern for
consistent page lifecycle (setup → render → cleanup).

The page routing system in app.py maps PageName enum values to page
class instances, creating a clean navigation architecture.
"""

from .base_page import BasePage
from .home_page import HomePage
from .processing_page import ProcessingPage
from .mindmap_page import MindMapPage

__all__ = ["BasePage", "HomePage", "ProcessingPage", "MindMapPage"]
