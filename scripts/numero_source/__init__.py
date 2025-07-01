"""
numero_source - A pipeline for processing book materials (cover, intro, table of contents)

This package provides a modular pipeline for extracting and processing:
- Cover images with thumbnail generation
- Introduction text from PDFs using OCR
- Table of Contents with structured parsing
"""

from .pipeline import BookProcessor

__version__ = "0.1.0"
__all__ = ["BookProcessor"]