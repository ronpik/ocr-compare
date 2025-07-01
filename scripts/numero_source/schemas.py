"""
Output schemas for the numero_source pipeline.

Defines the structure of the final JSON output.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from .toc.models import TableOfContents


class CoverInfo(BaseModel):
    """Information about the book cover."""
    original: Optional[str] = Field(None, description="Relative path to the original cover image (cover-1.png)")
    thumbnail: Optional[str] = Field(None, description="Path to the generated thumbnail")
    thumbnail_base64: Optional[str] = Field(None, description="Base64 encoded thumbnail for web display")


class BookProcessingResult(BaseModel):
    """Result of processing a complete book folder."""
    
    cover: CoverInfo = Field(..., description="Cover image information")
    intro: str = Field("", description="Combined introduction text from OCR")
    toc: TableOfContents = Field(..., description="Parsed table of contents structure")
    folder_name: str = Field(..., description="Name of the processed folder")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            # Ensure clean JSON serialization
        }