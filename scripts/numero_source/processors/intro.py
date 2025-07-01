"""
Intro Parser for processing introduction content from PDFs.

This module handles the OCR processing of intro.pdf files and combines
the markdown content from all pages into a single text.
"""

from typing import Optional
from pathlib import Path
from ..ocr.mistral_ocr import MistralOCR


class IntroParser:
    """
    Parser for extracting and processing introduction content from PDF files.
    
    Uses MistralOCR to process the intro.pdf and combines all pages into
    a single markdown text without page separators.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the intro parser."""
        self.ocr = MistralOCR(api_key=api_key)
    
    def parse_intro(self, intro_pdf_path: str, intermediates_dir: Optional[Path] = None) -> str:
        """
        Parse introduction content from a PDF file.
        
        Args:
            intro_pdf_path: Path to the intro.pdf file
            intermediates_dir: Directory to save intermediate responses (if debug mode)
            
        Returns:
            Combined markdown content from all pages without page separators
            
        Raises:
            FileNotFoundError: If the intro PDF file doesn't exist
            Exception: For OCR processing errors
        """
        try:
            # Process the PDF with OCR
            ocr_response = self.ocr.process_file(
                intro_pdf_path,
                intermediates_dir=intermediates_dir,
                response_prefix="intro_ocr"
            )
            
            # Extract markdown without page separators
            markdown_content = self.ocr.extract_markdown(
                ocr_response, 
                include_page_separators=False
            )
            
            return markdown_content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to process intro PDF: {e}")
    
    def parse_intro_safe(self, intro_pdf_path: str, intermediates_dir: Optional[Path] = None) -> str:
        """
        Safely parse introduction content, returning empty string on failure.
        
        Args:
            intro_pdf_path: Path to the intro.pdf file
            intermediates_dir: Directory to save intermediate responses (if debug mode)
            
        Returns:
            Combined markdown content or empty string if processing fails
        """
        try:
            return self.parse_intro(intro_pdf_path, intermediates_dir)
        except Exception as e:
            print(f"Warning: Failed to process intro PDF {intro_pdf_path}: {e}")
            return ""