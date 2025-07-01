"""
MistralOCR - A class for OCR processing using Mistral's API.

This module provides a clean interface for processing documents using 
Mistral's OCR capabilities, extracted from the original CLI script.
"""

import os
import json
import uuid
from typing import Dict, Any, List, Optional
from pathlib import Path

from mistralai import Mistral
from mistralai import models


class MistralOCR:
    """
    A class for performing OCR on documents using Mistral's API.
    
    Attributes:
        api_key: The Mistral API key
        client: Mistral client instance
        default_model: Default OCR model to use
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-ocr-latest"):
        """
        Initialize the MistralOCR instance.
        
        Args:
            api_key: Mistral API key. If not provided, will try to get from environment
            model: The OCR model to use (default: mistral-ocr-latest)
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required. Provide via parameter or environment variable.")
        
        self.client = Mistral(api_key=self.api_key)
        self.default_model = model
    
    def process_file(
        self,
        file_path: str,
        pages: Optional[List[int]] = None,
        include_image_base64: bool = False,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
        bbox_annotation_format: Optional[Dict[str, Any]] = None,
        document_annotation_format: Optional[Dict[str, Any]] = None,
        intermediates_dir: Optional[Path] = None,
        response_prefix: str = "ocr_response",
    ) -> Dict[str, Any]:
        """
        Process a local file with OCR.
        
        Args:
            file_path: Path to the document file
            pages: Specific pages to process (0-indexed). If None, processes all pages
            include_image_base64: Whether to include base64 encoded images in response
            image_limit: Maximum number of images to extract
            image_min_size: Minimum height and width of image to extract
            bbox_annotation_format: Structured output format for bounding box annotations
            document_annotation_format: Structured output format for document-level annotations
            
        Returns:
            Dictionary containing the OCR response
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: For other API errors
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Upload file
        with open(file_path, "rb") as f:
            content = f.read()
        
        file = models.File(
            file_name=Path(file_path).name,
            content=content
        )
        uploaded_file = self.client.files.upload(file=file, purpose="ocr")
        
        # Get signed URL
        signed_url_response = self.client.files.get_signed_url(file_id=uploaded_file.id)
        document_url = signed_url_response.url
        
        return self._process_document(
            document_url=document_url,
            pages=pages,
            include_image_base64=include_image_base64,
            image_limit=image_limit,
            image_min_size=image_min_size,
            bbox_annotation_format=bbox_annotation_format,
            document_annotation_format=document_annotation_format,
            intermediates_dir=intermediates_dir,
            response_prefix=response_prefix
        )
    
    def process_url(
        self,
        url: str,
        pages: Optional[List[int]] = None,
        include_image_base64: bool = False,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
        bbox_annotation_format: Optional[Dict[str, Any]] = None,
        document_annotation_format: Optional[Dict[str, Any]] = None,
        intermediates_dir: Optional[Path] = None,
        response_prefix: str = "ocr_response",
    ) -> Dict[str, Any]:
        """
        Process a document from URL with OCR.
        
        Args:
            url: URL of the document to process
            pages: Specific pages to process (0-indexed). If None, processes all pages
            include_image_base64: Whether to include base64 encoded images in response
            image_limit: Maximum number of images to extract
            image_min_size: Minimum height and width of image to extract
            bbox_annotation_format: Structured output format for bounding box annotations
            document_annotation_format: Structured output format for document-level annotations
            
        Returns:
            Dictionary containing the OCR response
        """
        return self._process_document(
            document_url=url,
            pages=pages,
            include_image_base64=include_image_base64,
            image_limit=image_limit,
            image_min_size=image_min_size,
            bbox_annotation_format=bbox_annotation_format,
            document_annotation_format=document_annotation_format,
            intermediates_dir=intermediates_dir,
            response_prefix=response_prefix
        )
    
    def _process_document(
        self,
        document_url: str,
        pages: Optional[List[int]] = None,
        include_image_base64: bool = False,
        image_limit: Optional[int] = None,
        image_min_size: Optional[int] = None,
        bbox_annotation_format: Optional[Dict[str, Any]] = None,
        document_annotation_format: Optional[Dict[str, Any]] = None,
        intermediates_dir: Optional[Path] = None,
        response_prefix: str = "ocr_response",
    ) -> Dict[str, Any]:
        """
        Internal method to process document with OCR.
        
        Returns:
            Dictionary containing the OCR response
        """
        document = {
            "type": "document_url",
            "document_url": document_url,
        }
        
        ocr_response = self.client.ocr.process(
            model=self.default_model,
            document=document,
            pages=pages,
            include_image_base64=include_image_base64,
            image_limit=image_limit,
            image_min_size=image_min_size,
            bbox_annotation_format=bbox_annotation_format,
            document_annotation_format=document_annotation_format,
        )
        
        response_dict = ocr_response.model_dump()
        
        # Save intermediate response if debug mode is enabled
        if intermediates_dir:
            self._save_intermediate_response(
                response_dict, 
                intermediates_dir, 
                f"{response_prefix}_{uuid.uuid4().hex[:8]}.json"
            )
        
        return response_dict
    
    def _save_intermediate_response(self, response: Dict[str, Any], intermediates_dir: Path, filename: str):
        """Save intermediate response to the intermediates directory."""
        try:
            intermediates_dir.mkdir(parents=True, exist_ok=True)
            response_file = intermediates_dir / filename
            
            with open(response_file, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2, ensure_ascii=False, default=str)
                
            print(f"Saved intermediate response: {response_file}")
        except Exception as e:
            print(f"Warning: Failed to save intermediate response: {e}")
    
    def extract_markdown(self, ocr_response: Dict[str, Any], include_page_separators: bool = True) -> str:
        """
        Extract markdown content from OCR response.
        
        Args:
            ocr_response: The OCR response dictionary
            include_page_separators: Whether to include page separators
            
        Returns:
            Combined markdown content from all pages
        """
        if "pages" not in ocr_response:
            return ""
        
        markdown_parts = []
        for page in ocr_response["pages"]:
            if "markdown" in page:
                page_idx = page.get("index", "?")
                
                if include_page_separators and markdown_parts:
                    markdown_parts.append("\n---\n\n")
                    
                if include_page_separators:
                    markdown_parts.append(f"<!-- Page {page_idx + 1} -->\n")
                    
                markdown_parts.append(page["markdown"])
        
        return "\n".join(markdown_parts)
    
    def extract_pages_markdown(self, ocr_response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract markdown content per page from OCR response.
        
        Args:
            ocr_response: The OCR response dictionary
            
        Returns:
            List of dictionaries with page index and markdown content
        """
        if "pages" not in ocr_response:
            return []
        
        pages_markdown = []
        for page in ocr_response["pages"]:
            if "markdown" in page:
                pages_markdown.append({
                    "index": page.get("index", 0),
                    "markdown": page["markdown"]
                })
        
        return pages_markdown