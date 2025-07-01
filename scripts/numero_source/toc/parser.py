"""
Table of Contents Parser with simplified page-by-page workflow.

This module implements a streamlined approach to parsing ToC from multiple pages,
building the ToC incrementally using context from previous pages.
"""

import json
import base64
import mimetypes
from typing import List, Dict, Any, Optional
from pathlib import Path

from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model

from .models import TocEntry, TableOfContents
from ..ocr.mistral_ocr import MistralOCR


class TableOfContentsParser:
    """
    Parser for extracting Table of Contents from multiple page images.
    
    Uses a page-by-page approach where each page is processed sequentially,
    using context from previous pages to build the complete ToC structure.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the ToC parser."""
        self.api_key = api_key
        self.client = Mistral(api_key=api_key) if api_key else None
        self.ocr = MistralOCR(api_key=api_key)
    
    def parse_from_pdf(self, pdf_path: str, toc_images_folder: str) -> TableOfContents:
        """
        Parse ToC from a PDF file and corresponding page images.
        
        Args:
            pdf_path: Path to the ToC PDF file
            toc_images_folder: Path to folder containing toc-1.jpg, toc-2.jpg, etc.
            
        Returns:
            TableOfContents object with parsed entries
        """
        if not self.client:
            raise ValueError("API key required for ToC parsing")
        
        # First, extract markdown from the entire PDF using OCR
        ocr_response = self.ocr.process_file(pdf_path)
        pages_markdown = self.ocr.extract_pages_markdown(ocr_response)
        
        # Find all ToC page images
        toc_folder = Path(toc_images_folder)
        toc_images = []
        for i in range(1, 20):  # Support up to 20 pages
            image_path = toc_folder / f"toc-{i}.jpg"
            if image_path.exists():
                toc_images.append(str(image_path))
            else:
                break
        
        if not toc_images:
            raise FileNotFoundError(f"No toc-*.jpg images found in {toc_images_folder}")
        
        # Process pages sequentially
        all_entries = []
        previous_context = None
        
        for i, image_path in enumerate(toc_images):
            current_markdown = ""
            if i < len(pages_markdown):
                current_markdown = pages_markdown[i]["markdown"]
            
            page_entries = self._process_single_page(
                image_path=image_path,
                page_number=i + 1,
                current_page_markdown=current_markdown,
                previous_page_markdown=previous_context,
                previous_entries=all_entries
            )
            
            all_entries.extend(page_entries)
            previous_context = current_markdown
        
        # Assign sequential IDs to all entries
        for idx, entry in enumerate(all_entries, 1):
            entry.id = idx
        
        return TableOfContents(entries=all_entries)
    
    def _process_single_page(
        self,
        image_path: str,
        page_number: int,
        current_page_markdown: str,
        previous_page_markdown: Optional[str],
        previous_entries: List[TocEntry]
    ) -> List[TocEntry]:
        """
        Process a single ToC page to extract entries.
        
        Args:
            image_path: Path to the current page image
            page_number: Current page number (1-indexed)
            current_page_markdown: OCR markdown for current page
            previous_page_markdown: OCR markdown from previous page
            previous_entries: TocEntry objects from previous pages
            
        Returns:
            List of TocEntry objects for the current page
        """
        image_base64 = self._encode_image_to_base64(image_path)
        
        # Build context about previous entries
        previous_entries_json = [entry.model_dump() for entry in previous_entries[-5:]]  # Last 5 entries
        
        prompt = self._build_page_processing_prompt(
            page_number=page_number,
            current_markdown=current_page_markdown,
            previous_markdown=previous_page_markdown,
            previous_entries=previous_entries_json
        )
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": image_base64}
            ]
        }]
        
        response = self.client.chat.complete(
            model="pixtral-12b-latest",
            messages=messages,
            response_format=response_format_from_pydantic_model(TableOfContents),
            temperature=0.1,
            max_tokens=8000
        )
        
        try:
            response_data = json.loads(response.choices[0].message.content)
            toc_data = TableOfContents(**response_data)
            return toc_data.entries
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse page {page_number} response: {e}")
            return []
    
    def _build_page_processing_prompt(
        self,
        page_number: int,
        current_markdown: str,
        previous_markdown: Optional[str],
        previous_entries: List[Dict[str, Any]]
    ) -> str:
        """Build the prompt for processing a single page."""
        
        is_first_page = page_number == 1
        
        prompt = f"""You are analyzing page {page_number} of a Table of Contents. Your task is to extract ToC entries from this page that continue from the previous pages.

**CURRENT PAGE (Page {page_number}) MARKDOWN:**
```
{current_markdown}
```

**CURRENT PAGE IMAGE:**
The image shows the visual layout of this ToC page. Use it to understand the structure and hierarchy.

"""
        
        if not is_first_page and previous_markdown:
            prompt += f"""**PREVIOUS PAGE MARKDOWN (for context):**
```
{previous_markdown}
```

"""
        
        if previous_entries:
            prompt += f"""**PREVIOUS TOC ENTRIES (last few entries for context):**
```json
{json.dumps(previous_entries, indent=2)}
```

"""
        
        prompt += f"""**INSTRUCTIONS:**
{"This is the FIRST page of the ToC." if is_first_page else f"This is page {page_number} of the ToC, continuing from previous pages."}

1. **Extract ToC entries** from the current page image and markdown
2. **Maintain consistency** with previous entries in terms of:
   - ID numbering (continue from where previous entries left off)
   - Hierarchy structure (parent-child relationships)
   - Formatting patterns

3. **For each entry, determine:**
   - `type`: "section" or "subsection" based on hierarchy level
   - `text`: Exact title text as shown
   - `parent`: ID of parent section (null for top-level sections)
   - `section_start_page`: Page number where the actual content begins
   - `toc_list_page`: Page number where this entry appears in the ToC (this page: {page_number})

4. **Important guidelines:**
   - Use temporary ID numbers (will be reassigned later)
   - Maintain logical parent-child relationships
   - Extract page numbers accurately from the content
   - Only include entries visible on this specific page

**RESPONSE FORMAT:**
Provide a JSON object with an "entries" array containing TocEntry objects for this page only.

Example:
```json
{{
  "entries": [
    {{
      "id": 1,
      "type": "section",
      "text": "Chapter Title",
      "parent": null,
      "section_start_page": 25,
      "toc_list_page": {page_number}
    }}
  ]
}}
```"""
        
        return prompt
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """Encode an image file to base64 data URI."""
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type:
            mime_type = "image/jpeg"
        
        with open(image_path, "rb") as f:
            image_content = f.read()
        
        base64_encoded = base64.b64encode(image_content).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded}"