"""
Main pipeline for processing book folders.

This module orchestrates all the components to process a complete book folder
containing cover, intro, and ToC files.
"""

import json
import os
from typing import Optional, Dict
from pathlib import Path

from globalog import LOG

from .ocr.mistral_ocr import MistralOCR
from .toc.parser import TableOfContentsParser
from .toc.models import TableOfContents
from .processors.cover import CoverProcessor
from .processors.intro import IntroParser
from .schemas import BookProcessingResult, CoverInfo


class BookProcessor:
    """
    Main processor for handling complete book folders.
    
    Orchestrates the processing of cover images, introduction PDFs,
    and table of contents to produce a structured JSON output.
    """
    
    def __init__(self, api_key: Optional[str] = None, output_base_dir: str = "results"):
        """
        Initialize the book processor.
        
        Args:
            api_key: Mistral API key for OCR processing
            output_base_dir: Base directory for output files
        """
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is required")
        
        self.output_base_dir = Path(output_base_dir)
        
        # Initialize components
        self.ocr = MistralOCR(api_key=self.api_key)
        self.toc_parser = TableOfContentsParser(api_key=self.api_key)
        self.cover_processor = CoverProcessor()
        self.intro_parser = IntroParser(api_key=self.api_key)
    
    def process_volume(self, volume_folder: str) -> BookProcessingResult:
        """
        Process a complete volume folder.
        
        Args:
            volume_folder: Path to the volume folder containing cover, intro, and ToC files
            
        Returns:
            BookProcessingResult with all processed components
            
        Raises:
            FileNotFoundError: If required files are missing
            Exception: For processing errors
        """
        volume_path = Path(volume_folder)
        if not volume_path.exists():
            raise FileNotFoundError(f"Volume folder not found: {volume_folder}")
        
        folder_name = volume_path.name
        
        # Create output directory for this volume
        output_dir = self.output_base_dir / folder_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        LOG.info(f"Processing volume: {folder_name}")
        
        # Process cover
        LOG.info("Processing cover...")
        cover_info = self._process_cover(volume_path, output_dir)
        
        # Process intro
        LOG.info("Processing intro...")
        intro_text = self._process_intro(volume_path)
        
        # Process ToC
        LOG.info("Processing table of contents...")
        toc = self._process_toc(volume_path)
        
        # Create result
        result = BookProcessingResult(
            cover=cover_info,
            intro=intro_text,
            toc=toc,
            folder_name=folder_name
        )
        
        # Save result to JSON file
        output_file = output_dir / f"{folder_name}_processed.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        
        LOG.info(f"Processing complete. Results saved to: {output_file}")
        return result
    
    def _process_cover(self, volume_path: Path, output_dir: Path) -> CoverInfo:
        """Process the cover image."""
        cover_image_path = self.cover_processor.find_cover_image(str(volume_path))
        
        if cover_image_path:
            cover_data = self.cover_processor.process_cover_safe(
                cover_image_path, 
                str(output_dir / "thumbnails")
            )
            return CoverInfo(**cover_data)
        else:
            LOG.info("Warning: No cover image found")
            return CoverInfo(original=None, thumbnail=None)
    
    def _process_intro(self, volume_path: Path) -> str:
        """Process the introduction PDF."""
        intro_pdf_path = volume_path / "intro.pdf"
        
        if intro_pdf_path.exists():
            return self.intro_parser.parse_intro_safe(str(intro_pdf_path))
        else:
            LOG.info("Warning: No intro.pdf found")
            return ""
    
    def _process_toc(self, volume_path: Path) -> TableOfContents:
        """Process the table of contents."""
        toc_pdf_path = volume_path / "toc.pdf"
        
        if not toc_pdf_path.exists():
            raise FileNotFoundError(f"Required toc.pdf not found in {volume_path}")
        
        # Find ToC image files
        toc_images = list(volume_path.glob("toc-*.jpg"))
        if not toc_images:
            raise FileNotFoundError(f"No toc-*.jpg images found in {volume_path}")
        
        return self.toc_parser.parse_from_pdf(str(toc_pdf_path), str(volume_path))
    
    def process_multiple_volumes(self, volumes_folder: str) -> Dict[str, BookProcessingResult]:
        """
        Process multiple volume folders.
        
        Args:
            volumes_folder: Path to folder containing multiple volume subfolders
            
        Returns:
            Dictionary mapping folder names to processing results
        """
        volumes_path = Path(volumes_folder)
        if not volumes_path.exists():
            raise FileNotFoundError(f"Volumes folder not found: {volumes_folder}")
        
        results = {}
        
        # Process each subfolder
        for subfolder in volumes_path.iterdir():
            if subfolder.is_dir():
                try:
                    LOG.info(f"\n=== Processing {subfolder.name} ===")
                    result = self.process_volume(str(subfolder))
                    results[subfolder.name] = result
                except Exception as e:
                    LOG.error(f"Error processing {subfolder.name}", exc_info=True)
                    continue
        
        LOG.info(f"\nProcessed {len(results)} volumes successfully")
        return results


def main():
    """CLI entry point for the book processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process book volumes with OCR")
    parser.add_argument("volume_path", type=Path, required=True, help="Path to volume folder or volumes folder")
    parser.add_argument("--output-dir", default="numero_results", help="Output directory")
    parser.add_argument("--api-key", help="Mistral API key (or set MISTRAL_API_KEY env var)")
    parser.add_argument("--multiple", action="store_true", help="Process multiple volumes in folder")
    
    args = parser.parse_args()

    volume_path = args.volume_path
    output_dir = args.output_dir
    if not output_dir:
        output_dir = Path(volume_path) / "results"
    
    try:
        processor = BookProcessor(api_key=args.api_key, output_base_dir=output_dir)
        
        if args.multiple:
            results = processor.process_multiple_volumes(volume_path)
            LOG.info(f"\nProcessed {len(results)} volumes")
        else:
            result = processor.process_volume(volume_path)
            LOG.info(f"\nProcessed volume: {result.folder_name}")
            
    except Exception as e:
        LOG.error(f"Error", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())