"""
Cover Processor for handling cover images and thumbnail generation.

This module processes cover images and creates thumbnails for the final output.
"""

import os
from typing import Dict, Optional
from pathlib import Path
from PIL import Image


class CoverProcessor:
    """
    Processor for handling cover images and generating thumbnails.
    
    Processes cover image files and creates thumbnail versions for
    inclusion in the final JSON output.
    """
    
    def __init__(self, thumbnail_size: tuple = (200, 300)):
        """
        Initialize the cover processor.
        
        Args:
            thumbnail_size: Target size for thumbnails (width, height)
        """
        self.thumbnail_size = thumbnail_size
    
    def process_cover(self, cover_image_path: str, output_dir: str) -> Dict[str, str]:
        """
        Process a cover image and create a thumbnail.
        
        Args:
            cover_image_path: Path to the original cover image
            output_dir: Directory to save the thumbnail
            
        Returns:
            Dictionary with 'original' and 'thumbnail' paths
            
        Raises:
            FileNotFoundError: If the cover image doesn't exist
            Exception: For image processing errors
        """
        if not os.path.exists(cover_image_path):
            raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
        
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Generate thumbnail
            original_path = Path(cover_image_path)
            thumbnail_filename = f"thumbnail_{original_path.name}"
            thumbnail_path = Path(output_dir) / thumbnail_filename
            
            # Create thumbnail using PIL
            with Image.open(cover_image_path) as img:
                # Convert to RGB if necessary (for JPEG output)
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Create thumbnail maintaining aspect ratio
                img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Save thumbnail
                img.save(thumbnail_path, 'JPEG', quality=85)
            
            return {
                "original": str(Path(cover_image_path).resolve()),
                "thumbnail": str(thumbnail_path.resolve())
            }
            
        except Exception as e:
            raise Exception(f"Failed to process cover image: {e}")
    
    def process_cover_safe(self, cover_image_path: str, output_dir: str) -> Dict[str, Optional[str]]:
        """
        Safely process a cover image, returning partial results on failure.
        
        Args:
            cover_image_path: Path to the original cover image
            output_dir: Directory to save the thumbnail
            
        Returns:
            Dictionary with 'original' and 'thumbnail' paths (may be None on failure)
        """
        try:
            return self.process_cover(cover_image_path, output_dir)
        except Exception as e:
            print(f"Warning: Failed to process cover image {cover_image_path}: {e}")
            # Return original path if it exists, None for thumbnail
            if os.path.exists(cover_image_path):
                return {
                    "original": str(Path(cover_image_path).resolve()),
                    "thumbnail": None
                }
            else:
                return {
                    "original": None,
                    "thumbnail": None
                }
    
    def find_cover_image(self, folder_path: str) -> Optional[str]:
        """
        Find the first cover image in a folder.
        
        Args:
            folder_path: Path to the folder to search
            
        Returns:
            Path to the first cover image found, or None
        """
        folder = Path(folder_path)
        if not folder.exists():
            return None
        
        # Look for cover images in order of preference
        cover_patterns = ["cover-1.jpg", "cover-1.jpeg", "cover-1.png", "cover.jpg", "cover.jpeg", "cover.png"]
        
        for pattern in cover_patterns:
            cover_path = folder / pattern
            if cover_path.exists():
                return str(cover_path)
        
        # If no specific pattern found, look for any image starting with "cover"
        for file in folder.glob("cover*"):
            if file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                return str(file)
        
        return None