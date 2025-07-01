"""
Cover Processor for handling cover images and thumbnail generation.

This module processes cover images and creates thumbnails for the final output.
"""

import os
import base64
from typing import Dict, Optional
from pathlib import Path
from PIL import Image
from io import BytesIO


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
    
    def process_cover(self, cover_image_path: str, volume_folder: str, output_dir: str) -> Dict[str, Optional[str]]:
        """
        Process a cover image and create a thumbnail with base64 encoding.
        
        Args:
            cover_image_path: Path to the original cover image
            volume_folder: Path to the volume folder (for relative path calculation)
            output_dir: Directory to save the thumbnail
            
        Returns:
            Dictionary with 'original', 'thumbnail', and 'thumbnail_base64'
            
        Raises:
            FileNotFoundError: If the cover image doesn't exist
            Exception: For image processing errors
        """
        if not os.path.exists(cover_image_path):
            raise FileNotFoundError(f"Cover image not found: {cover_image_path}")
        
        try:
            # Create output directory if it doesn't exist
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            volume_path = Path(volume_folder)
            volume_name = volume_path.name
            
            # Create standardized cover-1.png in the volume folder
            standardized_cover_path = volume_path / "cover-1.png"
            
            # Generate thumbnail and base64
            original_path = Path(cover_image_path)
            thumbnail_filename = f"thumbnail_{original_path.stem}.jpg"
            thumbnail_path = Path(output_dir) / thumbnail_filename
            
            # Process image with PIL
            with Image.open(cover_image_path) as img:
                # Convert to RGB if necessary
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Save standardized cover-1.png
                img.save(standardized_cover_path, 'PNG', quality=95)
                
                # Create thumbnail maintaining aspect ratio
                thumbnail_img = img.copy()
                thumbnail_img.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
                
                # Save thumbnail to file
                thumbnail_img.save(thumbnail_path, 'JPEG', quality=85)
                
                # Generate base64 thumbnail
                buffer = BytesIO()
                thumbnail_img.save(buffer, format='JPEG', quality=85)
                thumbnail_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                thumbnail_base64_data_uri = f"data:image/jpeg;base64,{thumbnail_base64}"
            
            # Return relative path for original
            relative_original_path = f"{volume_name}/cover-1.png"
            
            return {
                "original": relative_original_path,
                "thumbnail": str(thumbnail_path.resolve()),
                "thumbnail_base64": thumbnail_base64_data_uri
            }
            
        except Exception as e:
            raise Exception(f"Failed to process cover image: {e}")
    
    def process_cover_safe(self, cover_image_path: str, volume_folder: str, output_dir: str) -> Dict[str, Optional[str]]:
        """
        Safely process a cover image, returning partial results on failure.
        
        Args:
            cover_image_path: Path to the original cover image
            volume_folder: Path to the volume folder (for relative path calculation)
            output_dir: Directory to save the thumbnail
            
        Returns:
            Dictionary with 'original', 'thumbnail', and 'thumbnail_base64' (may be None on failure)
        """
        try:
            return self.process_cover(cover_image_path, volume_folder, output_dir)
        except Exception as e:
            print(f"Warning: Failed to process cover image {cover_image_path}: {e}")
            # Return relative path if it exists, None for others
            if os.path.exists(cover_image_path):
                volume_name = Path(volume_folder).name
                return {
                    "original": f"{volume_name}/cover-1.png",
                    "thumbnail": None,
                    "thumbnail_base64": None
                }
            else:
                return {
                    "original": None,
                    "thumbnail": None,
                    "thumbnail_base64": None
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