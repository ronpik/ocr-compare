#!/usr/bin/env python3
"""
Convert categorized images to PDF files

This script processes a folder containing images organized by categories (cover, intro, toc)
and creates a separate PDF for each category containing all images in order.

Usage:
    python images_to_category_pdfs.py /path/to/volume/folder
    python images_to_category_pdfs.py /path/to/volume/folder --output-dir ./pdfs
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import typer
from typing_extensions import Annotated


app = typer.Typer(
    help="Convert categorized images (cover, intro, toc) to separate PDF files"
)


def parse_image_filename(filename: str) -> Tuple[str, int]:
    """
    Parse image filename to extract category and order number.
    
    Args:
        filename: Image filename (e.g., "toc-2.jpg")
        
    Returns:
        Tuple of (category, order_number) or (None, None) if invalid
    """
    # Pattern matches: category-number.extension
    pattern = r'^(cover|intro|toc)-(\d+)\.(jpg|jpeg|png|gif|bmp)$'
    match = re.match(pattern, filename.lower())
    
    if match:
        category = match.group(1)
        order_num = int(match.group(2))
        return category, order_num
    
    return None, None


def collect_images_by_category(folder_path: Path) -> Dict[str, List[Tuple[int, Path]]]:
    """
    Collect and organize images by category.
    
    Args:
        folder_path: Path to the volume folder
        
    Returns:
        Dictionary mapping category to list of (order, path) tuples
    """
    categories = {
        'cover': [],
        'intro': [],
        'toc': []
    }
    
    # Scan folder for matching images
    for file_path in folder_path.iterdir():
        if file_path.is_file():
            category, order_num = parse_image_filename(file_path.name)
            if category:
                categories[category].append((order_num, file_path))
    
    # Sort each category by order number
    for category in categories:
        categories[category].sort(key=lambda x: x[0])
    
    return categories


def create_pdf_from_images(images: List[Tuple[int, Path]], output_path: Path) -> None:
    """
    Create a PDF from a list of images.
    
    Args:
        images: List of (order, path) tuples
        output_path: Path for the output PDF
    """
    if not images:
        return
    
    # Open all images
    pil_images = []
    for _, img_path in images:
        try:
            img = Image.open(img_path)
            # Convert to RGB if necessary (for PNG with alpha channel, etc.)
            if img.mode != 'RGB':
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[3])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            pil_images.append(img)
        except Exception as e:
            print(f"Warning: Failed to open {img_path}: {e}")
            continue
    
    if pil_images:
        # Save as PDF (first image with rest appended)
        pil_images[0].save(
            output_path,
            "PDF",
            save_all=True,
            append_images=pil_images[1:] if len(pil_images) > 1 else []
        )
        print(f"Created {output_path} with {len(pil_images)} pages")


@app.command()
def main(
    volume_folder: Annotated[
        str,
        typer.Argument(help="Path to the volume folder containing categorized images")
    ],
    output_dir: Annotated[
        str,
        typer.Option(
            "--output-dir",
            help="Directory to save the PDF files (default: same as input folder)"
        )
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show detailed processing information"
        )
    ] = False
) -> None:
    """
    Process a volume folder and create PDFs for each category.
    """
    # Validate input folder
    folder_path = Path(volume_folder)
    if not folder_path.exists():
        print(f"Error: Folder '{volume_folder}' does not exist")
        raise typer.Exit(code=1)
    
    if not folder_path.is_dir():
        print(f"Error: '{volume_folder}' is not a directory")
        raise typer.Exit(code=1)
    
    # Set output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = folder_path
    
    print(f"Processing folder: {folder_path}")
    print(f"Output directory: {output_path}")
    
    # Collect images by category
    categories = collect_images_by_category(folder_path)
    
    # Report findings
    total_images = 0
    for category, images in categories.items():
        if images:
            print(f"\nFound {len(images)} {category} images:")
            if verbose:
                for order, path in images:
                    print(f"  {category}-{order}: {path.name}")
            total_images += len(images)
    
    if total_images == 0:
        print("\nNo categorized images found in the folder.")
        print("Expected format: category-number.extension")
        print("Categories: cover, intro, toc")
        print("Example: cover-1.jpg, intro-1.jpg, toc-1.jpg")
        raise typer.Exit(code=1)
    
    # Create PDFs for each category
    print(f"\nCreating PDFs...")
    created_pdfs = []
    
    for category, images in categories.items():
        if images:
            pdf_path = output_path / f"{category}.pdf"
            try:
                create_pdf_from_images(images, pdf_path)
                created_pdfs.append(pdf_path)
            except Exception as e:
                print(f"Error creating {category}.pdf: {e}")
    
    # Summary
    if created_pdfs:
        print(f"\nSuccessfully created {len(created_pdfs)} PDF files:")
        for pdf in created_pdfs:
            print(f"  - {pdf}")
    else:
        print("\nNo PDFs were created.")


if __name__ == "__main__":
    app()