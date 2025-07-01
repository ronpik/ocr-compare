#!/usr/bin/env python3
"""
Convert categorized images to PDF files for multiple volumes.

This script processes a root folder containing multiple volume subfolders.
For each volume, it finds images organized by category (cover, intro, toc)
and creates a separate PDF for each category containing all images in order.

Usage:
    python images_to_category_pdfs.py /path/to/root/folder/with/volumes
    python images_to_category_pdfs.py /path/to/root/folder --output-dir ./all_pdfs
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
import typer
from typing_extensions import Annotated


app = typer.Typer(
    help="Convert categorized images (cover, intro, toc) from one or more volume folders into separate PDF files."
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


def process_volume_folder(volume_path: Path, output_dir: Optional[Path], verbose: bool):
    """
    Processes a single volume folder to create categorized PDFs.
    
    Args:
        volume_path: Path to the single volume folder.
        output_dir: The root directory for output PDFs.
        verbose: Flag for detailed logging.
    """
    print(f"\n--- Processing Volume: {volume_path.name} ---")

    # Set output directory for the current volume
    # If an output_dir is specified, create a subfolder for the volume.
    # Otherwise, save PDFs in the volume folder itself.
    current_output_path = output_dir / volume_path.name if output_dir else volume_path
    current_output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Output directory for this volume: {current_output_path}")

    # Collect images by category
    categories = collect_images_by_category(volume_path)
    
    # Report findings
    total_images = sum(len(imgs) for imgs in categories.values())
    if total_images == 0:
        print("No categorized images found (e.g., toc-1.jpg). Skipping.")
        return 0
    
    if verbose:
        print(f"Found {total_images} categorized images.")

    # Create PDFs for each category
    created_pdfs_for_volume = 0
    for category, images in categories.items():
        if images:
            pdf_path = current_output_path / f"{category}.pdf"
            try:
                create_pdf_from_images(images, pdf_path)
                created_pdfs_for_volume += 1
            except Exception as e:
                print(f"Error creating {category}.pdf for volume {volume_path.name}: {e}")
    
    if created_pdfs_for_volume > 0:
        print(f"Successfully created {created_pdfs_for_volume} PDF(s) for volume {volume_path.name}.")
    
    return created_pdfs_for_volume


@app.command()
def main(
    input_path: Annotated[
        str,
        typer.Argument(help="Path to the root folder containing volume subfolders, or to a single volume folder if --single-volume is used.")
    ],
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            "--output-dir",
            help="Root directory to save the PDF files. A subfolder will be created for each volume if processing a root directory."
        )
    ] = None,
    single_volume: Annotated[
        bool,
        typer.Option(
            "--single-volume",
            help="Treat the input path as a single volume folder instead of a root directory of volumes."
        )
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            help="Show detailed processing information"
        )
    ] = False
) -> None:
    """
    Process a directory of categorized images to create PDFs.
    Can process either a single volume folder or a root directory containing multiple volume subfolders.
    """
    # Validate input folder
    path_to_process = Path(input_path)
    if not path_to_process.is_dir():
        print(f"Error: Input path '{input_path}' does not exist or is not a directory.")
        raise typer.Exit(code=1)

    # Determine output path object if specified
    output_path_obj = Path(output_dir) if output_dir else None
    if output_path_obj:
        output_path_obj.mkdir(parents=True, exist_ok=True)

    if single_volume:
        # Process as a single folder
        print(f"Processing as a single volume folder: {path_to_process}")
        total_pdfs_created = process_volume_folder(path_to_process, output_path_obj, verbose)
    else:
        # Process as a root directory of volumes
        print(f"Processing as a root directory: {path_to_process}")
        volume_folders = [d for d in path_to_process.iterdir() if d.is_dir()]
        if not volume_folders:
            print(f"No volume subfolders found in '{input_path}'. If this is a single volume, use the --single-volume flag.")
            raise typer.Exit(code=1)

        print(f"Found {len(volume_folders)} volume folders to process.")
        total_pdfs_created = 0
        for volume_path in volume_folders:
            total_pdfs_created += process_volume_folder(volume_path, output_path_obj, verbose)

    print(f"\n--- Summary ---")
    print(f"Total PDFs created: {total_pdfs_created}")


if __name__ == "__main__":
    app()