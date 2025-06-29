# Images to Category PDFs

This script converts categorized images in a volume folder into separate PDF files for each category.

## Overview

The script processes a folder containing images organized by categories (cover, intro, toc) and creates a separate PDF for each category. Each image becomes a page in the corresponding PDF.

## Expected Folder Structure

Images must be named following the pattern: `category-number.extension`

- Categories: `cover`, `intro`, `toc`
- Number: Sequential order (1, 2, 3, ...)
- Extensions: jpg, jpeg, png, gif, bmp

Example folder structure:
```
volume_folder/
├── cover-1.jpg
├── intro-1.jpg
├── intro-2.jpg
├── intro-3.jpg
├── toc-1.jpg
└── toc-2.jpg
```

## Installation

Ensure you have the required dependencies:
```bash
pip install pillow typer
```

## Usage

### Basic usage:
```bash
python images_to_category_pdfs.py /path/to/volume/folder
```

This will create PDFs in the same folder as the input images.

### Specify output directory:
```bash
python images_to_category_pdfs.py /path/to/volume/folder --output-dir ./pdfs
```

### Verbose mode:
```bash
python images_to_category_pdfs.py /path/to/volume/folder --verbose
```

## Output

The script creates up to 3 PDF files:
- `cover.pdf` - Contains all cover images in order
- `intro.pdf` - Contains all intro images in order
- `toc.pdf` - Contains all table of contents images in order

Only PDFs for categories with images will be created.

## Features

- **Automatic sorting**: Images are sorted by their number suffix
- **Format handling**: Converts various image formats to PDF-compatible RGB
- **Error handling**: Skips problematic images with warnings
- **Flexible output**: Can save PDFs to a different directory
- **Case insensitive**: Handles both uppercase and lowercase extensions

## Example

Given a folder with:
- cover-1.jpg
- intro-1.jpg
- intro-2.jpg
- intro-3.jpg
- toc-1.jpg
- toc-2.jpg

Running:
```bash
python images_to_category_pdfs.py ./volume1
```

Will create:
- `cover.pdf` (1 page)
- `intro.pdf` (3 pages)
- `toc.pdf` (2 pages)

## Error Handling

The script will:
- Validate that the input folder exists
- Skip files that don't match the naming pattern
- Skip images that can't be opened
- Report detailed information about what was processed
- Exit with error code if no valid images are found