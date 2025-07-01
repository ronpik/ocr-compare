# numero_source

A modular pipeline for processing book materials including cover images, introduction text, and table of contents using Mistral's OCR API.

## Features

- **Cover Processing**: Handles cover images and generates thumbnails
- **Introduction OCR**: Extracts and combines text from intro PDFs  
- **Table of Contents Parsing**: Sophisticated page-by-page ToC extraction using LangGraph workflow
- **Structured Output**: Generates clean JSON output with all components

## Package Structure

```
numero_source/
├── __init__.py              # Main package exports
├── pipeline.py              # Main orchestrator (BookProcessor)
├── schemas.py               # Output data schemas
├── ocr/
│   ├── __init__.py
│   └── mistral_ocr.py       # MistralOCR class
├── toc/
│   ├── __init__.py
│   ├── models.py            # TocEntry, TableOfContents schemas
│   └── parser.py            # TableOfContentsParser
└── processors/
    ├── __init__.py
    ├── cover.py             # CoverProcessor
    └── intro.py             # IntroParser
```

## Requirements

- Python 3.8+
- Mistral API key
- Required packages:
  - `mistralai`
  - `pydantic`
  - `langgraph`
  - `pillow` (for thumbnail generation)

## Usage

### Basic Usage

```python
from numero_source import BookProcessor

# Initialize with API key
processor = BookProcessor(api_key='your-mistral-api-key')

# Process a single volume
result = processor.process_volume('path/to/volume/folder')

# Access results
print(f"Cover: {result.cover.original}")
print(f"Intro length: {len(result.intro)} characters")
print(f"ToC entries: {len(result.toc.entries)}")
```

### Volume Folder Structure

Each volume folder should contain:

```
volume_folder/
├── cover-1.jpg              # Cover image (required)
├── cover.pdf               # Cover PDF (optional)
├── intro-1.jpg             # Intro page images (optional)
├── intro-2.jpg
├── intro.pdf               # Introduction PDF (optional)
├── toc-1.jpg               # ToC page images (required)
├── toc-2.jpg
├── toc-3.jpg
└── toc.pdf                 # Table of contents PDF (required)
```

### Processing Multiple Volumes

```python
# Process all volumes in a directory
results = processor.process_multiple_volumes('path/to/volumes/')

for folder_name, result in results.items():
    print(f"Processed {folder_name}: {len(result.toc.entries)} ToC entries")
```

### Command Line Usage

```bash
# Process a single volume
python -m numero_source.pipeline -i volume_folder/

# Process multiple volumes
python -m numero_source.pipeline -i volumes_folder/ --multiple

# Specify output directory
python -m numero_source.pipeline -i volume_folder/ --output-dir results/

# Enable debug mode to save intermediate model responses
python -m numero_source.pipeline -i volume_folder/ --debug
```

### Debug Mode

Enable debug mode with the `--debug` flag to save all intermediate model responses:

```bash
# Using the entry point script
MISTRAL_API_KEY=your-key uv run scripts/run_numero_source.py -i volume_folder/ --debug

# Using the pipeline directly
MISTRAL_API_KEY=your-key uv run scripts/numero_source/pipeline.py -i volume_folder/ --debug
```

Debug mode creates an `intermediates/` folder in the output directory containing:

- `intro_ocr_*.json` - OCR responses from intro PDF processing
- `toc_ocr_*.json` - OCR responses from ToC PDF processing  
- `toc_page_*_response_*.json` - Individual page processing responses from Pixtral model

These files contain the raw API responses for debugging and analysis.

## Output Format

The pipeline generates a JSON file for each processed volume:

```json
{
  "cover": {
    "original": "volume_name/cover-1.png",
    "thumbnail": "/path/to/thumbnail_cover-1.jpg", 
    "thumbnail_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
  },
  "intro": "Combined introduction text from OCR...",
  "toc": {
    "entries": [
      {
        "id": 1,
        "type": "section",
        "text": "Chapter 1: Introduction",
        "parent": null,
        "section_start_page": 15,
        "toc_list_page": 1
      },
      {
        "id": 2,
        "type": "subsection", 
        "text": "1.1 Overview",
        "parent": 1,
        "section_start_page": 16,
        "toc_list_page": 1
      }
    ]
  },
  "folder_name": "volume_name"
}
```

## Components

### MistralOCR
- Processes PDFs and images using Mistral's OCR API
- Extracts structured markdown content
- Handles file uploads and URL processing

### TableOfContentsParser
- Implements page-by-page ToC parsing workflow
- Uses Pixtral model for visual understanding
- Builds ToC incrementally with context from previous pages
- Maintains hierarchy relationships between sections

### CoverProcessor
- Finds and processes cover images
- Creates standardized `cover-1.png` in the volume folder
- Generates thumbnails with configurable size
- Provides base64-encoded thumbnails for web display
- Returns relative paths for portability
- Handles multiple image formats

### IntroParser
- Processes introduction PDFs with OCR
- Combines content from multiple pages
- Removes page separators for clean output

## Error Handling

The pipeline includes comprehensive error handling:

- Graceful fallbacks for missing optional files
- Safe processing methods that continue on errors
- Detailed error messages and warnings
- Fallback ToC extraction if advanced parsing fails

## Testing

Run the test suite:

```bash
python test_numero_source.py
```

## Implementation Notes

This implementation is based on existing OCR and ToC parsing scripts but provides:

- **Modular Architecture**: Clean separation of concerns
- **Simplified Workflow**: Streamlined ToC parsing without complex validation loops  
- **Reusable Components**: Each component can be used independently
- **Error Resilience**: Robust handling of various failure scenarios
- **Clean API**: Simple interfaces for all operations

The ToC parsing workflow is significantly simplified compared to the original LangGraph implementation, focusing on direct page-by-page processing rather than complex multi-step validation and execution cycles.