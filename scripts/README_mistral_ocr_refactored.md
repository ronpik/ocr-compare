# Mistral OCR API Client - Refactored Version

This is a refactored version of the Mistral OCR client that uses the official `mistralai` Python SDK instead of direct HTTP requests. It includes new functionality for extracting structured Table of Contents information from documents.

## New Features

### 1. **Official Mistral Python Client**
- Uses the official `mistralai` SDK for cleaner, more maintainable code
- Better error handling and type safety
- Automatic retry and timeout handling

### 2. **Table of Contents Extraction**
- Intelligently identifies and extracts Table of Contents from documents
- Structured output with section hierarchy (sections and subsections)
- Captures page numbers when available
- Outputs in both JSON and Markdown formats

### 3. **Improved Document Handling**
- Better detection of image vs PDF inputs
- Proper handling of different document types
- Support for both URLs and local files

## Installation

Install the required dependencies:

```bash
pip install mistralai pydantic typer httpx
```

Or add them to your project:

```bash
pip install -e ".[mistral]"
```

## Usage Examples

### Basic OCR

```bash
# Process a document from URL
python mistral_ocr_refactored.py --url https://example.com/document.pdf

# Process a local file
python mistral_ocr_refactored.py --file ./document.pdf

# Process specific pages
python mistral_ocr_refactored.py --file document.pdf --pages "0,1,2"
```

### Table of Contents Extraction

```bash
# Extract ToC and save as JSON
python mistral_ocr_refactored.py --file document.pdf --extract-toc

# Extract ToC with custom output paths
python mistral_ocr_refactored.py --file document.pdf \
    --extract-toc \
    --toc-output my_toc.json \
    --toc-markdown my_toc.md

# Process only first few pages (ToC is usually at the beginning)
python mistral_ocr_refactored.py --file document.pdf \
    --extract-toc \
    --pages "0,1,2,3,4"
```

### Full Example with All Options

```bash
python mistral_ocr_refactored.py \
    --file research_paper.pdf \
    --output ocr_result.json \
    --markdown extracted_text.md \
    --extract-toc \
    --toc-output table_of_contents.json \
    --toc-markdown table_of_contents.md \
    --pages "0,1,2,3,4" \
    --include-images \
    --model mistral-ocr-latest
```

## Table of Contents Structure

The extracted Table of Contents follows this structure:

```json
{
  "entries": [
    {
      "type": "section",
      "text": "Introduction",
      "parent": null,
      "page_number": 1
    },
    {
      "type": "subsection",
      "text": "Background and Motivation",
      "parent": "Introduction",
      "page_number": 2
    },
    {
      "type": "section",
      "text": "Methodology",
      "parent": null,
      "page_number": 5
    }
  ]
}
```

### Fields Explanation:
- **type**: Either "section" (main sections) or "subsection" (sub-sections)
- **text**: The title/name of the section as it appears in the ToC
- **parent**: For subsections, the name of the parent section. `null` for main sections
- **page_number**: The page where this section starts (if available in the ToC)

## Programmatic Usage

```python
from mistralai import Mistral
from mistralai.extra import response_format_from_pydantic_model
from mistralai.models.ocr import DocumentURLChunk
from pydantic import BaseModel, Field
from typing import List, Optional

# Define ToC structure
class TocEntry(BaseModel):
    type: str
    text: str
    parent: Optional[str] = None
    page_number: Optional[int] = None

class TableOfContents(BaseModel):
    entries: List[TocEntry]

# Initialize client
client = Mistral(api_key="your-api-key")

# Process document with ToC extraction
response = client.ocr.process(
    model="mistral-ocr-latest",
    document=DocumentURLChunk(document_url="https://example.com/doc.pdf"),
    bbox_annotation_format=response_format_from_pydantic_model(TableOfContents),
    include_image_base64=False
)
```

## Comparison with Original Version

| Feature | Original | Refactored |
|---------|----------|------------|
| HTTP Client | httpx (manual) | mistralai SDK |
| Error Handling | Basic | Comprehensive |
| ToC Extraction | ❌ | ✅ |
| Type Safety | Limited | Full (Pydantic) |
| Code Complexity | Higher | Lower |
| Maintenance | Manual API updates | SDK handles updates |

## Tips for Best Results

1. **For ToC Extraction**: Process only the first few pages (usually 0-5) where the ToC is located
2. **Model Selection**: Use `mistral-ocr-latest` for best results
3. **Large Documents**: Use the `--pages` option to process specific pages
4. **API Key**: Set the `MISTRAL_API_KEY` environment variable for security

## Error Handling

The refactored version provides better error messages:
- Clear indication of missing API keys
- Detailed API error responses
- Validation of input parameters
- Graceful handling of parsing errors

## Future Enhancements

- Support for extracting other structured elements (figures, tables)
- Batch processing of multiple documents
- Integration with document databases
- Support for additional output formats