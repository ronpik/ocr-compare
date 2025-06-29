# Mistral OCR API Client

A refactored command-line tool for using Mistral's OCR API to extract text from documents and images.

## Features

- Support for both URL and local file inputs
- Process specific pages from multi-page documents
- Extract images with base64 encoding
- Structured data extraction using JSON schemas
- Configurable OCR model selection
- Comprehensive error handling
- Automatic markdown content extraction to separate file

## Setup

1. Install dependencies:
   ```bash
   pip install httpx typer typing-extensions
   ```

2. Set your Mistral API key:
   ```bash
   export MISTRAL_API_KEY="your-api-key-here"
   ```

## Usage

### Basic OCR from URL
```bash
python mistral_ocr.py --url https://example.com/document.pdf
```

### Process local file
```bash
python mistral_ocr.py --file ./path/to/document.pdf
```

### Process specific pages
```bash
python mistral_ocr.py --file document.pdf --pages "0,2,4"
```

### Include extracted images
```bash
python mistral_ocr.py --url https://example.com/doc.pdf --include-images
```

### Use specific model
```bash
python mistral_ocr.py --file document.pdf --model pixtral-12b-2409
```

### Extract markdown to custom file
```bash
python mistral_ocr.py --file document.pdf --markdown extracted_text.md
```

By default, the script will:
- Save the full JSON response to `output.json`
- Save the extracted markdown content to `output.md`

## API Changes

The refactored implementation includes:

1. **Proper document field structure**: Correctly handles both URL and base64 inputs with appropriate field names
2. **Model configuration**: Uses pixtral-12b-2409 as default (vision model)
3. **Enhanced parameters**: Support for all API parameters including image extraction options
4. **Better error handling**: Detailed error messages and proper status code handling
5. **Structured output support**: Helper function for creating JSON schema formats

## Advanced Features

- **Structured data extraction**: Use `bbox_annotation_format` and `document_annotation_format` for extracting structured data
- **Image filtering**: Control image extraction with `image_limit` and `image_min_size`
- **Flexible authentication**: Support for both environment variable and command-line API key

See `example_mistral_ocr.py` for detailed usage examples including structured data extraction.