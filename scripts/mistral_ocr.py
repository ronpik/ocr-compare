"""
Mistral OCR API Client

A command-line tool for using Mistral's OCR API to extract text from documents and images.
Supports both URL and local file inputs with customizable extraction parameters.

Usage Examples:
    # Process a document from URL
    python mistral_ocr.py --url https://example.com/document.pdf

    # Process a local file
    python mistral_ocr.py --file ./path/to/image.png

    # Process specific pages with custom model
    python mistral_ocr.py --file document.pdf --pages "0,2,4" --model pixtral-12b-2409

    # Include base64 images in response
    python mistral_ocr.py --url https://example.com/doc.pdf --include-images

    # Save markdown content to a custom file
    python mistral_ocr.py --file document.pdf --markdown extracted_text.md
"""

import httpx
import os
import json
from typing import Dict, Any, List, Optional
import typer
from typing_extensions import Annotated
import base64
import mimetypes

# Load API key from environment variable for security
# You can set it with: export MISTRAL_API_KEY="your-api-key"
API_KEY = os.environ.get("MISTRAL_API_KEY", "")
API_URL = "https://api.mistral.ai/v1/ocr"

# Available OCR models (update based on actual available models)
DEFAULT_MODEL = "mistral-ocr-2505"  # Using pixtral as it's a vision model

app = typer.Typer(
    help="Mistral OCR CLI tool for document processing with AI-powered text extraction."
)


def create_json_schema_format(
    name: str,
    description: str,
    properties: Dict[str, Dict[str, Any]],
    strict: bool = False
) -> Dict[str, Any]:
    """
    Create a JSON schema format for structured output.

    Args:
        name: Name of the schema.
        description: Description of what this schema extracts.
        properties: Dictionary of property definitions.
        strict: Whether to enforce strict schema validation.

    Returns:
        A dictionary representing the JSON schema format.
    """
    return {
        "type": "text",
        "json_schema": {
            "name": name,
            "description": description,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties.keys()) if strict else []
            },
            "strict": strict
        }
    }


def encode_file_to_base64(file_path: str) -> str:
    """
    Encodes a file to a base64 data URI.

    Args:
        file_path: The path to the file.

    Returns:
        The base64 encoded data URI.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    try:
        mime_type, _ = mimetypes.guess_type(file_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            file_content = f.read()

        base64_encoded_content = base64.b64encode(file_content).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_content}"
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        raise


def run_mistral_ocr(
    document_input: str,
    api_key: str,
    is_base64: bool = False,
    pages: Optional[List[int]] = None,
    include_image_base64: bool = False,
    model: str = DEFAULT_MODEL,
    image_limit: Optional[int] = None,
    image_min_size: Optional[int] = None,
    bbox_annotation_format: Optional[Dict[str, Any]] = None,
    document_annotation_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sends a request to the Mistral OCR API and returns the results.

    Args:
        document_input: The URL or base64 data URI of the document to process.
        api_key: Your Mistral API key.
        is_base64: Whether the document_input is a base64 data URI.
        pages: Specific pages to process (0-indexed). If None, processes all pages.
        include_image_base64: Whether to include base64 encoded images in the response.
        model: The model to use for OCR (default: pixtral-12b-2409).
        image_limit: Maximum number of images to extract.
        image_min_size: Minimum height and width of image to extract.
        bbox_annotation_format: Structured output format for bounding box annotations.
        document_annotation_format: Structured output format for document-level annotations.

    Returns:
        The JSON response from the API as a dictionary.

    Raises:
        ValueError: If the API key is not provided.
        httpx.HTTPStatusError: If the API returns an error status code.
        httpx.RequestError: For other request-related issues.
    """
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set. Please provide a valid API key.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    # Build the document field based on input type
    if is_base64:
        # For base64 input, we need to use image_url field
        document = {
            "image_url": document_input,
            "type": "image_url"
        }
    else:
        # For URL input
        document = {
            "document_url": document_input,
            "document_name": (
                os.path.basename(document_input) if document_input else "document"
            ),
            "type": "document_url"
        }

    # Build the payload
    payload = {
        "model": model,
        "document": document,
    }

    # Add optional parameters
    if pages is not None:
        payload["pages"] = pages

    payload["include_image_base64"] = include_image_base64

    if image_limit is not None:
        payload["image_limit"] = image_limit

    if image_min_size is not None:
        payload["image_min_size"] = image_min_size

    if bbox_annotation_format is not None:
        payload["bbox_annotation_format"] = bbox_annotation_format

    if document_annotation_format is not None:
        payload["document_annotation_format"] = document_annotation_format

    try:
        with httpx.Client() as client:
            print("Sending request to Mistral OCR API...")
            response = client.post(API_URL, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            print("Request successful.")
            return response.json()
    except httpx.HTTPStatusError as e:
        print(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        raise
    except httpx.RequestError as e:
        print(f"An error occurred while requesting {e.request.url!r}: {e}")
        raise


@app.command()
def main(
    document_url: Annotated[
        Optional[str],
        typer.Option(
            "--url",
            help="URL of the document (e.g., PDF, PNG, JPEG) to process."
        ),
    ] = None,
    file_path: Annotated[
        Optional[str],
        typer.Option(
            "--file",
            help="Path to a local image or PDF file to process."
        ),
    ] = None,
    output_file: Annotated[
        str, typer.Option(help="Path to save the JSON output.")
    ] = "output.json",
    markdown_file: Annotated[
        Optional[str],
        typer.Option(
            "--markdown",
            help="Path to save the extracted markdown content. Default: output.md"
        ),
    ] = None,
    pages: Annotated[
        Optional[str],
        typer.Option(
            "--pages",
            help="Comma-separated list of page numbers to process (0-indexed). Example: '0,1,2'"
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help=f"Model to use for OCR. Default: {DEFAULT_MODEL}"
        ),
    ] = DEFAULT_MODEL,
    include_images: Annotated[
        bool,
        typer.Option(
            "--include-images",
            help="Include base64 encoded images in the response."
        ),
    ] = False,
    api_key: Annotated[
        Optional[str],
        typer.Option(
            "--api-key",
            help="Mistral API key. Can also be set via MISTRAL_API_KEY env var."
        ),
    ] = None,
) -> None:
    """Main function to run the Mistral OCR example."""
    if not document_url and not file_path:
        print("Error: Please provide either a --url or a --file.")
        raise typer.Exit(code=1)

    if document_url and file_path:
        print("Error: Please provide either --url or --file, not both.")
        raise typer.Exit(code=1)

    # Use provided API key or fallback to environment variable
    final_api_key = api_key or API_KEY

    if not final_api_key:
        print("Error: No API key provided. "
              "Set MISTRAL_API_KEY environment variable or use --api-key option.")
        raise typer.Exit(code=1)

    # Parse pages if provided
    pages_list = None
    if pages:
        try:
            pages_list = [int(p.strip()) for p in pages.split(",")]
        except ValueError:
            print("Error: Invalid pages format. Use comma-separated numbers, e.g., '0,1,2'")
            raise typer.Exit(code=1)

    document_to_process = ""
    is_base64_input = False

    if file_path:
        print(f"Processing local file: {file_path}")
        try:
            document_to_process = encode_file_to_base64(file_path)
            is_base64_input = True
        except FileNotFoundError:
            raise typer.Exit(code=1)

    if document_url:
        document_to_process = document_url
        is_base64_input = False
        print(f"Processing document from URL: {document_to_process}")

    try:
        print(f"Using model: {model}")
        result = run_mistral_ocr(
            document_to_process,
            final_api_key,
            is_base64=is_base64_input,
            pages=pages_list,
            include_image_base64=include_images,
            model=model,
        )
        print(f"\nOCR Result successfully generated. Writing to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print("Done.")

        # Extract and save markdown content
        if "pages" in result and len(result["pages"]) > 0:
            # Determine markdown output file path
            md_output_file = markdown_file if markdown_file else "output.md"

            # Extract markdown content from all pages
            markdown_content = []
            for page in result["pages"]:
                if "markdown" in page:
                    page_idx = page.get("index", "?")
                    # Add page separator
                    if markdown_content:  # Not the first page
                        markdown_content.append("\n\n---\n\n")
                    markdown_content.append(f"<!-- Page {page_idx + 1} -->\n\n")
                    markdown_content.append(page["markdown"])

            if markdown_content:
                print(f"\nExtracting markdown content to {md_output_file}...")
                with open(md_output_file, "w", encoding="utf-8") as f:
                    f.write("".join(markdown_content))
                print(f"Markdown content saved to {md_output_file}")

        # Print a summary of the results
        if "pages" in result:
            print(f"\nProcessed {len(result['pages'])} pages.")
            if "usage_info" in result:
                usage = result["usage_info"]
                print(f"Pages processed: {usage.get('pages_processed', 'N/A')}")
                print(f"Document size: {usage.get('doc_size_bytes', 'N/A')} bytes")
    except httpx.HTTPStatusError as e:
        print(f"\nHTTP Error {e.response.status_code}: {e.response.text}")
        raise typer.Exit(code=1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
