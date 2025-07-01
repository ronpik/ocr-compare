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
    python mistral_ocr.py --file document.pdf --pages "0,2,4" --model mistral-ocr-latest

    # Include base64 images in response
    python mistral_ocr.py --url https://example.com/doc.pdf --include-images

    # Save markdown content to a custom file
    python mistral_ocr.py --file document.pdf --markdown extracted_text.md
"""

import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from typing_extensions import Annotated
from mistralai import Mistral
from mistralai import models
from globalog import LOG
import typer

# Load API key from environment variable for security
# You can set it with: export MISTRAL_API_KEY="your-api-key"
API_KEY = os.environ.get("MISTRAL_API_KEY", "")

# Available OCR models (update based on actual available models)
DEFAULT_MODEL = "mistral-ocr-latest"

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


def run_mistral_ocr(
    api_key: str,
    model: str = DEFAULT_MODEL,
    url: Optional[str] = None,
    file_path: Optional[str] = None,
    pages: Optional[List[int]] = None,
    include_image_base64: bool = False,
    image_limit: Optional[int] = None,
    image_min_size: Optional[int] = None,
    bbox_annotation_format: Optional[Dict[str, Any]] = None,
    document_annotation_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Sends a request to the Mistral OCR API and returns the results.

    Args:
        api_key: Your Mistral API key.
        model: The model to use for OCR.
        url: The URL of the document to process.
        file_path: The local path of the document to process.
        pages: Specific pages to process (0-indexed). If None, processes all pages.
        include_image_base64: Whether to include base64 encoded images in the response.
        image_limit: Maximum number of images to extract.
        image_min_size: Minimum height and width of image to extract.
        bbox_annotation_format: Structured output format for bounding box annotations.
        document_annotation_format: Structured output format for document-level annotations.

    Returns:
        The JSON response from the API as a dictionary.

    Raises:
        ValueError: If the API key is not provided or no input is given.
    """
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set. Please provide a valid API key.")

    client = Mistral(api_key=api_key)
    document_url_for_ocr: str

    if file_path:
        LOG.info(f"Uploading file: {file_path}")
        try:
            with open(file_path, "rb") as f:
                content=f.read()
            
            file = models.File(
                file_name=Path(file_path).name,
                content=content
            )
            uploaded_file = client.files.upload(file=file, purpose="ocr")
        except FileNotFoundError:
            LOG.error(f"Error: The file {file_path} was not found.", exc_info=True)
            raise

        LOG.info("File uploaded, getting signed URL...")
        signed_url_response = client.files.get_signed_url(file_id=uploaded_file.id)
        document_url_for_ocr = signed_url_response.url
    elif url:
        document_url_for_ocr = url
    else:
        raise ValueError("Either a URL or a file path must be provided.")

    document = {
        "type": "document_url",
        "document_url": document_url_for_ocr,
    }

    try:
        LOG.info("Sending request to Mistral OCR API...")
        ocr_response = client.ocr.process(
            model=model,
            document=document,
            pages=pages,
            include_image_base64=include_image_base64,
            image_limit=image_limit,
            image_min_size=image_min_size,
            bbox_annotation_format=bbox_annotation_format,
            document_annotation_format=document_annotation_format,
        )
        LOG.info("Request successful.")
        return ocr_response.model_dump()
    except Exception as e:
        LOG.error(f"An unexpected error occurred", exc_info=True)
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
        LOG.info("Error: Please provide either a --url or a --file.")
        raise typer.Exit(code=1)

    if document_url and file_path:
        LOG.info("Error: Please provide either --url or --file, not both.")
        raise typer.Exit(code=1)

    # Use provided API key or fallback to environment variable
    final_api_key = api_key or API_KEY

    if not final_api_key:
        LOG.info("Error: No API key provided. "
              "Set MISTRAL_API_KEY environment variable or use --api-key option.")
        raise typer.Exit(code=1)

    # Parse pages if provided
    pages_list = None
    if pages:
        try:
            pages_list = [int(p.strip()) for p in pages.split(",")]
        except ValueError:
            LOG.error("Error: Invalid pages format. Use comma-separated numbers, e.g., '0,1,2'", exc_info=True)
            raise typer.Exit(code=1)

    if file_path:
        LOG.info(f"Processing local file: {file_path}")
    if document_url:
        LOG.info(f"Processing document from URL: {document_url}")

    try:
        LOG.info(f"Using model: {model}")
        result = run_mistral_ocr(
            api_key=final_api_key,
            url=document_url,
            file_path=file_path,
            pages=pages_list,
            include_image_base64=include_images,
            model=model,
        )
        LOG.info(f"\nOCR Result successfully generated. Writing to {output_file}...")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        LOG.info("Done.")

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
                LOG.info(f"\nExtracting markdown content to {md_output_file}...")
                with open(md_output_file, "w", encoding="utf-8") as f:
                    f.write("".join(markdown_content))
                LOG.info(f"Markdown content saved to {md_output_file}")

        # Print a summary of the results
        if "pages" in result:
            LOG.info(f"\nProcessed {len(result['pages'])} pages.")
            if "usage_info" in result:
                usage = result["usage_info"]
                LOG.info(f"Pages processed: {usage.get('pages_processed', 'N/A')}")
                LOG.info(f"Document size: {usage.get('doc_size_bytes', 'N/A')} bytes")
    except Exception as e:
        LOG.error(f"\nAn unexpected error occurred", exc_info=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
