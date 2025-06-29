#!/usr/bin/env python3
"""
Example usage of the refactored Mistral OCR client.

This script demonstrates how to use various features of the mistral_ocr module,
including custom annotation formats and advanced options.
"""

import json
from mistral_ocr import run_mistral_ocr, create_json_schema_format, encode_file_to_base64


def example_basic_url_ocr():
    """Example: Basic OCR from URL"""
    print("\n=== Example 1: Basic URL OCR ===")
    
    # Replace with your actual API key
    api_key = "your-api-key-here"
    document_url = "https://example.com/sample-document.pdf"
    
    try:
        result = run_mistral_ocr(
            document_input=document_url,
            api_key=api_key,
            model="pixtral-12b-2409"
        )
        
        print(f"Processed {len(result.get('pages', []))} pages")
        
        # Extract text from first page
        if result.get('pages'):
            first_page = result['pages'][0]
            print(f"First page text preview: {first_page.get('markdown', '')[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")


def example_local_file_with_options():
    """Example: OCR local file with specific pages and image extraction"""
    print("\n=== Example 2: Local File with Options ===")
    
    api_key = "your-api-key-here"
    file_path = "path/to/your/document.pdf"
    
    try:
        # Encode file to base64
        document_base64 = encode_file_to_base64(file_path)
        
        # Process specific pages with images
        result = run_mistral_ocr(
            document_input=document_base64,
            api_key=api_key,
            is_base64=True,
            pages=[0, 2, 4],  # Process pages 1, 3, and 5 (0-indexed)
            include_image_base64=True,
            image_limit=5,
            image_min_size=100
        )
        
        # Extract images from results
        for page in result.get('pages', []):
            page_idx = page.get('index', 'unknown')
            images = page.get('images', [])
            print(f"Page {page_idx}: Found {len(images)} images")
            
            for img in images:
                print(f"  - Image {img.get('id')}: "
                      f"{img.get('bottom_right_x', 0) - img.get('top_left_x', 0)}x"
                      f"{img.get('bottom_right_y', 0) - img.get('top_left_y', 0)} px")
                      
    except Exception as e:
        print(f"Error: {e}")


def example_structured_extraction():
    """Example: Using structured output formats for data extraction"""
    print("\n=== Example 3: Structured Data Extraction ===")
    
    api_key = "your-api-key-here"
    document_url = "https://example.com/invoice.pdf"
    
    # Define schema for invoice data extraction
    invoice_schema = create_json_schema_format(
        name="invoice_extraction",
        description="Extract invoice information",
        properties={
            "invoice_number": {
                "type": "string",
                "description": "The invoice number"
            },
            "date": {
                "type": "string",
                "description": "Invoice date in YYYY-MM-DD format"
            },
            "total_amount": {
                "type": "number",
                "description": "Total amount due"
            },
            "vendor_name": {
                "type": "string",
                "description": "Name of the vendor"
            },
            "line_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "quantity": {"type": "number"},
                        "price": {"type": "number"}
                    }
                },
                "description": "List of line items"
            }
        },
        strict=True
    )
    
    # Define schema for document summary
    summary_schema = create_json_schema_format(
        name="document_summary",
        description="Generate a summary of the document",
        properties={
            "document_type": {
                "type": "string",
                "description": "Type of document (invoice, receipt, contract, etc.)"
            },
            "summary": {
                "type": "string",
                "description": "Brief summary of the document content"
            },
            "key_entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key entities mentioned in the document"
            }
        }
    )
    
    try:
        result = run_mistral_ocr(
            document_input=document_url,
            api_key=api_key,
            bbox_annotation_format=invoice_schema,
            document_annotation_format=summary_schema
        )
        
        # Access structured data
        if "document_annotation" in result:
            doc_summary = json.loads(result["document_annotation"])
            print(f"Document Type: {doc_summary.get('document_type')}")
            print(f"Summary: {doc_summary.get('summary')}")
            
        # Access page-level structured data
        for page in result.get('pages', []):
            for image in page.get('images', []):
                if image.get('image_annotation'):
                    invoice_data = json.loads(image['image_annotation'])
                    print(f"\nExtracted Invoice Data:")
                    print(f"  Invoice #: {invoice_data.get('invoice_number')}")
                    print(f"  Date: {invoice_data.get('date')}")
                    print(f"  Total: ${invoice_data.get('total_amount')}")
                    
    except Exception as e:
        print(f"Error: {e}")


def example_command_line_equivalent():
    """Show equivalent command-line usage"""
    print("\n=== Command Line Usage Examples ===")
    
    examples = [
        "# Basic URL processing",
        "python mistral_ocr.py --url https://example.com/document.pdf",
        "",
        "# Local file with specific pages",
        "python mistral_ocr.py --file invoice.pdf --pages 0,1 --output invoice_ocr.json",
        "",
        "# With custom model and image extraction",
        "python mistral_ocr.py --file report.pdf --model pixtral-12b-2409 --include-images",
        "",
        "# Using API key from command line",
        "python mistral_ocr.py --url https://example.com/doc.pdf --api-key YOUR_KEY",
        "",
        "# Or set environment variable",
        "export MISTRAL_API_KEY='your-key-here'",
        "python mistral_ocr.py --file document.pdf"
    ]
    
    for line in examples:
        print(line)


if __name__ == "__main__":
    print("Mistral OCR API Examples")
    print("=" * 50)
    
    # Uncomment the examples you want to run:
    
    # example_basic_url_ocr()
    # example_local_file_with_options()
    # example_structured_extraction()
    example_command_line_equivalent()