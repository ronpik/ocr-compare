#!/usr/bin/env python3
"""
Example script demonstrating how to use Google Document AI for OCR with ocrtool.

This script shows how to properly set up authentication and use Google Document AI
as an OCR engine within the ocrtool framework.

Prerequisites:
1. Set up a Document AI processor in Google Cloud Console
2. Have appropriate authentication credentials
   - Either set GOOGLE_APPLICATION_CREDENTIALS environment variable
   - Or provide service account credentials in the config file
3. Install required packages:
   pip install google-cloud-documentai python-magic
"""

import argparse
import json
import os
from pathlib import Path

from ocrtool import execute_ocr
from ocrtool.ocr_impls.gdai import GoogleDocumentAIOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_config import GdaiConfig


def create_example_config(output_path: str):
    """
    Create an example configuration file.
    
    Args:
        output_path: Path to write the example config
    """
    example_config = {
        "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
        "location": "us",
        "timeout": 300,
        "service_account_file": "/path/to/your/service-account.json"
    }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example configuration written to {output_path}")
    print("Please edit this file with your actual Google Document AI settings.")


def main():
    parser = argparse.ArgumentParser(description="Process a document with Google Document AI OCR")
    parser.add_argument("document_path", help="Path to the document file to process", nargs="?")
    parser.add_argument("--config", help="Path to Google Document AI configuration file")
    parser.add_argument("--create-config", help="Create an example configuration file at the specified path and exit")
    parser.add_argument("--output", help="Output file path for OCR results (JSON)", default=None)
    
    args = parser.parse_args()
    
    # If create-config option was used, create the example config and exit
    if args.create_config:
        create_example_config(args.create_config)
        return 0
    
    # Validate document path
    if not args.document_path:
        print("Error: Document path is required unless --create-config is specified.")
        return 1
    
    document_path = Path(args.document_path)
    if not document_path.exists():
        print(f"Error: Document file not found: {document_path}")
        return 1
        
    with open(document_path, "rb") as f:
        document_data = f.read()
    
    # Load config from file or provide example usage message
    if not args.config:
        print("Error: Configuration file is required. Use --config to specify.")
        print("You can create an example configuration with --create-config.")
        return 1
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    try:
        # Load the GDAI configuration
        gdai_config = GdaiConfig.from_file(str(config_path))
        config_dict = gdai_config
        
        # Verify processor name
        processor_name = gdai_config.processor_name
        if not processor_name or "your-project-id" in processor_name:
            print("Error: Invalid processor_name in configuration file.")
            print("Please update the configuration with your actual Google Document AI settings.")
            return 1
        
        # Verify service account if specified
        if gdai_config.service_account_file:
            service_account_path = Path(gdai_config.service_account_file)
            if not service_account_path.exists():
                print(f"Error: Service account file not found: {service_account_path}")
                print("Please update the configuration with a valid service account file path.")
                return 1
        
        print(f"Processing document: {document_path}")
        print(f"Using processor: {processor_name}")
        
        # Method 1: Using the factory pattern
        print("\nMethod 1: Using factory pattern")
        result1 = execute_ocr(document_data, engine="gdai", engine_config=config_dict.to_dict())
        
        # Method 2: Creating executor instance directly
        print("\nMethod 2: Creating executor instance directly")
        gdai_executor = GoogleDocumentAIOcrExecutor(config=config_dict)
        result2 = gdai_executor.execute_ocr(document_data)
        
        # Display results
        print("\nDocument Analysis Results:")
        print(f"  - Pages: {len(result1.document.pages)}")
        print(f"  - Blocks: {sum(len(page.blocks) for page in result1.document.pages)}")
        
        # Print first few words of text
        text = result1.document.text()
        preview = text[:100] + "..." if len(text) > 100 else text
        print(f"\nText preview:\n{preview}")
        
        # Access original format
        native_result = gdai_executor.get_native_result()
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            
            # Create a serializable version of the results
            output_data = {
                "text": text,
                "page_count": len(result1.document.pages),
                "structure": {
                    "pages": [
                        {
                            "page_number": page.page_no,
                            "blocks": [
                                {
                                    "block_type": block.blockType,
                                    "text": block.text()
                                }
                                for block in page.blocks
                            ]
                        }
                        for page in result1.document.pages
                    ]
                }
            }
            
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            
            print(f"\nResults saved to {output_path}")
        
    except Exception as e:
        raise

    return 0


if __name__ == "__main__":
    exit(main())