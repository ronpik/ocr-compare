#!/usr/bin/env python3
"""
Example of using CachedOcrExecutor with different storage backends.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, Dict, Any

from ocrtool.ocr_impls.tesseract.tesseract_executor import TesseractOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_executor import GoogleDocumentAIOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_config import GdaiConfig
from ocrtool.ocr_impls.cached_ocr_executor import CachedOcrExecutor
from ocrtool.storage.config import StorageConfig
from ocrtool.storage.handlers.local_handler import LocalStorageHandler
from ocrtool.storage.handlers.gcs_handler import GCSHandler


def read_image(image_path: str) -> bytes:
    """
    Read image file into bytes.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bytes: Image file contents
    """
    with open(image_path, 'rb') as f:
        return f.read()


def create_example_config(output_path: str, config_type: str):
    """
    Create an example configuration file.
    
    Args:
        output_path: Path to write the example config
        config_type: Type of configuration to create ("gdai" or "tesseract")
    """
    if config_type == "gdai":
        example_config = {
            "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
            "location": "us",
            "timeout": 300,
            "service_account_file": "/path/to/your/service-account.json"
        }
    else:  # tesseract
        example_config = {
            "lang": "eng",
            "config": "",
            "timeout": 30
        }
    
    with open(output_path, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"Example {config_type} configuration written to {output_path}")
    print("Please edit this file with your actual settings.")


def main():
    parser = argparse.ArgumentParser(description='OCR with caching example')
    parser.add_argument('image_path', help='Path to the image file to process', nargs='?')
    parser.add_argument(
        '--storage-config', 
        help='Path to the storage configuration file'
    )
    parser.add_argument(
        '--ocr-config',
        help='Path to the OCR engine configuration file'
    )
    parser.add_argument(
        '--create-config',
        help='Create an example configuration file at the specified path and exit'
    )
    parser.add_argument(
        '--config-type',
        choices=['gdai', 'tesseract'],
        default='tesseract',
        help='Type of configuration to create with --create-config'
    )
    parser.add_argument(
        '--cache-prefix', 
        default='ocr_cache',
        help='Prefix for cache storage (default: ocr_cache)'
    )
    parser.add_argument(
        '--method', 
        choices=['tesseract', 'gdai'], 
        default='tesseract',
        help='OCR method to use (default: tesseract)'
    )
    parser.add_argument(
        '--storage-type',
        choices=['local', 'gcs'],
        default='local',
        help='Storage type to use when not using config file (default: local)'
    )
    args = parser.parse_args()
    
    # If create-config option was used, create the example config and exit
    if args.create_config:
        create_example_config(args.create_config, args.config_type)
        return 0
    
    # Validate image path
    if not args.image_path:
        print("Error: Image path is required unless --create-config is specified.")
        return 1
        
    # Load the image
    image_data = read_image(args.image_path)
    
    # Initialize storage handler
    storage_handler = None
    if args.storage_config:
        # Use config file
        storage_handler = StorageConfig.create_handler_from_file(args.storage_config)
    else:
        # Use command line parameters
        if args.storage_type == 'local':
            # Create a cache directory if it doesn't exist
            cache_dir = Path.home() / '.ocr_cache'
            os.makedirs(cache_dir, exist_ok=True)
            storage_handler = LocalStorageHandler(cache_dir)
        elif args.storage_type == 'gcs':
            # This is a mock example - in real usage, you'd need to provide actual credentials
            storage_config = {
                'bucket': 'my-ocr-cache-bucket',
                'credentials_path': 'path/to/credentials.json'
            }
            # Note: This would fail without actual credentials, just for example
            # storage_handler = GCSHandler(storage_config)
            print("GCS storage selected, but no actual credentials provided.")
            print("This is a mock example. Using local storage instead.")
            # Fallback to local storage for the example
            cache_dir = Path.home() / '.ocr_cache'
            os.makedirs(cache_dir, exist_ok=True)
            storage_handler = LocalStorageHandler(cache_dir)
    
    # Load OCR configuration if provided
    ocr_config: Dict[str, Any] = {}
    if args.ocr_config:
        config_path = Path(args.ocr_config)
        if not config_path.exists():
            print(f"Error: OCR configuration file not found: {config_path}")
            return 1
            
        with open(config_path, 'r') as f:
            ocr_config = json.load(f)
    
    # Initialize the appropriate OCR executor
    if args.method == 'tesseract':
        base_executor = TesseractOcrExecutor(config=ocr_config)
    elif args.method == 'gdai':
        if args.ocr_config:
            # Use the ocr_config loaded from file
            gdai_config = GdaiConfig.from_file(args.ocr_config)
            base_executor = GoogleDocumentAIOcrExecutor(config=gdai_config.to_dict())
        else:
            # This is a mock example - in real usage, you'd need to provide actual processor details
            print("GDAI selected, but no configuration file provided.")
            print("This is a mock example. Using Tesseract instead.")
            # Fallback to Tesseract for the example
            base_executor = TesseractOcrExecutor()
    
    # Create the cached executor
    cached_executor = CachedOcrExecutor(
        base_executor,
        storage_handler,
        args.cache_prefix
    )
    
    # Execute OCR
    print(f"Processing image: {args.image_path}")
    print(f"Using cache prefix: {args.cache_prefix}")
    print(f"Using OCR method: {args.method}")
    
    # First execution (might be from cache or fresh)
    result = cached_executor.execute_ocr(image_data)
    print(f"OCR Result (first execution):")
    print(f"- Number of pages: {len(result.document.pages)}")
    print(f"- First page dimensions: {result.document.pages[0].width}x{result.document.pages[0].height}")
    print(f"- Number of blocks: {len(list(result.blocks()))}")
    print(f"- Text sample: {result.document.text()[:100]}...")
    
    # Second execution (should be from cache)
    result2 = cached_executor.execute_ocr(image_data)
    print("\nOCR Result (second execution, should be cached):")
    print(f"- Number of pages: {len(result2.document.pages)}")
    
    # Native result access
    native_result = cached_executor.get_native_result()
    print("\nNative result type:", type(native_result))
    
    # Implementation information
    impl_info = cached_executor.get_implementation_info()
    print("\nImplementation info:")
    for key, value in impl_info.items():
        print(f"- {key}: {value}")
    
    return 0


if __name__ == "__main__":
    main()