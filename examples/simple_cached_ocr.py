#!/usr/bin/env python3
"""
Simplified example of using cached OCR functionality.
"""

import argparse
from pathlib import Path

from ocrtool.cached_ocr import get_cached_ocr_engine


def main():
    parser = argparse.ArgumentParser(description='Simple OCR with caching example')
    parser.add_argument('image_path', help='Path to the image file to process')
    parser.add_argument(
        '--engine', 
        choices=['tesseract', 'gdai'], 
        default='tesseract',
        help='OCR engine to use (default: tesseract)'
    )
    parser.add_argument(
        '--storage-config', 
        help='Path to the storage configuration file'
    )
    args = parser.parse_args()
    
    # Read the image
    with open(args.image_path, 'rb') as f:
        image_data = f.read()
    
    # Get a cached OCR engine
    engine = get_cached_ocr_engine(
        executor=args.engine,
        storage_path=args.storage_config,
        cache_prefix=f"{args.engine}_cache"
    )
    
    # Process the image
    print(f"Processing image: {args.image_path}")
    
    # First run
    print("First run (may use cache if available):")
    result = engine.execute_ocr(image_data)
    print(f"- Text (excerpt): {result.document.text()[:150]}...")
    
    # Second run (should be from cache)
    print("\nSecond run (should be from cache):")
    result2 = engine.execute_ocr(image_data)
    print(f"- Text (excerpt): {result2.document.text()[:150]}...")
    
    # Show engine info
    print("\nEngine information:")
    for key, value in engine.get_implementation_info().items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()