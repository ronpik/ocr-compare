#!/usr/bin/env python3
"""
Basic example demonstrating how to use ocrtool to process images with different OCR engines
and compare the results.
"""

import argparse
import json
from pathlib import Path

from ocrtool import (
    list_available_engines,
    execute_ocr,
    compare_ocr_results
)


def main():
    parser = argparse.ArgumentParser(description="Process an image with different OCR engines and compare results")
    parser.add_argument("image_path", help="Path to the image file to process")
    parser.add_argument("--engines", help="Comma-separated list of OCR engines to use", default=None)
    parser.add_argument("--output", help="Output file path for comparison results (JSON)", default=None)
    
    args = parser.parse_args()
    
    # Load the image data
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}")
        return 1
        
    with open(image_path, "rb") as f:
        image_data = f.read()
    
    # Get available engines
    available_engines = list_available_engines()
    if not available_engines:
        print("No OCR engines available. Please install at least one OCR implementation.")
        return 1
    
    print("Available OCR engines:")
    for name, info in available_engines.items():
        print(f"  - {name}: {info.get('description')} (version: {info.get('version')})")
    
    # Determine which engines to use
    engines_to_use = []
    if args.engines:
        requested_engines = [e.strip().lower() for e in args.engines.split(',')]
        engines_to_use = [e for e in requested_engines if e in available_engines]
        
        if not engines_to_use:
            print("None of the requested engines are available.")
            print(f"Available engines: {', '.join(available_engines.keys())}")
            return 1
    else:
        # Use all available engines
        engines_to_use = list(available_engines.keys())
    
    # Process the image with each OCR engine
    results = {}
    print(f"\nProcessing image with {len(engines_to_use)} OCR engines...")
    
    for engine_name in engines_to_use:
        print(f"  - Running {engine_name}...")
        try:
            result = execute_ocr(image_data, engine=engine_name)
            results[engine_name] = result
            print(f"    Completed with {len(result.document.pages)} pages detected")
        except Exception as e:
            print(f"    Error processing with {engine_name}: {e}")
    
    if len(results) < 2:
        print("Need at least two successful OCR results to perform comparison.")
        return 1
    
    # Compare the results
    print("\nComparing OCR results...")
    comparison = compare_ocr_results(results)
    
    # Display some basic comparison information
    print("\nText extraction:")
    for engine, text in comparison["text_extraction"].items():
        print(f"  - {engine}: {len(text)} characters")
    
    print("\nConfidence scores:")
    for engine, score in comparison["confidence_scores"].items():
        print(f"  - {engine}: {score:.2f}")
    
    # Save results if requested
    if args.output:
        output_path = Path(args.output)
        
        # Convert OcrResult objects to simple dict representation for JSON serialization
        serializable_results = {}
        for engine, result in results.items():
            serializable_results[engine] = {
                "text": result.document.text(),
                "page_count": len(result.document.pages),
                "block_count": sum(len(page.blocks) for page in result.document.pages)
            }
        
        output_data = {
            "comparison": comparison,
            "results": serializable_results
        }
        
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())