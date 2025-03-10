#!/usr/bin/env python3
"""
Example script that compares OCR results from multiple engines (Tesseract and Google Document AI).

This script shows how to:
1. Set up and configure multiple OCR engines
2. Process the same document with each engine
3. Compare the results to identify differences in text extraction, confidence, etc.

Prerequisites:
1. For Tesseract: install pytesseract and tesseract-ocr
2. For Google Document AI: 
   - Set up a processor in Google Cloud Console
   - Configure authentication (service account or application default credentials)
   - Install google-cloud-documentai
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocrtool import create_ocr_engine, compare_ocr_results
from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.ocr_impls.tesseract import TesseractOcrExecutor
from ocrtool.ocr_impls.gdai import GoogleDocumentAIOcrExecutor


def configure_engines(args):
    """Configure OCR engines based on command line arguments."""
    engines = {}
    
    # Configure Tesseract
    if args.use_tesseract:
        tesseract_config = {
            "lang": args.tesseract_lang
        }
        engines["tesseract"] = (TesseractOcrExecutor(tesseract_config), {})
    
    # Configure Google Document AI
    if args.use_gdai:
        if not (args.processor_id and args.project_id):
            print("Error: --processor-id and --project-id are required for Google Document AI")
            return None
        
        processor_name = f"projects/{args.project_id}/locations/{args.location}/processors/{args.processor_id}"
        gdai_config = {
            "processor_name": processor_name,
            "location": args.location,
        }
        
        if args.service_account:
            service_account_path = Path(args.service_account)
            if not service_account_path.exists():
                print(f"Error: Service account file not found: {service_account_path}")
                return None
            
            gdai_config["service_account_file"] = str(service_account_path)
        
        engines["gdai"] = (GoogleDocumentAIOcrExecutor(gdai_config), {})
    
    if not engines:
        print("Error: At least one OCR engine must be enabled")
        return None
    
    return engines


def visualize_results(image_path, results, output_dir):
    """Create visualizations of OCR results for each engine."""
    # Load the original image
    image = Image.open(image_path)
    
    for engine_name, result in results.items():
        # Create a copy of the original image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("Arial", 14)
        except IOError:
            font = ImageFont.load_default()
        
        # Draw bounding boxes for each block
        for page in result.document.pages:
            for block in page.blocks:
                if block.boundingBox:
                    # Get coordinates (assuming normalized coordinates 0-1)
                    left = block.boundingBox.left or 0
                    top = block.boundingBox.top or 0
                    width = block.boundingBox.width or 0
                    height = block.boundingBox.height or 0
                    
                    # If coordinates are normalized (0-1), convert to pixels
                    if left < 1 and top < 1:
                        left *= image.width
                        top *= image.height
                        width *= image.width
                        height *= image.height
                    
                    # Draw rectangle
                    draw.rectangle(
                        [(left, top), (left + width, top + height)],
                        outline=(255, 0, 0),
                        width=2
                    )
                    
                    # Draw block text
                    text = block.text()[:20] + "..." if len(block.text()) > 20 else block.text()
                    draw.text((left, top - 15), text, fill=(255, 0, 0), font=font)
        
        # Save the annotated image
        output_path = Path(output_dir) / f"{engine_name}_annotated.png"
        annotated_image.save(output_path)
        print(f"Saved annotated image for {engine_name} to {output_path}")


def analyze_text_differences(results):
    """Analyze and visualize text differences between OCR engines."""
    # Get extracted text from each engine
    texts = {engine: result.document.text() for engine, result in results.items()}
    
    # Calculate basic text statistics
    stats = {
        engine: {
            "char_count": len(text),
            "word_count": len(text.split()),
            "line_count": len(text.split("\n")),
        }
        for engine, text in texts.items()
    }
    
    # Calculate Jaccard similarity between each pair of engines
    engines = list(texts.keys())
    similarities = {}
    
    for i, engine1 in enumerate(engines):
        for engine2 in engines[i+1:]:
            # Split into words
            words1 = set(texts[engine1].lower().split())
            words2 = set(texts[engine2].lower().split())
            
            # Calculate Jaccard similarity (intersection over union)
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = intersection / union if union > 0 else 0
            
            similarities[f"{engine1}_vs_{engine2}"] = similarity
    
    return {"stats": stats, "similarities": similarities}


def main():
    parser = argparse.ArgumentParser(description="Compare OCR results from multiple engines")
    parser.add_argument("document_path", help="Path to the document file to process")
    parser.add_argument("--output", help="Output directory for results and visualizations", default="ocr_comparison_results")
    
    # Tesseract options
    parser.add_argument("--use-tesseract", action="store_true", help="Use Tesseract OCR")
    parser.add_argument("--tesseract-lang", help="Tesseract language", default="eng")
    
    # Google Document AI options
    parser.add_argument("--use-gdai", action="store_true", help="Use Google Document AI")
    parser.add_argument("--processor-id", help="Google Document AI processor ID")
    parser.add_argument("--project-id", help="Google Cloud project ID")
    parser.add_argument("--location", help="API location (e.g., 'us', 'eu')", default="us")
    parser.add_argument("--service-account", help="Path to service account JSON file")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true", help="Create visualizations of OCR results")
    
    args = parser.parse_args()
    
    # Ensure at least one engine is selected
    if not (args.use_tesseract or args.use_gdai):
        print("Error: At least one OCR engine must be enabled")
        return 1
    
    # Load the document data
    document_path = Path(args.document_path)
    if not document_path.exists():
        print(f"Error: Document file not found: {document_path}")
        return 1
    
    with open(document_path, "rb") as f:
        document_data = f.read()
    
    # Configure OCR engines
    engines = configure_engines(args)
    if not engines:
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Process document with each engine
    results = {}
    print(f"Processing document with {len(engines)} OCR engines...")
    
    for engine_name, (executor, kwargs) in engines.items():
        print(f"  - Running {engine_name}...")
        try:
            result = executor.execute_ocr(document_data, **kwargs)
            results[engine_name] = result
            print(f"    Completed with {len(result.document.pages)} pages detected")
        except Exception as e:
            print(f"    Error processing with {engine_name}: {e}")
    
    if len(results) < 1:
        print("No successful OCR results to analyze.")
        return 1
    
    # Compare results if we have multiple engines
    if len(results) > 1:
        print("\nComparing OCR results...")
        comparison = compare_ocr_results(results)
        
        # Analyze text differences
        text_analysis = analyze_text_differences(results)
        
        # Display basic comparison information
        print("\nText extraction statistics:")
        for engine, stats in text_analysis["stats"].items():
            print(f"  - {engine}:")
            print(f"      Characters: {stats['char_count']}")
            print(f"      Words: {stats['word_count']}")
            print(f"      Lines: {stats['line_count']}")
        
        print("\nText similarity between engines:")
        for pair, similarity in text_analysis["similarities"].items():
            print(f"  - {pair}: {similarity:.2f}")
        
        print("\nConfidence scores:")
        for engine, score in comparison["confidence_scores"].items():
            print(f"  - {engine}: {score:.2f}")
        
        # Save comparison results
        comparison_path = output_dir / "comparison_results.json"
        with open(comparison_path, "w") as f:
            # Combine comparison results with text analysis
            output_data = {
                "comparison": comparison,
                "text_analysis": text_analysis,
            }
            
            # Add limited text samples
            output_data["text_samples"] = {
                engine: text[:1000] + "..." if len(text) > 1000 else text
                for engine, text in comparison["text_extraction"].items()
            }
            
            json.dump(output_data, f, indent=2)
        
        print(f"\nComparison results saved to {comparison_path}")
    
    # Create visualizations if requested
    if args.visualize and document_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'):
        print("\nGenerating visualizations...")
        visualize_results(document_path, results, output_dir)
    
    return 0


if __name__ == "__main__":
    exit(main())