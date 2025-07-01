#!/usr/bin/env python3
"""
Example usage of the numero_source pipeline.

This script demonstrates how to use the numero_source package to process
book volumes containing cover, intro, and table of contents files.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to the path so we can import numero_source
sys.path.insert(0, str(Path(__file__).parent))

from numero_source import BookProcessor

def main():
    """Example usage of the BookProcessor."""
    
    # Check for API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable is required")
        print("Set it with: export MISTRAL_API_KEY='your-api-key'")
        return 1
    
    print("numero_source Pipeline Example")
    print("=" * 40)
    
    # Initialize the processor
    processor = BookProcessor(api_key=api_key, output_base_dir="numero_results")
    
    # Example 1: Process a single volume
    print("\n1. Processing a single volume:")
    sample_volume = "../volumes/3-10-goren-a"  # Adjust path as needed
    
    if Path(sample_volume).exists():
        try:
            print(f"Processing: {sample_volume}")
            result = processor.process_volume(sample_volume)
            
            print(f"✓ Successfully processed: {result.folder_name}")
            print(f"  - Cover: {'Found' if result.cover.original else 'Not found'}")
            print(f"  - Intro: {len(result.intro)} characters")
            print(f"  - ToC: {len(result.toc.entries)} entries")
            
        except Exception as e:
            print(f"✗ Error processing volume: {e}")
    else:
        print(f"✗ Sample volume not found: {sample_volume}")
    
    # Example 2: Process multiple volumes (uncomment to use)
    print("\n2. To process multiple volumes:")
    print("   results = processor.process_multiple_volumes('../volumes')")
    print("   # This would process all volumes in the volumes folder")
    
    print("\n3. Output files:")
    print("   Results are saved to the 'numero_results' directory")
    print("   Each volume gets its own subfolder with:")
    print("   - {volume_name}_processed.json (main result)")
    print("   - thumbnails/ (generated cover thumbnails)")
    
    return 0

if __name__ == "__main__":
    exit(main())