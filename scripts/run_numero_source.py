#!/usr/bin/env python3
"""
Entry point script for running the numero_source pipeline.

This script can be run directly and handles the import path setup.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to Python path so we can import numero_source
scripts_dir = Path(__file__).parent
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Now we can import numero_source
from numero_source.pipeline import main

if __name__ == "__main__":
    exit(main())