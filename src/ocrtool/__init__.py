"""
OCR Compare - A tool for comparing OCR results from different OCR engines
"""

__version__ = "0.1.1"

from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.ocr import (
    list_available_engines,
    execute_ocr,
    create_ocr_engine,
    compare_ocr_results
)
from ocrtool.ocr_impls.ocr_executor import OcrExecutor, ExternalOcrExecutor
from ocrtool.ocr_impls.cached_ocr_executor import CachedOcrExecutor
from ocrtool.cached_ocr import get_cached_executor, get_cached_ocr_engine