from ocrtool.ocr_impls.ocr_executor import OcrExecutor
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory

# Import all OCR implementations to register them with the factory
try:
    from ocrtool.ocr_impls.tesseract import TesseractOcrExecutor
except ImportError:
    pass  # Tesseract not installed

try:
    from ocrtool.ocr_impls.gdai import GoogleCloudDocumentOCR
except ImportError:
    pass  # Google Document AI not installed