from ocrtool.ocr_impls.tesseract.tesseract_executor import TesseractOcrExecutor
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory

# Register the Tesseract OCR implementation with the factory
OcrExecutorFactory.register("tesseract", TesseractOcrExecutor)