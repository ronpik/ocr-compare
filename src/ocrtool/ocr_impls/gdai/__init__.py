from ocrtool.ocr_impls.gdai.gdai_executor import GoogleDocumentAIOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_config import GdaiConfig
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory

# Register the Google Document AI OCR implementation with the factory
OcrExecutorFactory.register("gdai", GoogleDocumentAIOcrExecutor)