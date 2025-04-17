from ocrtool.ocr_impls.gdai.gdai_config import GdaiConfig
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory
from ocrtool.ocr_impls.gdai.gdai_executor import GoogleDocumentAIOcrExecutor, GoogleDocumentAILayoutExecutor

# Register the Google Document AI OCR implementation with the factory
OcrExecutorFactory.register("gdai-ocr", GoogleDocumentAIOcrExecutor)
OcrExecutorFactory.register("gdai-layout", GoogleDocumentAILayoutExecutor)