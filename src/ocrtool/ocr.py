from typing import Dict, List, Any, Optional

from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.ocr_impls.ocr_executor import OcrExecutor
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory


def list_available_engines() -> Dict[str, Dict[str, str]]:
    """
    List all available OCR engines.
    
    Returns:
        Dict[str, Dict[str, str]]: Dictionary of engine names and their metadata
    """
    return OcrExecutorFactory.list_available()


def create_ocr_engine(name: str, **config) -> OcrExecutor:
    """
    Create an OCR engine instance by name.
    
    Args:
        name: Name of the OCR engine to create
        **config: Engine-specific configuration parameters
        
    Returns:
        OcrExecutor: An instance of the specified OCR engine
        
    Raises:
        ValueError: If the specified engine is not available
    """
    return OcrExecutorFactory.create(name, **config)


def execute_ocr(image_data: bytes, engine: str = None, engine_config: Optional[Dict[str, Any]] = None,
                  **kwargs) -> OcrResult:
    """
    Execute OCR on an image with the specified engine.
    
    Args:
        image_data: Raw bytes of the image to process
        engine: Name of the OCR engine to use (default: first available)
        engine_config: Configuration for the OCR engine
        **kwargs: Additional parameters for OCR execution
        
    Returns:
        OcrResult: OCR results in canonical format
        
    Raises:
        ValueError: If no engines are available or the specified engine is not found
    """
    available_engines = list_available_engines()
    
    if not available_engines:
        raise ValueError("No OCR engines available. Please install at least one OCR implementation.")
    
    if engine is None:
        # Use the first available engine
        engine = next(iter(available_engines.keys()))
    
    config = engine_config or {}
    ocr_executor = create_ocr_engine(engine, **config)
    
    return ocr_executor.execute_ocr(image_data, **kwargs)


def compare_ocr_results(results: Dict[str, OcrResult]) -> Dict[str, Any]:
    """
    Compare OCR results from multiple engines.
    
    Args:
        results: Dictionary mapping engine names to their OcrResult
        
    Returns:
        Dict[str, Any]: Comparison results including:
            - text_similarity: Text similarity metrics between engines
            - layout_similarity: Layout similarity metrics between engines
            - confidence_comparison: Confidence score comparisons
            - word_level_comparison: Detailed word-level differences
    """
    # This is a placeholder for the comparison logic
    # In a real implementation, this would analyze text, layout, confidence scores, etc.
    
    # Basic text comparison
    extracted_texts = {engine: result.document.text() for engine, result in results.items()}
    
    # Basic confidence comparison
    confidence_scores = {}
    for engine, result in results.items():
        page_confidences = [page.confidence for page in result.document.pages]
        confidence_scores[engine] = sum(page_confidences) / len(page_confidences) if page_confidences else 0
    
    return {
        "text_extraction": extracted_texts,
        "confidence_scores": confidence_scores,
        # Additional comparison metrics would be added here
    }