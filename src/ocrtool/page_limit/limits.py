from enum import Enum, auto
from typing import Dict, Optional

class OcrExecutorType(Enum):
    DOCUMENT_AI_ONLINE = auto()
    DOCUMENT_AI_IMAGELESS = auto()
    DOCUMENT_AI_LAYOUT_PARSER = auto()
    DOCUMENT_AI_BATCH_CUSTOM_EXTRACTOR = auto()
    DOCUMENT_AI_BATCH_FORM_PARSER = auto()
    DOCUMENT_AI_BATCH_LAYOUT_PARSER = auto()
    TESSERACT = auto()
    # Add more as needed

PAGE_LIMITS: Dict[OcrExecutorType, Optional[int]] = {
    OcrExecutorType.DOCUMENT_AI_ONLINE: 15,
    OcrExecutorType.DOCUMENT_AI_LAYOUT_PARSER: 15,
    OcrExecutorType.DOCUMENT_AI_IMAGELESS: 30,
    OcrExecutorType.DOCUMENT_AI_BATCH_CUSTOM_EXTRACTOR: 200,
    OcrExecutorType.DOCUMENT_AI_BATCH_FORM_PARSER: 100,
    OcrExecutorType.DOCUMENT_AI_BATCH_LAYOUT_PARSER: 500,
    OcrExecutorType.TESSERACT: None,  # No limit
}

def has_predefined_page_limit(executor_type: OcrExecutorType) -> bool:
    """
    Check if the given executor type has a predefined page limit.
    """
    return executor_type in PAGE_LIMITS and PAGE_LIMITS[executor_type] is not None

def get_page_limit(executor_type: OcrExecutorType) -> Optional[int]:
    """
    Get the page limit for the given executor type, or None if unlimited.
    """
    return PAGE_LIMITS.get(executor_type) 