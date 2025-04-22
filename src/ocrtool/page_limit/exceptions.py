class PageLimitExceededError(Exception):
    """
    Exception raised when a document exceeds the allowed page limit for OCR processing.
    """
    def __init__(self, message: str) -> None:
        super().__init__(message) 