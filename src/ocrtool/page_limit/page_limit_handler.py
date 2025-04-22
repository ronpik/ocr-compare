from typing import Any, TYPE_CHECKING

from globalog import LOG

from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.page_limit.exceptions import PageLimitExceededError

if TYPE_CHECKING:
    from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor

class PageLimitHandler:
    """
    Handles page limit errors for OCR executors.
    Detects page limit errors and provides a mechanism to handle them.
    """

    PAGE_LIMIT_ERROR_STRINGS = [
        "PAGE_LIMIT_EXCEEDED",
        "page limit",
        "pages in non-imageless mode exceed the limit",
        "Document pages in non-imageless mode exceed the limit"
    ]

    def is_page_limit_error(self, exc: Exception) -> bool:
        """
        Determine if the given exception is a page limit error.

        Args:
            exc: The exception to check.

        Returns:
            bool: True if the exception is a page limit error, False otherwise.
        """
        msg = str(exc)
        return any(s in msg for s in self.PAGE_LIMIT_ERROR_STRINGS)

    def handle(
        self,
        exc: Exception,
        image_data: bytes,
        executor: 'ExternalOcrExecutor',
        **kwargs: Any
    ) -> OcrResult:
        """
        Handle a page limit error. This method can be extended to implement custom logic,
        such as splitting the document or truncating pages.

        Args:
            exc: The page limit exception.
            image_data: The original image data.
            executor: The OCR executor instance.
            **kwargs: Additional arguments for OCR execution.

        Returns:
            OcrResult: The OCR result after handling the page limit error.
        """
        LOG.error(f"Page limit exceeded: {exc}")
        # Raise a custom exception for clarity
        raise PageLimitExceededError("Page limit exceeded and automatic handling is not yet implemented.") 