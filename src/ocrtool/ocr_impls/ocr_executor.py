from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging
import time

from ocrtool.canonical_ocr.ocr_schema import OcrResult, Document
from ocrtool.page_limit.page_limit_handler import PageLimitHandler
from ocrtool.page_limit.limits import OcrExecutorType, get_page_limit
from ocrtool.page_limit.page_count import is_pdf, count_pdf_pages, split_pdf_to_segments


class OcrExecutor(ABC):
    """
    Abstract base class for OCR executors implementing the Strategy pattern.
    Each concrete implementation should be able to execute OCR on an image
    and return results in both canonical and native formats.
    """
    
    @abstractmethod
    def execute_ocr(self, image_data: bytes, **kwargs) -> OcrResult:
        """
        Execute OCR on the provided image data and return results in canonical format.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            OcrResult: Results in canonical schema format
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_native_result(self) -> Any:
        """
        Return the native result from the most recent OCR execution.
        
        Returns:
            Any: The OCR result in the implementation's native format
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_implementation_info(self) -> Dict[str, str]:
        """
        Return information about this OCR implementation.
        
        Returns:
            Dict[str, str]: Information about the implementation including:
                - name: Name of the OCR implementation
                - version: Version of the OCR implementation
                - description: Brief description of the implementation
        """
        raise NotImplementedError()


class ExternalOcrExecutor(OcrExecutor):
    """
    Abstract base class for OCR executors that use external engines.
    Separates the execution of OCR and conversion to canonical format.
    Adds support for handling page limit errors and PDF splitting.
    """
    def __init__(self, handle_page_limit: bool = True) -> None:
        """
        Initialize the ExternalOcrExecutor.

        Args:
            handle_page_limit: Whether to handle page limit errors automatically (default: True)
        """
        self.handle_page_limit = handle_page_limit
        self._page_limit_handler = PageLimitHandler() if handle_page_limit else None
    
    @property
    @abstractmethod
    def type(self) -> OcrExecutorType:
        """
        Return the executor type for page limit and configuration purposes.
        """
        raise NotImplementedError()

    @property
    def page_limit(self) -> int | None:
        """
        Return the page limit for this executor, or None if unlimited.
        """
        return get_page_limit(self.type)
    
    @abstractmethod
    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR on the provided image data and return results in the 
        native format of the underlying OCR engine, as a JSON-serializable dictionary.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Dict[str, Any]: Results in native format of the OCR engine, 
                           formatted as a JSON-serializable dictionary
        """
        raise NotImplementedError()
    
    @abstractmethod
    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        """
        Convert the native OCR result to the canonical schema format.
        
        Args:
            native_result: OCR result in the implementation's native format
                          (JSON-serializable dictionary)
            
        Returns:
            OcrResult: Results in canonical schema format
        """
        raise NotImplementedError()
    
    def execute_ocr(self, image_data: bytes, **kwargs) -> OcrResult:
        """
        Execute OCR and convert the results to canonical format. Handles page limit errors, PDF splitting, and result combining.

        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters

        Returns:
            OcrResult: Results in canonical schema format
        """
        logger = logging.getLogger(__name__)
        if is_pdf(image_data) and self.page_limit is not None:
            num_pages = count_pdf_pages(image_data)
            logger.info(f"Detected PDF with {num_pages} pages. Page limit for {self.type.name}: {self.page_limit}.")
            if num_pages > self.page_limit:
                logger.warning(f"PDF exceeds page limit ({num_pages} > {self.page_limit}). Splitting into segments.")
                segments = split_pdf_to_segments(image_data, self.page_limit)
                logger.info(f"Split PDF into {len(segments)} segments of up to {self.page_limit} pages each.")
                super_execute_ocr = super().execute_ocr
                start_time = time.time()
                results = [super_execute_ocr(seg, **kwargs) for seg in segments]
                elapsed = time.time() - start_time
                combined = self._combine_ocr_results(results)
                self._log_ocr_result_summary(combined, elapsed, num_pages)
                return combined
        start_time = time.time()
        try:
            native_result = self.execute_ocr_original(image_data, **kwargs)
        except Exception as exc:
            logger.exception("Exception during OCR execution.")
            if self.handle_page_limit and self._page_limit_handler and self._page_limit_handler.is_page_limit_error(exc):
                return self._page_limit_handler.handle(exc, image_data, self, **kwargs)
            raise
        elapsed = time.time() - start_time
        canonical = self.convert_to_canonical(native_result)
        self._log_ocr_result_summary(canonical, elapsed, 1)
        return canonical

    @staticmethod
    def _combine_ocr_results(results: list[OcrResult]) -> OcrResult:
        """
        Combine multiple OcrResult objects by concatenating their pages.
        """
        if not results:
            return OcrResult(document=Document(pages=[]))
        base = results[0]
        all_pages = []
        for result in results:
            all_pages.extend(result.document.pages)
        base.document.pages = all_pages
        return base

    @staticmethod
    def _log_ocr_result_summary(result: OcrResult, elapsed: float, num_pages: int) -> None:
        """
        Log a summary of the OcrResult including number of pages, blocks, tables, symbols, and processing time.
        """
        logger = logging.getLogger(__name__)
        doc = result.document
        n_pages = len(getattr(doc, 'pages', []))
        n_blocks = sum(len(getattr(page, 'blocks', [])) for page in getattr(doc, 'pages', []))
        n_tables = sum(len(getattr(page, 'tables', [])) for page in getattr(doc, 'pages', []) if hasattr(page, 'tables'))
        n_symbols = 0
        for page in getattr(doc, 'pages', []):
            for block in getattr(page, 'blocks', []):
                for para in getattr(block, 'paragraphs', []):
                    for line in getattr(para, 'lines', []):
                        for word in getattr(line, 'words', []):
                            n_symbols += len(getattr(word, 'symbols', []))
        logger.info(
            f"OCR processed {num_pages} input pages in {elapsed:.2f}s. "
            f"Result: {n_pages} canonical pages, {n_blocks} blocks, {n_tables} tables, {n_symbols} symbols."
        )