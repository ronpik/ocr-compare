from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import replace, is_dataclass
import time
from pathlib import Path

from globalog import LOG

from ocrtool.canonical_ocr.ocr_schema import OcrResult, Document
from ocrtool.page_limit.page_limit_handler import PageLimitHandler
from ocrtool.page_limit.limits import OcrExecutorType, get_page_limit
from ocrtool.page_limit.page_count import is_pdf, count_pdf_pages, split_pdf_to_segments
from ocrtool.canonical_ocr.ocr_schema import OcrResultSummary


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
        start_time = time.time()
        if is_pdf(image_data) and self.page_limit is not None:
            num_pages = count_pdf_pages(image_data)
            LOG.info(f"Detected PDF with {num_pages} pages. Page limit for this executor: {self.page_limit}.")
            if num_pages > self.page_limit:
                LOG.info(f"Splitting PDF: {num_pages} pages exceeds limit of {self.page_limit}. Splitting will be applied.")
                segments = split_pdf_to_segments(image_data, self.page_limit)
                LOG.info(f"Splitting complete: PDF split into {len(segments)} segments of up to {self.page_limit} pages each.")
                super_execute_ocr = super().execute_ocr
                results: List[OcrResult] = []
                for idx, seg in enumerate(segments):
                    LOG.info(f"Processing segment {idx+1}/{len(segments)}...")
                    seg_start = time.time()
                    seg_original_result =self.execute_ocr_original(seg, **kwargs)
                    seg_result = self.convert_to_canonical(seg_original_result)
                    seg_time = time.time() - seg_start

                    seg_summary = OcrResultSummary.from_ocr_result(seg_result)
                    LOG.info(
                        f"Segment {idx+1} processed in {seg_time:.2f}s: "
                        f"pages={seg_summary.num_pages}, blocks={seg_summary.num_blocks}, "
                        f"tables={seg_summary.num_tables}, total_length={seg_summary.total_length}"
                    )
                    results.append(seg_result)
                combined = self._combine_ocr_results(results)
                total_time = time.time() - start_time
                summary = OcrResultSummary.from_ocr_result(combined)
                LOG.info(
                    f"Combined result: pages={summary.num_pages}, blocks={summary.num_blocks}, "
                    f"tables={summary.num_tables}, total_length={summary.total_length}, "
                    f"total_time={total_time:.2f}s"
                )
                return combined
        try:
            LOG.info("Starting OCR processing...")
            native_result = self.execute_ocr_original(image_data, **kwargs)
            LOG.info("OCR processing complete. Converting to canonical format...")
            result = self.convert_to_canonical(native_result)
            elapsed = time.time() - start_time
            summary = OcrResultSummary.from_ocr_result(result)
            LOG.info(
                f"OCR complete: pages={summary.num_pages}, blocks={summary.num_blocks}, "
                f"tables={summary.num_tables}, symbols={summary.total_length}, "
                f"elapsed={elapsed:.2f}s"
            )
            return result
        except Exception as exc:
            LOG.error("Error during OCR processing", exc_info=exc)
            if self.handle_page_limit and self._page_limit_handler and self._page_limit_handler.is_page_limit_error(exc):
                return self._page_limit_handler.handle(exc, image_data, self, **kwargs)
            raise

    @staticmethod
    def _combine_ocr_results(results: list[OcrResult]) -> OcrResult:
        """
        Combine multiple OcrResult objects by concatenating their pages, renumbering pages and updating element_path for all layout elements using dataclasses.replace.
        """
        if not results:
            return OcrResult(document=Document(pages=[]))
        all_pages = []
        for result in results:
            all_pages.extend(result.document.pages)
        renumbered_pages = ExternalOcrExecutor._renumber_and_repath_pages(all_pages)
        base = results[0]
        base.document.pages = renumbered_pages
        return base

    @staticmethod
    def _renumber_and_repath_pages(pages: list) -> list:
        """
        Renumber pages and update element_path for all layout elements recursively using dataclasses.replace.
        """
        def update_element_path(element, new_page_no: int):
            if not hasattr(element, 'element_path') or element.element_path is None:
                return element
            
            parts = list(element.element_path.parts)
            if len(parts) > 1:
                parts[1] = str(new_page_no)
                new_path = Path(*parts)
                element = replace(element, element_path=new_path)
            
            # Recursively update children fields
            for field_name, field_def in getattr(element, '__dataclass_fields__', {}).items():
                value = getattr(element, field_name)
                if isinstance(value, list):
                    new_list = [update_element_path(child, new_page_no) for child in value]
                    element = replace(element, **{field_name: new_list})
                elif is_dataclass(value) and hasattr(value, 'element_path'):
                    element = replace(element, **{field_name: update_element_path(value, new_page_no)})
            
            return element
        
        new_pages = []
        for new_page_no, page in enumerate(pages, 1):
            page = replace(page, page_no=new_page_no)
            page = update_element_path(page, new_page_no)
            new_pages.append(page)
        return new_pages