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
        Renumber pages and update element_path for all layout elements recursively.
        Uses the page_span fields in Block elements to determine the proper page numbering.
        Ensures continuous page numbering across combined documents.
        """
        if not pages:
            return []
            
        # First pass: create a mapping from old page numbers to new page numbers
        new_pages = []
        current_page_number = 1
        page_number_map = {}
        
        for page_idx, page in enumerate(pages):
            old_page_no = page.page_no
            page_number_map[old_page_no] = current_page_number
            
            # Update the page number
            page = replace(page, page_no=current_page_number)
            new_pages.append(page)
            
            # Find the maximum end_page in this page's blocks
            max_end_page = current_page_number
            for block in page.blocks:
                if hasattr(block, 'page_span') and block.page_span != (-1, -1):
                    start_page, end_page = block.page_span
                    if end_page > 0:  # Only consider valid end pages
                        max_end_page = max(max_end_page, end_page)
            
            # Next page will be numbered after the highest end_page in this page
            current_page_number = max_end_page + 1
        
        # Second pass: update all element_paths and page_spans using the mapping
        for page_idx, page in enumerate(new_pages):
            new_page_no = page.page_no
            updated_page = ExternalOcrExecutor._update_page_elements(page, new_page_no, page_number_map)
            new_pages[page_idx] = updated_page
            
        return new_pages
    
    @staticmethod
    def _update_page_elements(page, new_page_no, page_number_map):
        """
        Update all elements within a page, including element_paths and page_spans.
        
        Args:
            page: The page to update
            new_page_no: The new page number
            page_number_map: Mapping from old page numbers to new page numbers
        
        Returns:
            Updated page with all elements properly updated
        """
        # Update element_path for the page itself
        if hasattr(page, 'element_path') and page.element_path is not None:
            parts = list(page.element_path.parts)
            if len(parts) > 1:
                parts[1] = str(new_page_no)
                new_path = Path(*parts)
                page = replace(page, element_path=new_path)
        
        # Update blocks
        updated_blocks = []
        for block in page.blocks:
            # Update page_span for the block
            if hasattr(block, 'page_span') and block.page_span != (-1, -1):
                old_start, old_end = block.page_span
                # Map old page numbers to new page numbers, defaulting to the current page if not in the map
                new_start = page_number_map.get(old_start, new_page_no)
                new_end = page_number_map.get(old_end, new_page_no)
                block = replace(block, page_span=(new_start, new_end))
            
            # Update element_path for the block
            if hasattr(block, 'element_path') and block.element_path is not None:
                parts = list(block.element_path.parts)
                if len(parts) > 1:
                    parts[1] = str(new_page_no)
                    new_path = Path(*parts)
                    block = replace(block, element_path=new_path)
            
            # Recursively update all child elements
            updated_block = ExternalOcrExecutor._update_recursive(block, new_page_no, page_number_map)
            updated_blocks.append(updated_block)
        
        # Replace the blocks list with the updated blocks
        page = replace(page, blocks=updated_blocks)
        return page
    
    @staticmethod
    def _update_recursive(element, new_page_no, page_number_map):
        """
        Recursively update all nested elements within an element.
        
        Args:
            element: The element to update
            new_page_no: The new page number
            page_number_map: Mapping from old page numbers to new page numbers
        
        Returns:
            Updated element with all nested elements properly updated
        """
        if not is_dataclass(element):
            return element
            
        # Update element_path if present
        if hasattr(element, 'element_path') and element.element_path is not None:
            parts = list(element.element_path.parts)
            if len(parts) > 1:
                parts[1] = str(new_page_no)
                new_path = Path(*parts)
                element = replace(element, element_path=new_path)
        
        # Update page_span if present
        if hasattr(element, 'page_span') and element.page_span != (-1, -1):
            old_start, old_end = element.page_span
            new_start = page_number_map.get(old_start, new_page_no)
            new_end = page_number_map.get(old_end, new_page_no)
            if (new_start, new_end) != element.page_span:
                element = replace(element, page_span=(new_start, new_end))
        
        # Process all fields that might contain nested elements
        for field_name, field_def in getattr(element, '__dataclass_fields__', {}).items():
            value = getattr(element, field_name)
            
            if isinstance(value, list):
                # Update all elements in a list
                updated_list = []
                for item in value:
                    updated_item = ExternalOcrExecutor._update_recursive(item, new_page_no, page_number_map)
                    updated_list.append(updated_item)
                if value != updated_list:
                    element = replace(element, **{field_name: updated_list})
            elif is_dataclass(value):
                # Update a single nested dataclass
                updated_value = ExternalOcrExecutor._update_recursive(value, new_page_no, page_number_map)
                if value != updated_value:
                    element = replace(element, **{field_name: updated_value})
        
        return element