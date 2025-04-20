import io
from typing import Any, Dict, List, Optional

import pytesseract
from PIL import Image

from ocrtool.canonical_ocr.ocr_schema import (
    BoundingBox, Symbol, Word, Line, Paragraph, Block, Page, Document, OcrResult
)
from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor
from ocrtool.page_limit.page_count import is_pdf, count_pdf_pages, split_pdf_to_segments
from ocrtool.page_limit.exceptions import PageLimitExceededError
from ocrtool.page_limit.limits import OcrExecutorType, get_page_limit


class TesseractOcrExecutor(ExternalOcrExecutor):
    """
    OCR executor implementation using Tesseract OCR.
    """
    
    @property
    def type(self) -> OcrExecutorType:
        return OcrExecutorType.TESSERACT

    @property
    def page_limit(self) -> int | None:
        return get_page_limit(self.type)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, handle_page_limit: bool = True) -> None:
        """
        Initialize the Tesseract OCR executor.
        
        Args:
            config: Optional configuration parameters for Tesseract
                - lang: Language code (default: 'eng')
                - config: Tesseract configuration string
                - timeout: Timeout in seconds
            handle_page_limit: Whether to handle page limit errors automatically (default: True)
        """
        super().__init__(handle_page_limit=handle_page_limit)
        self.config = config or {}
        self.lang = self.config.get('lang', 'eng')
        self.tesseract_config = self.config.get('config', '')
        self.timeout = self.config.get('timeout', 30)
        
        # Store the most recent native result
        self._last_result = None
    
    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR on the provided image data and return results in native format
        as a JSON-serializable dictionary.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
                - lang: Override the default language
                - config: Override the default tesseract config
                
        Returns:
            Dict[str, Any]: Tesseract OCR result as a JSON-serializable dictionary
        """
        # Override config with kwargs if provided
        lang = kwargs.get('lang', self.lang)
        config = kwargs.get('config', self.tesseract_config)
        
        # Load image data
        image = Image.open(io.BytesIO(image_data))
        
        # Execute Tesseract OCR with detailed output
        result = pytesseract.image_to_data(
            image, 
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
            timeout=self.timeout
        )
        
        # Convert NumPy int arrays to Python lists of integers for JSON serialization
        json_safe_result = {}
        for key, value in result.items():
            # Handle NumPy arrays - convert to standard Python types
            if hasattr(value, 'tolist'):
                json_safe_result[key] = value.tolist()
            else:
                json_safe_result[key] = value
        
        self._last_result = json_safe_result
        return json_safe_result
    
    def get_native_result(self) -> Any:
        """
        Return the native result from the most recent OCR execution.
        
        Returns:
            Dict: The Tesseract OCR result in its native dictionary format
        """
        return self._last_result
    
    def get_implementation_info(self) -> Dict[str, str]:
        """
        Return information about this OCR implementation.
        
        Returns:
            Dict[str, str]: Information about the implementation
        """
        return {
            "name": "Tesseract OCR",
            "version": pytesseract.get_tesseract_version(),
            "description": "Open source OCR engine with Python interface"
        }
    
    def convert_to_canonical(self, native_result: Dict) -> OcrResult:
        """
        Convert Tesseract's native output to the canonical OCR schema.
        
        Args:
            native_result: Tesseract OCR result in its native dictionary format
            
        Returns:
            OcrResult: The converted result in canonical format
        """
        self._last_result = native_result
        return self._convert_to_canonical()
    
    def _convert_to_canonical(self) -> OcrResult:
        """
        Convert Tesseract's native output to the canonical OCR schema.
        
        Returns:
            OcrResult: The converted result in canonical format
        """
        if not self._last_result:
            # Return empty document if no result is available
            return OcrResult(document=Document(pages=[]))
        
        # Extract data from the Tesseract result
        result = self._last_result
        
        # Organize data by page (Tesseract processes one page at a time)
        page_data = {1: {}}  # Default to page 1
        
        # Group blocks, paragraphs, lines, and words
        blocks = {}
        paragraphs = {}
        lines = {}
        words = {}
        
        # Process each detected text element
        for i in range(len(result['text'])):
            # Skip entries with empty text or very low confidence
            if not result['text'][i].strip() or float(result['conf'][i]) < 0:
                continue
                
            # Extract data
            block_num = result['block_num'][i]
            para_num = result['par_num'][i]
            line_num = result['line_num'][i]
            word_num = result['word_num'][i]
            
            # Create unique identifiers for each level
            block_id = f"{block_num}"
            para_id = f"{block_num}_{para_num}"
            line_id = f"{block_num}_{para_num}_{line_num}"
            word_id = f"{block_num}_{para_num}_{line_num}_{word_num}"
            
            # Calculate bounding box (Tesseract coordinates are top-left based)
            left = result['left'][i]
            top = result['top'][i]
            width = result['width'][i]
            height = result['height'][i]
            
            bbox = BoundingBox(
                left=left,
                top=top,
                width=width,
                height=height
            )
            
            # Create word if it doesn't exist
            if word_id not in words:
                text = result['text'][i]
                confidence = float(result['conf'][i]) / 100  # Convert to 0-1 range
                
                # Create symbols (characters)
                symbols = []
                for char_idx, char in enumerate(text):
                    symbol = Symbol(
                        element_path=f"symbol_{char_idx}",
                        text_value=char,
                        confidence=confidence
                    )
                    symbols.append(symbol)
                
                word = Word(
                    element_path=f"word_{word_id}",
                    boundingBox=bbox,
                    symbols=symbols,
                    confidence=confidence
                )
                words[word_id] = word
            
            # Add/update line
            if line_id not in lines:
                lines[line_id] = {
                    'bbox': bbox,
                    'words': [],
                    'confidence': float(result['conf'][i]) / 100
                }
            else:
                # Expand bounding box if needed
                current_bbox = lines[line_id]['bbox']
                lines[line_id]['bbox'] = BoundingBox(
                    left=min(current_bbox.left or 0, bbox.left or 0),
                    top=min(current_bbox.top or 0, bbox.top or 0),
                    width=max((current_bbox.left or 0) + (current_bbox.width or 0), 
                              (bbox.left or 0) + (bbox.width or 0)) - min(current_bbox.left or 0, bbox.left or 0),
                    height=max((current_bbox.top or 0) + (current_bbox.height or 0), 
                               (bbox.top or 0) + (bbox.height or 0)) - min(current_bbox.top or 0, bbox.top or 0)
                )
            
            if word_id not in lines[line_id]['words']:
                lines[line_id]['words'].append(word_id)
            
            # Add/update paragraph
            if para_id not in paragraphs:
                paragraphs[para_id] = {
                    'bbox': bbox,
                    'lines': [],
                    'confidence': float(result['conf'][i]) / 100
                }
            else:
                # Expand bounding box if needed
                current_bbox = paragraphs[para_id]['bbox']
                paragraphs[para_id]['bbox'] = BoundingBox(
                    left=min(current_bbox.left or 0, bbox.left or 0),
                    top=min(current_bbox.top or 0, bbox.top or 0),
                    width=max((current_bbox.left or 0) + (current_bbox.width or 0), 
                              (bbox.left or 0) + (bbox.width or 0)) - min(current_bbox.left or 0, bbox.left or 0),
                    height=max((current_bbox.top or 0) + (current_bbox.height or 0), 
                               (bbox.top or 0) + (bbox.height or 0)) - min(current_bbox.top or 0, bbox.top or 0)
                )
            
            if line_id not in paragraphs[para_id]['lines']:
                paragraphs[para_id]['lines'].append(line_id)
            
            # Add/update block
            if block_id not in blocks:
                blocks[block_id] = {
                    'bbox': bbox,
                    'paragraphs': [],
                    'confidence': float(result['conf'][i]) / 100,
                    'block_no': block_num
                }
            else:
                # Expand bounding box if needed
                current_bbox = blocks[block_id]['bbox']
                blocks[block_id]['bbox'] = BoundingBox(
                    left=min(current_bbox.left or 0, bbox.left or 0),
                    top=min(current_bbox.top or 0, bbox.top or 0),
                    width=max((current_bbox.left or 0) + (current_bbox.width or 0), 
                              (bbox.left or 0) + (bbox.width or 0)) - min(current_bbox.left or 0, bbox.left or 0),
                    height=max((current_bbox.top or 0) + (current_bbox.height or 0), 
                               (bbox.top or 0) + (bbox.height or 0)) - min(current_bbox.top or 0, bbox.top or 0)
                )
            
            if para_id not in blocks[block_id]['paragraphs']:
                blocks[block_id]['paragraphs'].append(para_id)
        
        # Build the canonical structure
        # First convert lines dictionary to Line objects
        line_objects = {}
        for line_id, line_data in lines.items():
            line_words = [words[word_id] for word_id in line_data['words']]
            line_objects[line_id] = Line(
                element_path=f"line_{line_id}",
                boundingBox=line_data['bbox'],
                words=line_words,
                confidence=line_data['confidence']
            )
        
        # Convert paragraphs dictionary to Paragraph objects
        para_objects = {}
        for para_id, para_data in paragraphs.items():
            para_lines = [line_objects[line_id] for line_id in para_data['lines']]
            para_objects[para_id] = Paragraph(
                element_path=f"paragraph_{para_id}",
                boundingBox=para_data['bbox'],
                lines=para_lines,
                confidence=para_data['confidence']
            )
        
        # Convert blocks dictionary to Block objects
        block_objects = []
        for block_id, block_data in blocks.items():
            block_paras = [para_objects[para_id] for para_id in block_data['paragraphs']]
            block = Block(
                element_path=f"block_{block_id}",
                boundingBox=block_data['bbox'],
                paragraphs=block_paras,
                blockType="TEXT",  # Tesseract doesn't distinguish block types
                confidence=block_data['confidence'],
                block_no=block_data['block_no']
            )
            block_objects.append(block)
        
        # Create image dimensions from detected content
        width = max(
            ((block.boundingBox.left or 0) + (block.boundingBox.width or 0))
            for block in block_objects
        ) if block_objects else 0
        
        height = max(
            ((block.boundingBox.top or 0) + (block.boundingBox.height or 0))
            for block in block_objects
        ) if block_objects else 0
        
        # Create the page
        page = Page(
            element_path="page_1",
            boundingBox=BoundingBox(left=0, top=0, width=width, height=height),
            width=int(width),
            height=int(height),
            blocks=block_objects,
            confidence=sum(block.confidence for block in block_objects) / len(block_objects) if block_objects else 0,
            page_no=1
        )
        
        # Create the document
        document = Document(
            element_path="document",
            boundingBox=BoundingBox(left=0, top=0, width=width, height=height),
            pages=[page],
            extra={"engine": "tesseract", "version": pytesseract.get_tesseract_version()}
        )
        
        return OcrResult(document=document)