from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ocrtool.canonical_ocr.ocr_schema import OcrResult


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
    """
    
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
        Execute OCR and convert the results to canonical format.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            OcrResult: Results in canonical schema format
        """
        native_result = self.execute_ocr_original(image_data, **kwargs)
        return self.convert_to_canonical(native_result)