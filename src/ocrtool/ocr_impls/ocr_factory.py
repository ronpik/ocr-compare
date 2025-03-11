from typing import Dict, Type, Any, Optional

from ocrtool.ocr_impls.ocr_executor import OcrExecutor


class OcrExecutorFactory:
    """
    Factory class for creating OCR executor instances based on the strategy pattern.
    This factory manages the registration and instantiation of OCR implementations.
    """
    
    _registered_executors: Dict[str, Type[OcrExecutor]] = {}
    
    @classmethod
    def register(cls, name: str, executor_class: Type[OcrExecutor]) -> None:
        """
        Register an OCR executor implementation with the factory.
        
        Args:
            name: Unique identifier for the OCR implementation
            executor_class: The class implementing the OcrExecutor interface
        """
        cls._registered_executors[name.lower()] = executor_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> OcrExecutor:
        """
        Create an instance of the specified OCR executor.
        
        Args:
            name: Name of the OCR implementation to create
            **kwargs: Configuration parameters for the OCR executor
            
        Returns:
            An instance of OcrExecutor
            
        Raises:
            ValueError: If the specified OCR implementation is not registered
        """
        executor_class = cls._registered_executors.get(name.lower())
        if not executor_class:
            registered = ", ".join(cls._registered_executors.keys())
            raise ValueError(
                f"OCR implementation '{name}' not found. Available implementations: {registered}"
            )

        return executor_class(kwargs)
    
    @classmethod
    def list_available(cls) -> Dict[str, Dict[str, str]]:
        """
        List all available OCR implementations with their metadata.
        
        Returns:
            Dictionary mapping implementation name to metadata
        """
        result = {}
        
        for name, executor_class in cls._registered_executors.items():
            # Create a temporary instance to get implementation info
            # This assumes implementations can be instantiated without args for info
            try:
                # Try to instantiate with no args just to get metadata
                instance = executor_class()
                result[name] = instance.get_implementation_info()
            except Exception:
                # If instantiation fails, provide basic info
                result[name] = {
                    "name": name,
                    "description": f"Implementation of {executor_class.__name__}",
                    "version": "unknown"
                }
                
        return result