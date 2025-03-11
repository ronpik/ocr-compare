import hashlib
import json
from typing import Any, Dict

from dstools.storage.handlers.storage_handler import StorageHandler

from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor



class CachedOcrExecutor(ExternalOcrExecutor):
    """
    A decorator for OcrExecutor implementations that caches OCR results.
    Caches the native (original) result of the decorated executor.
    """
    
    def __init__(
        self,
        decorated_executor: ExternalOcrExecutor,
        storage_handler: StorageHandler,
        cache_prefix: str
    ):
        """
        Initialize the CachedOcrExecutor with a decorated executor and storage handler.
        
        Args:
            decorated_executor: The OcrExecutor to be decorated with caching
            storage_handler: The storage handler to use for caching
            cache_prefix: The prefix path in the storage where cache files will be stored
        """
        self._executor = decorated_executor
        self._storage = storage_handler
        self._cache_prefix = cache_prefix
        self._last_result = None
        
    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR with caching. Tries to fetch from cache first, falls back to 
        executing OCR and caching the result.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Dict[str, Any]: Results in native format of the underlying OCR engine
                           as a JSON-serializable dictionary
        """
        # Generate a cache key from image_data and kwargs
        cache_key = self._generate_cache_key(image_data, kwargs)
        # Use JSON for storage to ensure serialization compatibility
        cache_path = f"{self._cache_prefix}/{cache_key}.json"
        
        # Try to get from cache
        try:
            if self._is_cached(cache_path):
                cached_data = self._storage.download(cache_path)
                self._last_result = json.loads(cached_data.decode('utf-8'))
                return self._last_result
        except Exception as e:
            # Log error but continue with execution
            print(f"Cache retrieval error: {str(e)}")
        
        # Execute OCR using the decorated executor if not in cache
        if isinstance(self._executor, ExternalOcrExecutor):
            # If the decorated executor is an ExternalOcrExecutor, call its execute_ocr_original method
            result = self._executor.execute_ocr_original(image_data, **kwargs)
        else:
            # Otherwise, call execute_ocr and then get the native result
            self._executor.execute_ocr(image_data, **kwargs)
            result = self._executor.get_native_result()
            # For non-ExternalOcrExecutor, we need to ensure the result is JSON serializable
            if not isinstance(result, dict):
                # Convert non-dict results to a dictionary
                result = {"raw_result": str(result)}
        
        # Cache the result
        try:
            self._cache_result(result, cache_path)
        except Exception as e:
            # Log error but return the result anyway
            print(f"Cache storage error: {str(e)}")
        
        self._last_result = result
        return result
    
    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        """
        Convert the native OCR result to the canonical schema format.
        Delegates to the decorated executor for conversion.
        
        Args:
            native_result: OCR result in the implementation's native format
                          (JSON-serializable dictionary)
            
        Returns:
            OcrResult: Results in canonical schema format
        """
        return self._executor.convert_to_canonical(native_result)

    
    def get_native_result(self) -> Any:
        """
        Return the native result from the most recent OCR execution.
        
        Returns:
            Any: The OCR result in the implementation's native format
        """
        return self._last_result
    
    def get_implementation_info(self) -> Dict[str, str]:
        """
        Return information about this OCR implementation, including the decorated executor.
        
        Returns:
            Dict[str, str]: Information about the implementation
        """
        decorated_info = self._executor.get_implementation_info()
        return {
            "name": f"Cached {decorated_info.get('name', 'OCR')}",
            "version": decorated_info.get('version', 'unknown'),
            "description": f"Cached version of: {decorated_info.get('description', '')}",
            "cache_prefix": self._cache_prefix
        }
    
    def _generate_cache_key(self, image_data: bytes, kwargs: Dict) -> str:
        """
        Generate a cache key from the image data and kwargs.
        
        Args:
            image_data: The image data bytes
            kwargs: The keyword arguments passed to execute_ocr
            
        Returns:
            str: A hash-based cache key
        """
        # Create a hash of the image data
        img_hash = hashlib.md5(image_data).hexdigest()
        
        # Create a stable representation of kwargs for hashing
        stable_kwargs = json.dumps(
            {k: str(v) for k, v in sorted(kwargs.items()) if v is not None},
            sort_keys=True
        )
        kwargs_hash = hashlib.md5(stable_kwargs.encode('utf-8')).hexdigest()
        
        # Combine with executor info for uniqueness across different implementations
        impl_info = self._executor.get_implementation_info()
        impl_name = impl_info.get('name', '').replace(' ', '_').lower()
        
        return f"{impl_name}_{img_hash}_{kwargs_hash}"
    
    def _is_cached(self, cache_path: str) -> bool:
        """
        Check if a cache file exists.
        
        Args:
            cache_path: The path to the cache file
            
        Returns:
            bool: True if the cache file exists, False otherwise
        """
        try:
            # For storage handlers that support existence checking
            if hasattr(self._storage, 'exists'):
                return self._storage.exists(cache_path)
                
            # Fallback: try to download and catch exceptions
            self._storage.download(cache_path)
            return True
        except Exception:
            return False
    
    def _cache_result(self, result: Dict[str, Any], cache_path: str) -> None:
        """
        Cache the OCR result using JSON serialization.
        
        Args:
            result: The OCR result to cache (JSON-serializable dictionary)
            cache_path: The path to the cache file
        """
        try:
            # Serialize to JSON
            json_data = json.dumps(result)
            
            # Upload to storage
            self._storage.upload(json_data.encode('utf-8'), cache_path)
        except Exception as e:
            # Log the error but continue
            print(f"Error caching OCR result: {str(e)}")