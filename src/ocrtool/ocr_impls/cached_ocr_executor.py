import hashlib
import json
from typing import Any, Dict

from dstools.storage.handlers.storage_handler import StorageHandler

from ocrtool.canonical_ocr.ocr_schema import OcrResult, Document
from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor
from ocrtool.page_limit.page_count import is_pdf, count_pdf_pages, split_pdf_to_segments
from ocrtool.page_limit.exceptions import PageLimitExceededError
from ocrtool.page_limit.limits import OcrExecutorType, get_page_limit



class CachedOcrExecutor(ExternalOcrExecutor):
    """
    A decorator for OcrExecutor implementations that caches OCR results.
    Caches the native (original) result of the decorated executor.
    """
    
    def __init__(
        self,
        decorated_executor: ExternalOcrExecutor,
        storage_handler: StorageHandler,
        cache_prefix: str,
        handle_page_limit: bool = False  # Changed to False by default
    ) -> None:
        """
        Initialize the CachedOcrExecutor with a decorated executor and storage handler.
        
        Args:
            decorated_executor: The OcrExecutor to be decorated with caching
            storage_handler: The storage handler to use for caching
            cache_prefix: The prefix path in the storage where cache files will be stored
            handle_page_limit: Whether to handle page limit errors automatically (default: False)
                              This is set to False by default as page limit handling should be
                              delegated to the underlying executor
        """
        # Initialize with handle_page_limit=False to delegate handling to the decorated executor
        super().__init__(handle_page_limit=False)
        
        # Set the handle_page_limit flag on the decorated executor
        if hasattr(decorated_executor, 'handle_page_limit'):
            decorated_executor.handle_page_limit = handle_page_limit
        
        self._executor = decorated_executor
        self._storage = storage_handler
        self._cache_prefix = cache_prefix
        self._last_result = None
        
    @property
    def type(self) -> OcrExecutorType:
        # Use the decorated executor's type if available
        if hasattr(self._executor, 'type'):
            return self._executor.type
        raise NotImplementedError("Decorated executor must define a type property.")
    
    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR with caching. Tries to fetch from cache first, falls back to 
        executing OCR and caching the result.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
                      force_cache_refresh: If True, ignore cached results and force execution
            
        Returns:
            Dict[str, Any]: Results in native format of the underlying OCR engine
                           as a JSON-serializable dictionary
        """
        # Extract force_cache_refresh from kwargs, defaulting to False
        force_cache_refresh = kwargs.pop('force_cache_refresh', False)
        
        # Generate a cache key from image_data and kwargs
        cache_key = self._generate_cache_key(image_data, kwargs)
        # Use JSON for storage to ensure serialization compatibility
        cache_path = f"{self._cache_prefix}/{cache_key}.json"
        
        # Try to get from cache if not forcing refresh
        if not force_cache_refresh:
            try:
                if self._is_cached(cache_path):
                    cached_data = self._storage.download(cache_path)
                    self._last_result = json.loads(cached_data.decode('utf-8'))
                    return self._last_result
            except Exception as e:
                # Log error but continue with execution
                print(f"Cache retrieval error: {str(e)}")
        
        # Execute OCR using the decorated executor if not in cache or forcing refresh
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
    
    # def execute_ocr(self, image_data: bytes, **kwargs) -> OcrResult:
    #     """
    #     Override the execute_ocr method to delegate page limit handling to the decorated executor.
    #
    #     Args:
    #         image_data: Raw bytes of the image
    #         **kwargs: Additional implementation-specific parameters
    #                  force_cache_refresh: If True, ignore cached results and force execution
    #
    #     Returns:
    #         OcrResult: Results in canonical schema format
    #     """
    #     # If we're forcing a cache refresh, set this flag so execute_ocr_original will respect it
    #     force_cache_refresh = kwargs.get('force_cache_refresh', False)
    #
    #     # For PDF handling with page limits, delegate to the decorated executor
    #     if is_pdf(image_data) and self._executor.page_limit is not None:
    #         # Only check cache if not forcing refresh
    #         if not force_cache_refresh:
    #             # Try to get from cache first using our regular caching logic
    #             try:
    #                 cache_key = self._generate_cache_key(image_data, kwargs)
    #                 cache_path = f"{self._cache_prefix}/{cache_key}.json"
    #
    #                 if self._is_cached(cache_path):
    #                     cached_data = self._storage.download(cache_path)
    #                     self._last_result = json.loads(cached_data.decode('utf-8'))
    #                     return self.convert_to_canonical(self._last_result)
    #             except Exception as e:
    #                 print(f"Cache retrieval error during PDF handling: {str(e)}")
    #
    #         # Let the decorated executor handle page limits and processing
    #         result = self._executor.execute_ocr(image_data, **kwargs)
    #
    #         # Cache the final native result if available
    #         if hasattr(self._executor, 'get_native_result'):
    #             native_result = self._executor.get_native_result()
    #             if native_result:
    #                 self._last_result = native_result
    #                 try:
    #                     cache_key = self._generate_cache_key(image_data, kwargs)
    #                     cache_path = f"{self._cache_prefix}/{cache_key}.json"
    #                     self._cache_result(native_result, cache_path)
    #                 except Exception as e:
    #                     print(f"Cache storage error after PDF handling: {str(e)}")
    #
    #         return result
    #
    #     # For non-PDF or no page limits, use the standard behavior
    #     return super().execute_ocr(image_data, **kwargs)
    
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
        # Exclude force_cache_refresh from the key generation
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'force_cache_refresh' and v is not None}
        
        stable_kwargs = json.dumps(
            {k: str(v) for k, v in sorted(filtered_kwargs.items())},
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
        