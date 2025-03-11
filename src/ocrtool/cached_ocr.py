"""
Utilities for creating cached OCR executors.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

from dstools.storage.handlers import StorageHandlerConfig

from ocrtool.ocr_impls.cached_ocr_executor import CachedOcrExecutor
from ocrtool.ocr_impls.ocr_executor import OcrExecutor
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory


def get_cached_executor(
    ocr_executor_name: str,
    ocr_config: Optional[Dict[str, Any]] = None,
    storage_config_path: Optional[str] = None,
    storage_type: Optional[str] = None,
    storage_config: Optional[Dict[str, Any]] = None,
    cache_prefix: str = "ocr_cache"
) -> CachedOcrExecutor:
    """
    Create a cached OCR executor with the specified OCR engine and storage backend.
    
    Args:
        ocr_executor_name: Name of the OCR executor to use
        ocr_config: Configuration for the OCR executor
        storage_config_path: Path to storage configuration file
        storage_type: Type of storage backend (if not using config file)
        storage_config: Storage configuration (if not using config file)
        cache_prefix: Prefix for cache storage
        
    Returns:
        CachedOcrExecutor: The cached OCR executor
    """
    # Initialize the OCR executor
    base_executor = OcrExecutorFactory.create(ocr_executor_name, ocr_config or {})
    
    # Initialize the storage handler
    storage_handler = None
    
    # Try to get storage handler from environment if not specified
    if not storage_config_path and not storage_type:
        storage_handler = StorageHandlerConfig.from_env()
    
    # Get storage handler from config file if provided
    if storage_config_path:
        storage_handler = StorageHandlerConfig.create_handler_from_file(storage_config_path)
    
    # Get storage handler from arguments if provided
    elif storage_type and storage_config:
        storage_handler = StorageHandlerConfig.create_handler_from_args(storage_type, storage_config)
        
    # Default to local storage if no other option specified
    if not storage_handler:
        from dstools.storage.handlers.local_handler import LocalStorageHandler
        # Use home directory for cache
        cache_dir = Path.home() / '.ocr_cache'
        os.makedirs(cache_dir, exist_ok=True)
        storage_handler = LocalStorageHandler(cache_dir)
    
    # Create and return the cached executor
    return CachedOcrExecutor(
        base_executor,
        storage_handler,
        cache_prefix
    )


def get_cached_ocr_engine(
    executor: Union[str, OcrExecutor],
    storage_path: Optional[str] = None,
    cache_prefix: str = "ocr_cache"
) -> CachedOcrExecutor:
    """
    Simplified function to get a cached OCR engine with minimal parameters.
    
    Args:
        executor: Either the name of an OCR executor or an existing executor instance
        storage_path: Path to storage configuration file (optional)
        cache_prefix: Prefix for cache storage
        
    Returns:
        CachedOcrExecutor: The cached OCR executor
    """
    # If executor is a string, create the executor
    if isinstance(executor, str):
        base_executor = OcrExecutorFactory.create(executor)
    else:
        base_executor = executor
    
    # Initialize storage
    storage_handler = None
    if storage_path:
        storage_handler = StorageHandlerConfig.create_handler_from_file(storage_path)
    else:
        storage_handler = StorageHandlerConfig.from_env()
        
    # Default to local storage if no other option specified
    if not storage_handler:
        from dstools.storage.handlers.local_handler import LocalStorageHandler
        # Use home directory for cache
        cache_dir = Path.home() / '.ocr_cache'
        os.makedirs(cache_dir, exist_ok=True)
        storage_handler = LocalStorageHandler(cache_dir)
    
    # Create and return the cached executor
    return CachedOcrExecutor(
        base_executor,
        storage_handler,
        cache_prefix
    )