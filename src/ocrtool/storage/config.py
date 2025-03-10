import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

from ocrtool.storage.handlers.storage_handler import StorageHandler, StorageHandlerFactory


class StorageConfig:
    """
    Configuration handler for storage settings.
    """
    
    @staticmethod
    def load_from_file(config_path: str) -> Dict[str, Any]:
        """
        Load storage configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Dict: Storage configuration
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate required fields
        if 'storage_type' not in config:
            raise ValueError("Storage configuration must contain 'storage_type'")
        
        if 'storage_config' not in config:
            raise ValueError("Storage configuration must contain 'storage_config'")
            
        return config
    
    @staticmethod
    def create_handler_from_file(config_path: str) -> StorageHandler:
        """
        Create a StorageHandler from a configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            StorageHandler: Initialized storage handler
        """
        config = StorageConfig.load_from_file(config_path)
        return StorageHandlerFactory.get_handler(
            config['storage_type'], 
            config['storage_config']
        )
    
    @staticmethod
    def create_handler_from_args(storage_type: str, storage_config: Dict[str, Any]) -> StorageHandler:
        """
        Create a StorageHandler from storage type and configuration.
        
        Args:
            storage_type: Type of storage (e.g., "LOCAL", "GCS", "S3")
            storage_config: Configuration for the storage handler
            
        Returns:
            StorageHandler: Initialized storage handler
        """
        return StorageHandlerFactory.get_handler(storage_type, storage_config)
    
    @staticmethod
    def from_env(env_var: str = "OCR_STORAGE_CONFIG") -> Optional[StorageHandler]:
        """
        Create a StorageHandler from an environment variable path.
        
        Args:
            env_var: Environment variable containing the path to config file
            
        Returns:
            Optional[StorageHandler]: Initialized storage handler, or None if env var not set
        """
        config_path = os.environ.get(env_var)
        if not config_path:
            return None
            
        return StorageConfig.create_handler_from_file(config_path)