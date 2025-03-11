"""
Configuration dataclass for Google Document AI OCR.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class GdaiConfig:
    """
    Configuration dataclass for Google Document AI OCR.
    """
    processor_name: str = ''
    location: str = "us"
    timeout: int = 300
    service_account_file: Optional[str] = None
    service_account_info: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_file(cls, config_path: str) -> "GdaiConfig":
        """
        Load configuration from a JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            GdaiConfig: Loaded configuration
        """
        with open(config_path, "r") as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to a dictionary.
        
        Returns:
            Dict[str, Any]: Configuration as a dictionary
        """
        config_dict = {
            "processor_name": self.processor_name,
            "location": self.location,
            "timeout": self.timeout,
        }
        
        if self.service_account_file:
            config_dict["service_account_file"] = self.service_account_file
            
        if self.service_account_info:
            config_dict["service_account_info"] = self.service_account_info
            
        return config_dict