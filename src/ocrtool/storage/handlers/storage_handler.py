from abc import ABC, abstractmethod
from typing import Dict

class StorageHandler(ABC):
    @abstractmethod
    def download(self, remote_relative_path: str) -> bytes:
        """Download the resource and return it as compressed bytes."""
        raise NotImplementedError()

    @abstractmethod
    def upload(self, compressed_data: bytes, remote_relative_path: str) -> bool:
        """Upload the compressed data to the remote path."""
        raise NotImplementedError()


class StorageHandlerFactory:
    @staticmethod
    def get_handler(storage_type: str, storage_config: dict) -> StorageHandler:
        storage_type = storage_type.upper()
        if storage_type == "S3":
            from ocrtool.storage.handlers.s3_handler import S3StorageHandler
            return S3StorageHandler(storage_config)
        elif storage_type == "GCS":
            from ocrtool.storage.handlers.gcs_handler import GCSHandler
            return GCSHandler(storage_config)
        elif storage_type == "SSH":
            from ocrtool.storage.handlers.ssh_handler import SSHStorageHandler
            return SSHStorageHandler(storage_config)
        elif storage_type == "LOCAL":
            from ocrtool.storage.handlers.local_handler import LocalStorageHandler
            return LocalStorageHandler(storage_config['root_dir'])

        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")