from dstools.storage.handlers.storage_handler import StorageHandler

class SSHStorageHandler(StorageHandler):
    def __init__(self, credentials: dict):
        self.host = credentials["host"]
        self.port = credentials["port"]
        self.user = credentials["user"]
        self.password = credentials["password"]
        # Initialize SSH client here (e.g., paramiko)

    def download(self, remote_relative_path: str) -> bytes:
        print(f"Downloading {remote_relative_path} via SSH from {self.host}:{self.port}")
        # Implement actual download logic and return compressed bytes
        return b""

    def upload(self, compressed_data: bytes, remote_relative_path: str):
        print(f"Uploading {remote_relative_path} via SSH to {self.host}:{self.port}")
        # Implement actual upload logic