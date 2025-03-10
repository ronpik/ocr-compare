from pathlib import Path

from dstools.common.io_utils import read_bytes, write_bytes
from dstools.storage.handlers.storage_handler import StorageHandler


class LocalStorageHandler(StorageHandler):

    def __init__(self, root_dir: Path):
        self._root = root_dir

    def download(self, remote_relative_path: str) -> bytes:
        return read_bytes(self._root / remote_relative_path)

    def upload(self, compressed_data: bytes, remote_relative_path: str):
        return write_bytes(self._root / remote_relative_path, compressed_data)
