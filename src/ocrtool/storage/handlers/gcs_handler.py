from pathlib import Path
from typing import Iterable, Iterator

from google.cloud import storage
from google.cloud.storage import Blob
from globalog import LOG


from dstools.storage.handlers.storage_handler import StorageHandler


def get_src_root() -> Path:
    return Path(__file__).parent


class GCSHandler(StorageHandler):
    def __init__(self, storage_config: dict[str, str]):
        self._bucket_name = storage_config["bucket"]
        credentials_path = str(get_src_root() / storage_config["credentials_path"])
        self._client = storage.Client.from_service_account_json(credentials_path)
        self._bucket = self._client.get_bucket(self._bucket_name)

    def download(self, remote_relative_path: str) -> bytes:
        blob = self._bucket.blob(remote_relative_path)
        LOG.debug(f"Download from {remote_relative_path}")
        content = blob.download_as_bytes()
        LOG.debug(f"Downloaded {len(content)} bytes from GCS at {remote_relative_path}.")
        return content

    def upload(self, content: bytes, remote_relative_path: str) -> bool:
        try:
            blob = self._bucket.blob(remote_relative_path)
            LOG.debug(f"Upload {len(content)} bytes to GCS at {remote_relative_path}")
            blob.upload_from_string(content)
            LOG.debug(f"Uploaded {remote_relative_path} to GCS.")
        except Exception as e:
            LOG.error(f"Failed to upload {remote_relative_path} to GCS.", exc_info=e)
            return False

        return True

    def exists(self, remote_relative_path: str) -> bool:
        """Check if the resource exists in GCS without downloading the full content."""
        blob = self._bucket.blob(remote_relative_path)
        return blob.exists()

    def size(self, remote_relative_path: str) -> int:
        blob = self._bucket.get_blob(remote_relative_path)
        return blob.size

    def list_objects(self, prefix: str) -> Iterator[Blob]:
        yield from self._bucket.list_blobs(prefix=prefix)

if __name__ == '__main__':
    config = get_gcs_config()
    gcs = GCSHandler(config)
    path = 'gv-ocr/012167cb-5303-4ff7-801e-f1385161bfa3/output-1-to-1.json'
    print(gcs.size(path))
