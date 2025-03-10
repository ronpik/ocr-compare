import asyncio

from dstools.storage.handlers.storage_handler import StorageHandler, StorageHandlerFactory


class AsyncStorageHandler:
    def __init__(self, handler: StorageHandler):
        if not isinstance(handler, StorageHandler):
            raise TypeError("handler must be an instance of StorageHandler")
        self.handler = handler

    async def download(self, remote_relative_path: str) -> bytes:
        """Download content asynchronously."""
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, self.handler.download, remote_relative_path)
        return data

    async def upload(self, compressed_data: bytes, remote_relative_path: str) -> bool:
        """Upload content asynchronously."""
        loop = asyncio.get_event_loop()
        status = await loop.run_in_executor(None, self.handler.upload, compressed_data, remote_relative_path)
        return status


class AsyncStorageHandlerFactory:
    @staticmethod
    def get_async_handler(storage_type: str, storage_config: dict) -> AsyncStorageHandler:
        handler = StorageHandlerFactory.get_handler(storage_type, storage_config)
        return AsyncStorageHandler(handler)
