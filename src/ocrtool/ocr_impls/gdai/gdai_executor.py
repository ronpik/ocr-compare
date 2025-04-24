import io
from typing import Any, Dict, Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict

from ocrtool.canonical_ocr.ocr_schema import OcrResult, Document
from ocrtool.ocr_impls.gdai import GdaiConfig
from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_convert import process_documentai_result
from ocrtool.ocr_impls.gdai.gdai_layout_executor import process_layout_result
from ocrtool.page_limit.limits import OcrExecutorType, get_page_limit

class GoogleDocumentAIBaseExecutor(ExternalOcrExecutor):
    """
    Base executor for Google Document AI processors (OCR, Layout, etc).
    Handles client setup, credentials, and request execution.
    """
    def __init__(self, config: Optional[GdaiConfig | dict[str, Any]] = None, handle_page_limit: bool = True) -> None:
        """
        Initialize the Google Document AI base executor.

        Args:
            config: Configuration for the executor.
            handle_page_limit: Whether to handle page limit errors automatically (default: True)
        """
        super().__init__(handle_page_limit=handle_page_limit)
        if isinstance(config, dict):
            self.config = GdaiConfig(**config)
        else:
            self.config = config or GdaiConfig()

        self.processor_name = self.config.processor_name
        self.location = self.config.location
        self.timeout = self.config.timeout
        self.credentials = self._setup_credentials()
        api_endpoint = f"{self.location}-documentai.googleapis.com"
        self.client = documentai.DocumentProcessorServiceClient(
            credentials=self.credentials,
            client_options=ClientOptions(api_endpoint=api_endpoint)
        )
        self._last_result = None

    def _setup_credentials(self):
        if self.config.service_account_file:
            return service_account.Credentials.from_service_account_file(
                self.config.service_account_file
            )
        if self.config.service_account_info:
            return service_account.Credentials.from_service_account_info(
                self.config.service_account_info
            )
        import google.auth
        credentials, _ = google.auth.default()
        return credentials

    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        mime_type = kwargs.get('mime_type')
        if not mime_type:
            mime_type = self._detect_mime_type(image_data)
        timeout = kwargs.get('timeout', self.timeout)
        raw_document = documentai.RawDocument(
            content=image_data,
            mime_type=mime_type
        )
        process_options = self._get_process_options()
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
            skip_human_review=True,
            # process_options=process_options,
        )
        result = self.client.process_document(request=request, timeout=timeout)
        result_dict = MessageToDict(result._pb)
        
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(i) for i in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)
            
        json_safe_result = sanitize_for_json(result_dict)
        self._last_result = json_safe_result
        return json_safe_result

    def _detect_mime_type(self, data: bytes) -> str:
        import magic
        try:
            mime_type = magic.from_buffer(data, mime=True)
            return mime_type
        except ImportError:
            if data.startswith(b'%PDF'):
                return 'application/pdf'
            elif data.startswith(b'\xff\xd8'):
                return 'image/jpeg'
            elif data.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
                return 'image/gif'
            elif data.startswith(b'RIFF') and data[8:12] == b'WEBP':
                return 'image/webp'
            elif data.startswith(b'II*\x00') or data.startswith(b'MM\x00*'):
                return 'image/tiff'
            else:
                return 'application/pdf'

    def get_native_result(self) -> Any:
        return self._last_result

    def get_implementation_info(self) -> Dict[str, str]:
        return {
            "name": "Google Document AI (Base)",
            "version": "1.0.0",
            "description": "Google Cloud Document AI base executor"
        }

    def _get_process_options(self):
        """
        Subclasses should override this to provide processor-specific options.
        """
        raise NotImplementedError

    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        """
        Subclasses should override this to provide processor-specific conversion.
        """
        raise NotImplementedError

    @property
    def type(self) -> OcrExecutorType:
        # Default to online; subclasses should override if needed
        return OcrExecutorType.DOCUMENT_AI_ONLINE

class GoogleDocumentAIOcrExecutor(GoogleDocumentAIBaseExecutor):
    """
    OCR executor implementation using Google Document AI OCR processor.
    """
    @property
    def type(self) -> OcrExecutorType:
        return OcrExecutorType.DOCUMENT_AI_ONLINE

    @property
    def page_limit(self) -> int | None:
        return get_page_limit(self.type)

    def __init__(self, config: Optional[GdaiConfig | dict[str, Any]] = None, handle_page_limit: bool = True) -> None:
        """
        Initialize the Google Document AI OCR executor.

        Args:
            config: Configuration for the executor.
            handle_page_limit: Whether to handle page limit errors automatically (default: True)
        """
        super().__init__(config=config, handle_page_limit=handle_page_limit)

    def _get_process_options(self):
        return documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_image_quality_scores=False,
            )
        )

    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        self._last_result = native_result
        return process_documentai_result(native_result)

    def get_implementation_info(self) -> Dict[str, str]:
        return {
            "name": "Google Document AI OCR",
            "version": "1.0.0",
            "description": "Google Cloud Document AI OCR engine"
        }

class GoogleDocumentAILayoutExecutor(GoogleDocumentAIBaseExecutor):
    """
    Layout executor implementation using Google Document AI Layout processor.
    """
    @property
    def type(self) -> OcrExecutorType:
        return OcrExecutorType.DOCUMENT_AI_LAYOUT_PARSER

    def _get_process_options(self):
        return documentai.ProcessOptions(
            layout_config=documentai.ProcessOptions.LayoutConfig(
                chunking_config=documentai.ProcessOptions.LayoutConfig.ChunkingConfig(
                    chunk_size=1000,
                    include_ancestor_headings=True,
                )
            )
        )

    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        self._last_result = native_result
        return process_layout_result(native_result)

    def get_implementation_info(self) -> Dict[str, str]:
        return {
            "name": "Google Document AI Layout",
            "version": "1.0.0",
            "description": "Google Cloud Document AI Layout engine"
        }