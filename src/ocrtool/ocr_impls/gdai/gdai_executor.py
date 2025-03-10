import io
from typing import Any, Dict, Optional

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToDict

from ocrtool.canonical_ocr.ocr_schema import OcrResult
from ocrtool.ocr_impls.ocr_executor import ExternalOcrExecutor
from ocrtool.ocr_impls.gdai.gdai_convert import process_documentai_result


class GoogleDocumentAIOcrExecutor(ExternalOcrExecutor):
    """
    OCR executor implementation using Google Document AI.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Google Document AI OCR executor.
        
        Args:
            config: Configuration parameters for Google Document AI
                - processor_name: Full resource name of the processor to use (required)
                - credentials: One of the following:
                    - service_account_file: Path to service account JSON file
                    - service_account_info: Service account info as dictionary
                    - credentials: google.oauth2.credentials.Credentials object
                - location: API endpoint location (default: "us")
                - timeout: Timeout in seconds for API calls (default: 300)
        """
        self.config = config or {}
        
        # Validate required configuration
        if 'processor_name' not in self.config:
            raise ValueError("'processor_name' is required in the configuration")
        
        self.processor_name = self.config['processor_name']
        self.location = self.config.get('location', 'us')
        self.timeout = self.config.get('timeout', 300)
        
        # Set up credentials
        self.credentials = self._setup_credentials()
        
        # Set up client
        api_endpoint = f"{self.location}-documentai.googleapis.com"
        self.client = documentai.DocumentProcessorServiceClient(
            credentials=self.credentials,
            client_options=ClientOptions(api_endpoint=api_endpoint)
        )
        
        # Store the most recent native result
        self._last_result = None
    
    def _setup_credentials(self):
        """
        Set up Google Cloud credentials based on the provided configuration.
        
        Returns:
            google.auth.credentials.Credentials: The credentials to use for API calls
        """
        # Check if credentials object is directly provided
        if 'credentials' in self.config:
            return self.config['credentials']
        
        # Check for service account file path
        if 'service_account_file' in self.config:
            return service_account.Credentials.from_service_account_file(
                self.config['service_account_file']
            )
        
        # Check for service account info as dictionary
        if 'service_account_info' in self.config:
            return service_account.Credentials.from_service_account_info(
                self.config['service_account_info']
            )
        
        # Use default credentials if no specific credentials provided
        import google.auth
        credentials, _ = google.auth.default()
        return credentials
    
    def execute_ocr_original(self, image_data: bytes, **kwargs) -> Dict[str, Any]:
        """
        Execute OCR on the provided image data and return results in native format
        as a JSON-serializable dictionary.
        
        Args:
            image_data: Raw bytes of the image
            **kwargs: Additional implementation-specific parameters
                - mime_type: MIME type of the input (default: auto-detected)
                - timeout: Override the default timeout in seconds
                
        Returns:
            Dict[str, Any]: The Document AI OCR result as a JSON-serializable dictionary
        """
        # Determine MIME type
        mime_type = kwargs.get('mime_type')
        if not mime_type:
            # Try to auto-detect MIME type
            mime_type = self._detect_mime_type(image_data)
        
        # Get timeout from kwargs or use default
        timeout = kwargs.get('timeout', self.timeout)
        
        # Create raw document
        raw_document = documentai.RawDocument(
            content=image_data,
            mime_type=mime_type
        )
        
        # Create process request
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document,
            skip_human_review=True
        )
        
        # Process the document
        result = self.client.process_document(request=request, timeout=timeout)
        
        # Convert to dictionary for easier processing
        result_dict = MessageToDict(result._pb)
        
        # MessageToDict already ensures JSON serializable output, but verify and sanitize if needed
        def sanitize_for_json(obj):
            if isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(i) for i in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Convert any non-JSON serializable types to strings
                return str(obj)
        
        # Sanitize the result to ensure it's JSON serializable
        json_safe_result = sanitize_for_json(result_dict)
        self._last_result = json_safe_result
        
        return json_safe_result
    
    def convert_to_canonical(self, native_result: Dict[str, Any]) -> OcrResult:
        """
        Convert the Document AI result to canonical schema format.
        
        Args:
            native_result: Document AI result in its native dictionary format
                         (JSON-serializable dictionary)
            
        Returns:
            OcrResult: Results in canonical schema format
        """
        self._last_result = native_result
        return process_documentai_result(native_result)
    
    def get_native_result(self) -> Any:
        """
        Return the native result from the most recent OCR execution.
        
        Returns:
            Dict: The Document AI OCR result in its native dictionary format
        """
        return self._last_result
    
    def get_implementation_info(self) -> Dict[str, str]:
        """
        Return information about this OCR implementation.
        
        Returns:
            Dict[str, str]: Information about the implementation
        """
        return {
            "name": "Google Document AI",
            "version": "1.0.0",
            "description": "Google Cloud Document AI OCR engine"
        }
    
    def _detect_mime_type(self, data: bytes) -> str:
        """
        Detect MIME type from image data.
        
        Args:
            data: Raw bytes of the image
            
        Returns:
            str: MIME type (e.g., 'application/pdf', 'image/jpeg')
        """
        import magic
        try:
            # Try to use python-magic if available
            mime_type = magic.from_buffer(data, mime=True)
            return mime_type
        except ImportError:
            # Fall back to basic detection
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
                # Default to PDF if unable to detect
                return 'application/pdf'