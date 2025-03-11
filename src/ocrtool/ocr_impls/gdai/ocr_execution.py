import json
import os
import asyncio
from typing import Optional

from globalog import LOG
from jserpy import serialize_json, serialize_json_as_dict
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.oauth2 import service_account
from google.protobuf.json_format import MessageToJson, MessageToDict

from ocrtool import OcrResult
from ocrtool.ocr_impls.gdai.gdai_convert import process_documentai_result


class GoogleCloudDocumentOCR:
    def __init__(
        self,
        prediction_endpoint: str,
        credentials: service_account.Credentials,
    ):
        """
        Initialize the OCR class.

        Args:
            prediction_endpoint: The full endpoint for your Document AI processor.
            credentials: An instance of google.oauth2.service_account.Credentials.
        """
        self.client = documentai.DocumentProcessorServiceClient(
            credentials=credentials,
            # client_options=ClientOptions(api_endpoint="https://us-documentai.googleapis.com/v1")
            client_options=ClientOptions(api_endpoint="us-documentai.googleapis.com")
        )
        self.processor_name = prediction_endpoint

    async def execute_ocr(self, pdf_bytes: bytes) -> OcrResult:
        """
        Asynchronously executes OCR on the given PDF bytes using Document AI,
        converts the vendor result to an internal OcrResult, and returns it.

        Args:
            pdf_bytes: The raw bytes of the PDF to process.

        Returns:
            An internal OcrResult.
        """
        raw_document = documentai.RawDocument(
            content=pdf_bytes,
            mime_type="application/pdf"
        )
        request = documentai.ProcessRequest(name=self.processor_name, raw_document=raw_document, skip_human_review=True)

        # Offload the blocking API call to a separate thread.
        # result = await asyncio.to_thread(self.client.process_document, request=request)
        LOG.info("process doc")
        result = self.client.process_document(request=request)

        LOG.info("serialize result")
        # j_result = MessageToJson(result._pb)
        d_result = MessageToDict(result._pb)


        LOG.info("convert to unified format")
        ocr_result = process_documentai_result(d_result)
        LOG.info("done")

        # Build a GVOcrResult from the Document AI result.
        # gv_result = GVOcrResult(pages=result.document.pages)

        # Use the conversion function to convert the vendor result
        # to an internal OcrResult.
        # ocr_result = await convert_gv_ocr_to_internal_ocr(gv_result)
        return ocr_result


if __name__ == '__main__':
    async def main():
        # Load your PDF bytes from file (or any other source).
        # path = "/Users/ronp/data/test-docs/feec4aa20a8ac5c78665eac2f3628f1a3e1138e9.pdf"
        path = "/Users/ronp/Documents/בהצדעה-כרטיסים-לקרקס.pdf"
        with open(path, "rb") as f:
            pdf_bytes = f.read()

        # Instantiate using the default_document_ocr helper and a FirebaseEnv value.
        ocr = default_document_ocr(FirebaseEnv.DEV)
        result = await ocr.execute_ocr(pdf_bytes)
        print(json.dumps(serialize_json_as_dict(result), indent=2))

    asyncio.run(main())