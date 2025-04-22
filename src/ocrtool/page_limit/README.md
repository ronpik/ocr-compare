# Page Limit Handler

This package provides functionality to detect and handle page limit errors encountered during OCR processing.

## Purpose
Many OCR engines impose a limit on the number of pages that can be processed in a single document. When this limit is exceeded, a specific error is raised. The `PageLimitHandler` class detects such errors and provides a mechanism to handle them gracefully.

## Usage
- The handler is integrated into the `ExternalOcrExecutor` class.
- By default, page limit errors are detected and a custom exception is raised.
- You can extend the `handle` method in `PageLimitHandler` to implement custom logic, such as splitting the document and retrying, or returning a partial result.

## Extending
To implement custom handling, override the `handle` method in `PageLimitHandler`:

```python
class MyPageLimitHandler(PageLimitHandler):
    def handle(self, exc, image_data, executor, **kwargs):
        # Custom logic here
        ...
```

## Exception
- `PageLimitExceededError` is raised when a document exceeds the allowed page limit.

## Tests

Unit tests for the page limit handler are provided in `tests/page_limit/test_page_limit_handler.py`.

Run tests with:

```
pytest tests/page_limit/
```

## Proactive Page Limit Enforcement

- Each OCR executor now has a `page_limit` property (int or None).
- If set, the executor will check the number of pages in a PDF before attempting OCR.
- If the limit is exceeded, a `PageLimitExceededError` is raised before any engine call.
- For non-PDFs (images), the check is skipped.

## Utility Functions

- `is_pdf(data: bytes) -> bool`: Returns True if the data is a PDF file.
- `count_pdf_pages(pdf_bytes: bytes) -> int`: Returns the number of pages in a PDF (requires PyPDF2).

## Configuration

- The `page_limit` can be set per executor (default is engine-specific or None).
- To disable the check, set `page_limit = None` on the executor instance or subclass.

## Example

```python
from ocrtool.ocr_impls.gdai.gdai_executor import GoogleDocumentAIOcrExecutor
ocr = GoogleDocumentAIOcrExecutor()
ocr.page_limit = 10  # Set a custom limit
``` 