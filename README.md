# OCR Compare

A Python library for running multiple OCR engines on documents and comparing the results. The library provides a unified interface to different OCR implementations using the Strategy design pattern.

## Features

- Unified interface to multiple OCR engines (Tesseract, Google Document AI, etc.)
- Canonical OCR schema for standardized result representation
- Ability to compare OCR results from different engines
- Easy access to both canonical and native output formats
- Extensible architecture - add new OCR engines by implementing the `OcrExecutor` interface

## Installation

```bash
# Basic installation
pip install -e .

# Install with Tesseract OCR support
pip install -e ".[tesseract]"

# Install with Google Document AI support
pip install -e ".[gdai]"

# Install with visualization tools
pip install -e ".[visualization]"

# Install with all OCR engines and tools
pip install -e ".[all]"

# Install with development tools
pip install -e ".[dev]"
```

## Usage

### Basic Example

```python
from pathlib import Path
from ocrtool import execute_ocr, list_available_engines

# List available OCR engines
engines = list_available_engines()
print(f"Available engines: {', '.join(engines.keys())}")

# Load an image
with open("test_image.png", "rb") as f:
    image_data = f.read()

# Run OCR with a specific engine
result = execute_ocr(image_data, engine="tesseract")

# Get the extracted text
text = result.document.text()
print(f"Extracted text: {text}")

# Access page, block, paragraph, line, and word information
for page in result.document.pages:
    print(f"Page {page.page_no}, confidence: {page.confidence:.2f}")
    for block in page.blocks:
        print(f"  Block {block.block_no}, type: {block.blockType}")
        print(f"  Text: {block.text()}")
```

### Using Google Document AI

To use Google Document AI, you'll need to set up authentication and provide a processor ID:

```python
from pathlib import Path
from ocrtool.ocr_impls.gdai import GoogleDocumentAIOcrExecutor

# Set up configuration with authentication
config = {
    "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
    "service_account_file": "/path/to/service-account-key.json",
    # OR use default credentials (ADC) - no service_account_file needed
}

# Create the executor
gdai_executor = GoogleDocumentAIOcrExecutor(config=config)

# Process a document
with open("document.pdf", "rb") as f:
    document_data = f.read()

result = gdai_executor.execute_ocr(document_data)
print(f"Extracted text: {result.document.text()[:100]}...")
```

You can also use the factory pattern:

```python
from ocrtool import execute_ocr

# Load document data
with open("document.pdf", "rb") as f:
    document_data = f.read()

# Set up config
config = {
    "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
    "service_account_file": "/path/to/service-account-key.json",
}

# Execute OCR using the factory
result = execute_ocr(document_data, engine="gdai", engine_config=config)
```

### Comparing OCR Results

```python
from ocrtool import execute_ocr, compare_ocr_results

# Load image data
with open("test_image.png", "rb") as f:
    image_data = f.read()

# Set up configs for each engine
gdai_config = {
    "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
    # Authentication handled through environment variable or explicitly
}

tesseract_config = {
    "lang": "eng"
}

# Execute OCR with multiple engines
results = {
    "tesseract": execute_ocr(image_data, engine="tesseract", engine_config=tesseract_config),
    "gdai": execute_ocr(image_data, engine="gdai", engine_config=gdai_config),
}

# Compare results
comparison = compare_ocr_results(results)

# Print comparison metrics
print("Confidence scores:")
for engine, score in comparison["confidence_scores"].items():
    print(f"  {engine}: {score:.2f}")

print("\nText extraction character count:")
for engine, text in comparison["text_extraction"].items():
    print(f"  {engine}: {len(text)} characters")
```

## Example Scripts

The library comes with several example scripts:

- `basic_usage.py`: Simple demonstration of using a single OCR engine
- `gdai_example.py`: Example of using Google Document AI with authentication
- `ocr_comparison.py`: Advanced example comparing multiple OCR engines with visualizations

To run the OCR comparison script:

```bash
python ocr_comparison.py document.png --use-tesseract --use-gdai \
  --project-id your-project-id --processor-id your-processor-id \
  --service-account path/to/service-account.json --visualize
```

## Implementing a New OCR Engine

To add a new OCR engine, implement the `OcrExecutor` interface:

```python
from ocrtool.ocr_impls.ocr_executor import OcrExecutor
from ocrtool.ocr_impls.ocr_factory import OcrExecutorFactory
from ocrtool.canonical_ocr.ocr_schema import OcrResult

class MyCustomOcrExecutor(OcrExecutor):
    def __init__(self, config=None):
        self.config = config or {}
        self._last_result = None
    
    def execute_ocr(self, image_data, **kwargs):
        # Execute OCR using your custom engine
        native_result = my_ocr_engine.process(image_data)
        self._last_result = native_result
        
        # Convert to canonical format
        return self._convert_to_canonical()
    
    def get_native_result(self):
        return self._last_result
    
    def get_implementation_info(self):
        return {
            "name": "My Custom OCR",
            "version": "1.0.0",
            "description": "Custom OCR engine implementation"
        }
    
    def _convert_to_canonical(self):
        # Convert native result to canonical format
        # ...

# Register your implementation
OcrExecutorFactory.register("my-ocr", MyCustomOcrExecutor)
```

## Authentication for Google Document AI

The Google Document AI implementation supports several authentication methods:

1. **Application Default Credentials (ADC)**
   - Run `gcloud auth application-default login` on your local machine
   - In GCP environments (Compute Engine, Cloud Run, etc.), this works automatically

2. **Service Account Key File**
   ```python
   config = {
       "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
       "service_account_file": "/path/to/service-account-key.json",
   }
   ```

3. **Service Account Info Dictionary**
   ```python
   import json
   
   # Load service account info from file or environment
   with open("/path/to/service-account-key.json") as f:
       service_account_info = json.load(f)
   
   config = {
       "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
       "service_account_info": service_account_info,
   }
   ```

4. **Direct Credentials Object**
   ```python
   from google.oauth2 import service_account
   
   # Create credentials object yourself
   credentials = service_account.Credentials.from_service_account_file(
       "/path/to/service-account-key.json"
   )
   
   config = {
       "processor_name": "projects/your-project-id/locations/us/processors/your-processor-id",
       "credentials": credentials,
   }
   ```

## License

MIT
