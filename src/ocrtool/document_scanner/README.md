# DocumentScanner

A modular class for document edge detection, alignment, and scan-like enhancement using OpenCV and numpy.

## Features
- Detect document edges in an image
- Align (warp) the document using perspective transform
- Enhance the aligned document to look like a scanned document
- Run each step separately or the full pipeline
- Control enhancement parameters

## Installation
Make sure you have the required dependencies:

```
pip install opencv-python numpy
```

## Usage
```python
from ocrtool.document_scanner import DocumentScanner
import cv2

scanner = DocumentScanner()
image = cv2.imread('input.jpg')

# Step 1: Detect edges
edges = scanner.detect_edges(image)

# Step 2: Align document
aligned = scanner.align_document(image)

# Step 3: Enhance (scan-like)
enhanced = scanner.enhance(
    aligned,
    upscale_factor=2.0,
    block_size=15,
    threshold_C=2.0,
    close_kernel=3,
    median_ksize=3,
)

# Or run the full pipeline in one call:
scanned = scanner.scan_document(image)
```

## API Reference
### DocumentScanner
#### detect_edges(image: np.ndarray) -> Optional[np.ndarray]
Detects document edges using Canny edge detection.

#### align_document(image: np.ndarray) -> Optional[np.ndarray]
Detects the document contour and applies a perspective transform to align it.

#### enhance(
    image: np.ndarray,
    upscale_factor: float = 2.0,
    block_size: int = 15,
    threshold_C: float = 2.0,
    close_kernel: int = 3,
    median_ksize: int = 3,
) -> np.ndarray
Enhances the aligned document image to look like a scanned document. All parameters are optional and control the enhancement pipeline.

#### scan_document(
    image: np.ndarray,
    upscale_factor: float = 2.0,
    block_size: int = 15,
    threshold_C: float = 2.0,
    close_kernel: int = 3,
    median_ksize: int = 3,
) -> Optional[np.ndarray]
Runs the full pipeline: aligns and enhances the document.

## Parameters
- `upscale_factor` (float): Upscale factor for resizing (default: 2.0)
- `block_size` (int): Block size for adaptive thresholding (odd number, default: 15)
- `threshold_C` (float): Constant subtracted in adaptive thresholding (default: 2.0)
- `close_kernel` (int): Kernel size for morphological closing (default: 3)
- `median_ksize` (int): Kernel size for median blur (default: 3)

## Error Handling
- Raises `ValueError` if input image is None.
- Returns `None` if document alignment fails.

## License
MIT 