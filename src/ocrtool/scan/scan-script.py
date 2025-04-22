"""
Document Scanner Script

This script takes an image or PDF file (e.g., a photo of a document captured by a phone camera
or a PDF document) and converts it into a scanned-like version by:
  1. Converting a PDF (first page) to PNG if needed.
  2. Detecting the document edges and applying a four-point perspective transform ("wrapping").
  3. Enhancing the warped image using an enhanced processing pipeline (upscaling, denoising,
     adaptive thresholding, morphological closing, optional median blur).
  4. Writing the final result.
Additionally, if the --debug_dir flag is provided, intermediate results (the warped and enhanced images)
will be saved into that folder.
You can also run a parameter grid (experiment) over the enhancement parameters using --experiment_grid.

Usage:
    python document_scanner.py --input path_to_input_file --output scanned_output.jpg
           [--upscale 2.0 --block_size 15 --C 2 --close_kernel 3 --median_ksize 3]
           [--debug_dir debug_folder]
           [--experiment_grid]
"""

from pathlib import Path
import cv2
import numpy as np
import argparse
import os
import tempfile
from typing import Optional, Tuple
import itertools
import csv

# Optional: using imutils makes image resizing a bit simpler.
try:
    import imutils
except ImportError:
    imutils = None


# ----------------- Basic Utility Functions -----------------
def read_image_file(path: Path) -> Optional[np.ndarray]:
    """Read an image from a file."""
    image = cv2.imread(str(path))
    if image is None:
        print(f"Error: Could not read image from {path}")
    return image

def write_image_file(image: np.ndarray, path: Path) -> None:
    """Write an image to file."""
    cv2.imwrite(str(path), image)
    print(f"Image saved to: {path}")

# ----------------- Document Wrapping Functions -----------------
def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders a list of 4 points in the order:
    top-left, top-right, bottom-right, bottom-left

    Args:
        pts (np.ndarray): Array of 4 points.

    Returns:
        np.ndarray: Array of ordered points.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # The top-left point will have the smallest sum, whereas the bottom-right will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # The top-right point will have the smallest difference and the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Performs a perspective transform of the given image using the four points.

    Args:
        image (np.ndarray): Input image.
        pts (np.ndarray): Array of 4 points defining the document.

    Returns:
        np.ndarray: The warped (transformed) image.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image: maximum of the distance between bottom and top x-coordinates.
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    # Compute the height of the new image: maximum of the distance between the y-coordinates.
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    # Set up destination points for the "birds eye view"
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(order_points(pts), dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def wrap_document(image: np.ndarray, debug_dir: Optional[Path] = None) -> Optional[np.ndarray]:
    """
    Detects the document contour in the image and applies a perspective transform.
    Optionally saves the intermediate warped image into debug_dir.
    """
    orig = image.copy()
    # Resize image for contour detection
    if imutils:
        resized = imutils.resize(image, height=500)
    else:
        ratio = 500.0 / image.shape[0]
        resized = cv2.resize(image, (int(image.shape[1]*ratio), 500))
    orig_ratio = orig.shape[0] / float(resized.shape[0])

    # Edge detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    print("Edge detection complete in wrap_document.")

    # Find contours
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    screenCnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        print("Error: Document contour not found.")
        return None
    pts = screenCnt.reshape(4, 2) * orig_ratio
    warped = four_point_transform(orig, pts)

    # Save intermediate warped image if debug_dir is given.
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        warped_path = debug_dir / "warped.png"
        write_image_file(warped, warped_path)

    return warped

# ----------------- Enhancement Functions -----------------
def apply_gaussian_blur(image: np.ndarray, kernel_size: Tuple[int, int]=(5, 5)) -> np.ndarray:
    """Applies Gaussian blur."""
    return cv2.GaussianBlur(image, kernel_size, 0)

def adaptive_thresholding(image: np.ndarray, block_size: int, C: float) -> np.ndarray:
    """Applies adaptive thresholding using Gaussian weighted method."""
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

def morphological_closing(image: np.ndarray, kernel_size: Tuple[int, int]=(3, 3)) -> np.ndarray:
    """Performs morphological closing."""
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

def apply_median_blur(image: np.ndarray, ksize: int=3) -> np.ndarray:
    """Applies median blur."""
    return cv2.medianBlur(image, ksize)

def enhanced_processing(image: np.ndarray, upscale_factor: float, block_size: int,
                        threshold_C: float, close_kernel: int, median_ksize: int) -> np.ndarray:
    """
    Enhances the image using upscaling, Gaussian blur, adaptive thresholding,
    morphological closing, and median blur.
    """
    if upscale_factor != 1.0:
        image = upscale_image(image, upscale_factor)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(gray, (5, 5))
    thresh = adaptive_thresholding(blurred, block_size, threshold_C)
    closed = morphological_closing(thresh, (close_kernel, close_kernel))
    final = apply_median_blur(closed, median_ksize)
    return final

def upscale_image(image: np.ndarray, factor: float) -> np.ndarray:
    """Upscales the image by the given factor using bicubic interpolation."""
    new_width = int(image.shape[1] * factor)
    new_height = int(image.shape[0] * factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

def enhance_document(image: np.ndarray, upscale_factor: float, block_size: int,
                     threshold_C: float, close_kernel: int, median_ksize: int,
                     debug_dir: Optional[Path] = None) -> np.ndarray:
    """
    Enhances the wrapped image using enhanced processing steps. Optionally writes the
    intermediate enhanced result to the debug folder.
    """
    processed = enhanced_processing(image, upscale_factor, block_size, threshold_C, close_kernel, median_ksize)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        enhanced_path = debug_dir / "enhanced.png"
        write_image_file(processed, enhanced_path)
    return processed

# ----------------- Grid Experiment Functions -----------------
def run_parameter_grid(warped: np.ndarray, output_dir: Path,
                       upscale_factors: Tuple[float, ...] = (1.0, 1.5, 2.0),
                       block_sizes: Tuple[int, ...] = (11, 15, 21),
                       Cs: Tuple[float, ...] = (2, 4),
                       close_kernels: Tuple[int, ...] = (3, 5),
                       median_ksizes: Tuple[int, ...] = (3, 5)) -> None:
    """
    Runs enhanced_processing over a grid of parameter combinations.
    Saves each output image into output_dir and writes a CSV mapping file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = output_dir / "parameter_mapping.csv"
    with mapping_file.open('w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "upscale_factor", "block_size", "C", "close_kernel", "median_ksize"])
        for upscale, block_size, C, close_kernel, median_ksize in itertools.product(
            upscale_factors, block_sizes, Cs, close_kernels, median_ksizes):
            processed = enhanced_processing(warped, upscale, block_size, C, close_kernel, median_ksize)
            filename = f"scanned_up{upscale}_bs{block_size}_C{C}_ck{close_kernel}_mb{median_ksize}.png"
            file_path = output_dir / filename
            cv2.imwrite(str(file_path), processed)
            writer.writerow([filename, upscale, block_size, C, close_kernel, median_ksize])
    print(f"Grid experiment completed. Results saved in {output_dir}; mapping in {mapping_file}")

def run_grid_experiment_on_input(image_path: Path, output_grid_dir: Path,
                                 debug_dir: Optional[Path] = None,
                                 upscale_factors: Tuple[float, ...] = (1.0, 2.0, 3.0),
                                 block_sizes: Tuple[int, ...] = (11, 15, 21),
                                 Cs: Tuple[float, ...] = (0, 2, 4),
                                 close_kernels: Tuple[int, ...] = (3, 5),
                                 median_ksizes: Tuple[int, ...] = (3, 5)) -> None:
    """
    Runs the grid experiment on a given image:
      - Reads the image.
      - Applies the wrap_document() function.
      - Saves the warped intermediate image (if debug_dir given).
      - Runs run_parameter_grid() on the warped image.
    """
    image = read_image_file(image_path)
    if image is None:
        return
    warped = wrap_document(image, debug_dir)
    if warped is None:
        return
    # Optionally, save the warped image to the output grid folder:
    temp_warped_path = output_grid_dir / "warped.png"
    output_grid_dir.mkdir(parents=True, exist_ok=True)
    write_image_file(warped, temp_warped_path)
    run_parameter_grid(warped, output_grid_dir)

# ----------------- Document Processing Pipeline -----------------
def process_document(image_path: Path, final_output_path: Path,
                     upscale_factor: float, block_size: int, threshold_C: float,
                     close_kernel: int, median_ksize: int,
                     debug_dir: Optional[Path] = None) -> None:
    """
    Full pipeline:
      1. Read image.
      2. Wrap the document (detect and transform contour).
      3. Enhance the wrapped image.
      4. Write the final result.
    Intermediate results are saved into debug_dir if provided.
    """
    image = read_image_file(image_path)
    if image is None:
        return
    warped = wrap_document(image, debug_dir)
    if warped is None:
        return
    final = enhance_document(warped, upscale_factor, block_size, threshold_C, close_kernel, median_ksize, debug_dir)
    write_image_file(final, final_output_path)

def pdf_to_png(pdf_path: Path, png_output_path: Path) -> None:
    """
    Converts the first page of a PDF file to PNG.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        print("Error: pdf2image library not found. Please install it using 'pip install pdf2image'.")
        return
    pages = convert_from_path(pdf_path, dpi=300)
    if not pages:
        print("Error: No pages found in PDF.")
        return
    pages[0].save(png_output_path, 'PNG')
    print("Converted PDF to image:", png_output_path)

def generate_output_path(input_path: Path) -> Path:
    """
    Generates an output path based on the input file.
    """
    suffix = input_path.suffix.strip('.')
    if suffix.lower() == 'pdf':
        suffix = 'png'
    return input_path.parent / f"{input_path.stem}-scanned.{suffix}"

# ----------------- Main Function -----------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert a document file (image or PDF) into a scanned-like version with intermediate debugging options and/or grid parameter experiments."
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to the input file (PDF or image)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path for the final output scanned image")
    parser.add_argument("--upscale", type=float, default=2.0,
                        help="Upscale factor for enhanced processing (default=2.0)")
    parser.add_argument("--block_size", type=int, default=15,
                        help="Block size for adaptive thresholding (odd number, default=15)")
    parser.add_argument("--C", type=float, default=2,
                        help="Constant subtracted in adaptive thresholding (default=2)")
    parser.add_argument("--close_kernel", type=int, default=3,
                        help="Kernel size for morphological closing (default=3)")
    parser.add_argument("--median_ksize", type=int, default=3,
                        help="Kernel size for median blur (default=3)")
    parser.add_argument("--debug_dir", type=Path, default=None,
                        help="Path to a folder where intermediate results are saved")
    parser.add_argument("--experiment_grid", action="store_true",
                        help="If set, run a grid experiment over enhancement parameters instead of normal processing")
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or generate_output_path(args.input)

    # If PDF, convert first page to PNG.
    if input_path.suffix.lower().endswith(".pdf"):
        temp_img_path = input_path.parent / f"{input_path.stem}.png"
        pdf_to_png(input_path, temp_img_path)
        input_path = temp_img_path

    # If grid experiment mode is enabled, run parameter grid experiment.
    if args.experiment_grid:
        # Specify an output folder for grid experiments.
        grid_output_dir = input_path.parent / "grid_experiment"
        run_grid_experiment_on_input(input_path, grid_output_dir, args.debug_dir)
    else:
        process_document(input_path, output_path,
                         upscale_factor=args.upscale,
                         block_size=args.block_size,
                         threshold_C=args.C,
                         close_kernel=args.close_kernel,
                         median_ksize=args.median_ksize,
                         debug_dir=args.debug_dir)

if __name__ == "__main__":
    main()
