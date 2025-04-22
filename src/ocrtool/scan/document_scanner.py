"""
DocumentScanner: A class for document edge detection, alignment, and scan-like enhancement.

Usage Example:
    from ocrtool.document_scanner import DocumentScanner
    import cv2
    
    scanner = DocumentScanner()
    image = cv2.imread('input.jpg')
    edges = scanner.detect_edges(image)
    aligned = scanner.align_document(image)
    enhanced = scanner.enhance(aligned, upscale_factor=2.0, block_size=15, threshold_C=2, close_kernel=3, median_ksize=3)

You can run each step separately, or the full pipeline with `scan_document`.
"""
from typing import Optional, Tuple
import numpy as np
import cv2


def image_bytes_to_ndarray(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Convert image bytes (e.g., from open(path, 'rb').read()) to a numpy ndarray suitable for DocumentScanner.

    Args:
        image_bytes (bytes): The image file bytes.

    Returns:
        Optional[np.ndarray]: The decoded image as a numpy ndarray (BGR), or None if decoding fails.
    """
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


def ndarray_to_image_bytes(image: np.ndarray, ext: str = '.png') -> Optional[bytes]:
    """
    Convert a numpy ndarray (BGR image) to image bytes (PNG by default).

    Args:
        image (np.ndarray): The image array (BGR).
        ext (str): The file extension/format for encoding (default: '.png').

    Returns:
        Optional[bytes]: The encoded image bytes, or None if encoding fails.
    """
    success, buf = cv2.imencode(ext, image)
    if not success:
        return None
    return buf.tobytes()


class DocumentScanner:
    """Class for document edge detection, alignment, and scan-like enhancement."""

    @staticmethod
    def prepare_input(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Convert image bytes to a numpy ndarray suitable for DocumentScanner processing.

        Args:
            image_bytes (bytes): The image file bytes.

        Returns:
            Optional[np.ndarray]: The decoded image as a numpy ndarray (BGR), or None if decoding fails.
        """
        return image_bytes_to_ndarray(image_bytes)

    def detect_edges(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects document edges in the image using Canny edge detection.

        Args:
            image (np.ndarray): Input image (BGR).

        Returns:
            Optional[np.ndarray]: Edge map or None if detection fails.
        """
        if image is None:
            raise ValueError("Input image is None.")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        return edged

    def align_document(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects the document contour and applies a perspective transform to align it.

        Args:
            image (np.ndarray): Input image (BGR).

        Returns:
            Optional[np.ndarray]: Warped (aligned) document image, or None if not found.
        """
        if image is None:
            raise ValueError("Input image is None.")
        orig = image.copy()
        ratio = 500.0 / image.shape[0]
        resized = cv2.resize(image, (int(image.shape[1] * ratio), 500))
        orig_ratio = orig.shape[0] / float(resized.shape[0])
        edged = self.detect_edges(resized)
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
            return None
        pts = screenCnt.reshape(4, 2) * orig_ratio
        warped = self._four_point_transform(orig, pts)
        return warped

    def enhance(
        self,
        image: np.ndarray,
        upscale_factor: float = 2.0,
        block_size: int = 15,
        threshold_C: float = 2.0,
        close_kernel: int = 3,
        median_ksize: int = 3,
    ) -> np.ndarray:
        """
        Enhances the aligned document image to look like a scanned document.

        Args:
            image (np.ndarray): Aligned document image (BGR).
            upscale_factor (float): Upscale factor for resizing.
            block_size (int): Block size for adaptive thresholding (odd number).
            threshold_C (float): Constant subtracted in adaptive thresholding.
            close_kernel (int): Kernel size for morphological closing.
            median_ksize (int): Kernel size for median blur.

        Returns:
            np.ndarray: Enhanced (scan-like) image.
        """
        if image is None:
            raise ValueError("Input image is None.")
        if upscale_factor != 1.0:
            image = self._upscale_image(image, upscale_factor)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, threshold_C
        )
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((close_kernel, close_kernel), np.uint8))
        final = cv2.medianBlur(closed, median_ksize)
        return final

    def scan_document(
        self,
        image: np.ndarray,
        upscale_factor: float = 2.0,
        block_size: int = 15,
        threshold_C: float = 2.0,
        close_kernel: int = 3,
        median_ksize: int = 3,
    ) -> Optional[np.ndarray]:
        """
        Full pipeline: aligns and enhances the document.

        Args:
            image (np.ndarray): Input image (BGR).
            upscale_factor (float): Upscale factor for resizing.
            block_size (int): Block size for adaptive thresholding (odd number).
            threshold_C (float): Constant subtracted in adaptive thresholding.
            close_kernel (int): Kernel size for morphological closing.
            median_ksize (int): Kernel size for median blur.

        Returns:
            Optional[np.ndarray]: Enhanced (scan-like) image, or None if alignment fails.
        """
        aligned = self.align_document(image)
        if aligned is None:
            return None
        return self.enhance(
            aligned,
            upscale_factor=upscale_factor,
            block_size=block_size,
            threshold_C=threshold_C,
            close_kernel=close_kernel,
            median_ksize=median_ksize,
        )

    @staticmethod
    def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Performs a perspective transform using four points.

        Args:
            image (np.ndarray): Input image.
            pts (np.ndarray): Array of 4 points.

        Returns:
            np.ndarray: Warped image.
        """
        rect = DocumentScanner._order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """
        Orders 4 points as top-left, top-right, bottom-right, bottom-left.

        Args:
            pts (np.ndarray): Array of 4 points.

        Returns:
            np.ndarray: Ordered points.
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _upscale_image(image: np.ndarray, factor: float) -> np.ndarray:
        """
        Upscales the image by the given factor using bicubic interpolation.

        Args:
            image (np.ndarray): Input image.
            factor (float): Upscale factor.

        Returns:
            np.ndarray: Upscaled image.
        """
        new_width = int(image.shape[1] * factor)
        new_height = int(image.shape[0] * factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC) 