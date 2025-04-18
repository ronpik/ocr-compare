import numpy as np
import cv2
from typing import Optional

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
