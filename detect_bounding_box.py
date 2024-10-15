import fitz  # PyMuPDF
import numpy as np
from typing import Tuple, Optional

def detect_top_bounding_box(pdf_path: str, page_num: int = 0, threshold: int = 200) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect a bounding box based on whitespace at the top of a PDF document.

    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number to process (0-based index). Default is 0 (first page).
        threshold (int): Pixel value threshold for considering a pixel as white. Default is 200.

    Returns:
        Optional[Tuple[float, float, float, float]]: Bounding box coordinates (x0, y0, x1, y1) or None if not found.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)

    try:
        # Get the specified page
        page = doc[page_num]

        # Convert the page to a PNG image
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert to grayscale if the image is in color
        if img.shape[2] == 3:
            img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

        # Find the first non-white row
        first_non_white_row = np.argmax(np.any(img < threshold, axis=1))

        if first_non_white_row == 0:
            print("No top whitespace detected.")
            return None

        # Find the leftmost and rightmost non-white pixels in the first non-white row
        non_white_cols = np.where(img[first_non_white_row] < threshold)[0]
        left_col = non_white_cols[0]
        right_col = non_white_cols[-1]

        # Convert pixel coordinates to PDF coordinates
        x0 = left_col / pix.w * page.rect.width
        y0 = 0
        x1 = right_col / pix.w * page.rect.width
        y1 = first_non_white_row / pix.h * page.rect.height

        return (x0, y0, x1, y1)

    finally:
        # Close the document
        doc.close()

if __name__ == "__main__":
    # Example usage
    pdf_path = "path/to/your/document.pdf"
    bounding_box = detect_top_bounding_box(pdf_path)

    if bounding_box:
        print(f"Detected bounding box: {bounding_box}")
    else:
        print("No bounding box detected.")
