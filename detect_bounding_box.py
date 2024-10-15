import fitz  # PyMuPDF
import numpy as np
from typing import Tuple, Optional
import cv2

def detect_top_bounding_box(pdf_path: str, page_num: int = 0, threshold: int = 200, add_text: bool = False) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect a bounding box based on whitespace at the top of a PDF document and optionally add text.

    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number to process (0-based index). Default is 0 (first page).
        threshold (int): Pixel value threshold for considering a pixel as white. Default is 200.
        add_text (bool): If True, add "PDF Seen" text to the detected bounding box. Default is False.

    Returns:
        Optional[Tuple[float, float, float, float]]: Bounding box coordinates (x0, y0, x1, y1) or None if not found.
    """
    # Open the PDF document
    doc = fitz.open(pdf_path)

    try:
        # Get the specified page
        page = doc[page_num]

        # Convert the page to a PNG image
        resolution_parameter = 300
        pix = page.get_pixmap(dpi=resolution_parameter)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # Convert to RGB if the image is in grayscale
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Create a grayscale version for detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the first non-white row
        first_non_white_row = np.argmax(np.any(gray < threshold, axis=1))

        if first_non_white_row == 0:
            print("No top whitespace detected.")
            return None

        # Find the leftmost and rightmost non-white pixels in the first non-white row
        non_white_cols = np.where(gray[first_non_white_row] < threshold)[0]
        left_col = non_white_cols[0]
        right_col = non_white_cols[-1]

        # Convert pixel coordinates to PDF coordinates
        x0 = left_col / pix.w * page.rect.width
        y0 = 0
        x1 = right_col / pix.w * page.rect.width
        y1 = first_non_white_row / pix.h * page.rect.height

        if add_text:
            # Create a copy of the PDF document that we can modify
            modified_pdf = fitz.open()
            new_page = modified_pdf.new_page(width=page.rect.width, height=page.rect.height)
            
            # Copy the content of the original page to the new page
            new_page.show_pdf_page(new_page.rect, doc, page_num)
            
            # Add "PDF Seen" text to the detected bounding box
            text_rect = fitz.Rect(x0, y0, x1, y1)
            text_color = (1, 0, 0)  # Red color
            
            # Calculate appropriate font size
            box_width = x1 - x0
            box_height = y1 - y0
            text = "PDF Seen"
            font_size = min(box_width / len(text) * 1.5, box_height * 0.8)
            
            # Center the text
            text_width, text_height = fitz.get_text_length(text, fontname="helv", fontsize=font_size), font_size
            text_x = x0 + (box_width - text_width) / 2
            text_y = y0 + (box_height - text_height) / 2 + text_height * 0.8  # Adjust for baseline
            
            new_page.insert_text((text_x, text_y), text, fontsize=font_size, color=text_color)
            
            # Save the modified PDF
            modified_pdf.save('modified_document.pdf')
            modified_pdf.close()

        return (x0, y0, x1, y1)

    finally:
        # Close the document
        doc.close()

if __name__ == "__main__":
    # Example usage
    pdf_path = "PDF/test.pdf"
    bounding_box = detect_top_bounding_box(pdf_path, add_text=True)

    if bounding_box:
        print(f"Detected bounding box: {bounding_box}")
    else:
        print("No bounding box detected.")
