import fitz  # PyMuPDF
import numpy as np
from typing import Tuple, Optional
import cv2

from PIL import ImageFont

def get_average_font_size(pdf_path: str, page_num: int = 0) -> float:
    """
    Calculate the average font size from a PDF page.

    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number to analyze.

    Returns:
        float: Average font size on the page.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        text = page.get_text("dict")
        sizes = [block['lines'][0]['spans'][0]['size'] for block in text['blocks'] if block['type'] == 0]
        return sum(sizes) / len(sizes) if sizes else 12  # Default to 12 if no text is found
    finally:
        doc.close()

def calculate_bounding_box(page_width: float, page_height: float, font_size: float, location: str) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box for text placement.

    Args:
        page_width (float): Width of the page.
        page_height (float): Height of the page.
        font_size (float): Font size of the text.
        location (str): 'top' or 'bottom'.

    Returns:
        Tuple[float, float, float, float]: Bounding box (x0, y0, x1, y1).
    """
    margin = 10  # Small margin for padding
    box_height = font_size * 1.5  # Adjust based on text line height

    if location == "top":
        return margin, margin, page_width - margin, margin + box_height
    elif location == "bottom":
        return margin, page_height - box_height - margin, page_width - margin, page_height - margin
    else:
        raise ValueError("Invalid location. Choose 'top' or 'bottom'.")

def add_text_to_pdf(pdf_path: str, output_path: str, text: str, location: str, font_size: Optional[float] = None):
    """
    Add text to a PDF at a specified location.

    Args:
        pdf_path (str): Path to the input PDF.
        output_path (str): Path to save the modified PDF.
        text (str): Text to add.
        location (str): 'top' or 'bottom'.
        font_size (Optional[float]): Font size for the text. If None, extract from the document.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        page = doc[0]  # First page
        if font_size is None:
            font_size = get_average_font_size(pdf_path)
        bbox = calculate_bounding_box(page.rect.width, page.rect.height, font_size, location)

        page.insert_text(
            fitz.Point(bbox[0], bbox[1]),  # Top-left corner of the bounding box
            text,
            fontsize=font_size,
            color=(0, 0, 0)  # Black color
        )
        doc.save(output_path)
    finally:
        doc.close()


def detect_bounding_box(pdf_path: str, location: str = "top", page_num: int = 0, threshold: int = 200,
                        add_text: bool = False) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect a bounding box based on whitespace at the top or bottom of a PDF document and optionally add text.

    Args:
        pdf_path (str): Path to the PDF file.
        location (str): Location to detect bounding box ('top' or 'bottom'). Default is 'top'.
        page_num (int): Page number to process (0-based index). Default is 0 (first page).
        threshold (int): Pixel value threshold for considering a pixel as white. Default is 200.
        add_text (bool): If True, add "PDF Seen" text to the detected bounding box. Default is False.

    Returns:
        Optional[Tuple[float, float, float, float]]: Bounding box coordinates (x0, y0, x1, y1) or None if not found.
    """
    if location not in ["top", "bottom"]:
        raise ValueError("Location must be either 'top' or 'bottom'")

    doc = fitz.open(pdf_path)

    try:
        page = doc[page_num]
        resolution_parameter = 300
        pix = page.get_pixmap(dpi=resolution_parameter)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if location == "top":
            # Find the first non-white row from top
            target_row = np.argmax(np.any(gray < threshold, axis=1))
            if target_row == 0:
                print("No top whitespace detected.")
                return None
            y0, y1 = 0, target_row / pix.h * page.rect.height
        else:  # bottom
            # Find the last non-white row from bottom
            target_row = pix.h - 1 - np.argmax(np.any(gray[::-1] < threshold, axis=1))
            if target_row == pix.h - 1:
                print("No bottom whitespace detected.")
                return None
            y0, y1 = target_row / pix.h * page.rect.height, page.rect.height

        # Find the leftmost and rightmost non-white pixels in the target row
        non_white_cols = np.where(gray[target_row] < threshold)[0]
        left_col = non_white_cols[0]
        right_col = non_white_cols[-1]

        # Convert pixel coordinates to PDF coordinates
        x0 = left_col / pix.w * page.rect.width
        x1 = right_col / pix.w * page.rect.width

        if add_text:
            # Create a copy of the PDF document that we can modify
            modified_pdf = fitz.open()
            new_page = modified_pdf.new_page(width=page.rect.width, height=page.rect.height)

            # Copy the content of the original page to the new page
            new_page.show_pdf_page(new_page.rect, doc, page_num)

            # Get the average font size from the document
            avg_font_size = get_average_font_size(pdf_path, page_num)

            # Add "PDF Seen" text to the detected bounding box
            text_rect = fitz.Rect(x0, y0, x1, y1)
            text_color = (1, 0, 0)  # Red color
            text = "PDF Seen"

            # Use the average font size, but ensure it fits within the box
            box_width = x1 - x0
            box_height = y1 - y0
            # Scale down the font size if it's too large for the box
            font_size = min(avg_font_size, box_width / len(text) * 1.5, box_height * 0.8)

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
        doc.close()



if __name__ == "__main__":
    # Example usage
    pdf_path = "PDF/test.pdf"
    # top_box = detect_bounding_box(pdf_path, location="top", add_text=True)
    # if top_box:
    #     print(f"Detected top bounding box: {top_box}")

    # Detect bottom bounding box
    bottom_box = detect_bounding_box(pdf_path, location="bottom", add_text=True)
    if bottom_box:
        print(f"Detected bottom bounding box: {bottom_box}")