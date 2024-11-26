import fitz  # PyMuPDF
import numpy as np
from typing import Tuple, Optional
import cv2
import lorem


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

def calculate_text_dimensions(text: str, font_size: float) -> Tuple[float, float]:
    """
    Calculate the width and height required for the given text and font size.

    Args:
        text (str): The text to measure.
        font_size (float): The font size.

    Returns:
        Tuple[float, float]: Width and height for the text.
    """
    char_width = font_size * 0.5  # Approximate width per character
    text_width = len(text) * char_width
    text_height = font_size * 1.2  # Adjust height to account for line height
    return text_width, text_height


def detect_bounding_box(
    pdf_path: str, text: str, location: str = "top", page_num: int = 0,
    threshold: int = 200, font_size: Optional[float] = None
) -> Optional[Tuple[float, float, float, float]]:
    """
    Detect a bounding box based on whitespace and size it according to the text.

    Args:
        pdf_path (str): Path to the PDF file.
        text (str): Text to consider for bounding box size.
        location (str): Location to detect bounding box ('top' or 'bottom'). Default is 'top'.
        page_num (int): Page number to process (0-based index). Default is 0 (first page).
        threshold (int): Pixel value threshold for considering a pixel as white. Default is 200.
        font_size (Optional[float]): Font size for the text. If None, extract from the document.

    Returns:
        Optional[Tuple[float, float, float, float]]: Bounding box coordinates (x0, y0, x1, y1) or None if not found.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num]
        if font_size is None:
            font_size = get_average_font_size(pdf_path, page_num)

        # Detect existing text blocks
        text_blocks = page.get_text("blocks")
        if not text_blocks:
            raise ValueError("No text blocks found on the page.")

        # Find the lowest y1 coordinate of existing text
        lowest_y1 = 0
        for block in text_blocks:
            x0, y0, x1, y1, text2, block_no, block_type = block  # Adjust tuple unpacking
            print(text2)
            if text2.strip() and text != "":
                if y1 > lowest_y1:
                    lowest_y1 = y1
        print(lowest_y1)

        # Define bounding box margins
        margin = 10
        box_width = page.rect.width - 2 * margin

        # Wrap the text to fit inside the bounding box
        words = text.split()
        lines = []
        current_line = ""
        max_line_width = 0
        for word in words:
            test_line = f"{current_line} {word}".strip()
            test_width, _ = calculate_text_dimensions(test_line, font_size)
            if test_width <= box_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
            max_line_width = max(max_line_width, test_width)

        # Add the last line
        if current_line:
            lines.append(current_line)

        # Calculate the total height for the bounding box based on the number of lines
        box_height = font_size * 1.5 * len(lines)

        # Determine the bounding box position
        if location == "top":
            x0, y0 = margin, margin
        elif location == "bottom":
            x0, y0 = margin, lowest_y1+margin  # Start from whitespace below the last text
        else:
            raise ValueError("Location must be either 'top' or 'bottom'.")

        x1 = min(x0 + max_line_width, page.rect.width - margin)  # No right padding
        y1 = y0 + box_height

        # Draw the bounding box
        rect = fitz.Rect(x0, y0, x1, y1)
        page.draw_rect(rect, color=(0, 0, 1), width=2)  # Blue border

        # Draw the text line by line
        current_y = y0 + font_size * 1.2  # First line position
        for line in lines:
            if current_y > y1:  # Stop if text exceeds bounding box
                break
            page.insert_text(
                fitz.Point(x0, current_y),
                line,
                fontsize=font_size,
                color=(1, 0, 0)  # Red text
            )
            current_y += font_size * 1.5  # Move to the next line

        # Save the updated PDF
        doc.save("modified_document.pdf")
        return (x0, y0, x1, y1)

    finally:
        doc.close()


if __name__ == "__main__":
    # Example usage
    pdf_path = "PDF/test.pdf"
    # top_box = detect_bounding_box(pdf_path, location="top", add_text=True)
    # if top_box:
    #     print(f"Detected top bounding box: {top_box}")
    paragraph = lorem.paragraph() + "\n" + lorem.paragraph() + "\n" + lorem.paragraph()
    # text = "This is dynamically placed text"
    # Detect bottom bounding box
    location = "top"
    font_size = get_average_font_size(pdf_path)

    bottom_box = detect_bounding_box(pdf_path, paragraph, location)
    if bottom_box:
        print(f"Detected bottom bounding box: {bottom_box}")