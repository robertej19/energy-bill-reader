
import fitz # PyMuPDF
import pytesseract
from PIL import Image
import os
import argparse
import cv2
import numpy as np
from pytesseract import Output # Import Output for image_to_data

# --- Helper functions for table detection (copied from identify_regions.py) ---

def rasterize_page(page: fitz.Page, dpi: int = 150) -> tuple[Image.Image, float]:
    """Rasterize a PDF page to a PIL Image."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom

def _binarize(gray: np.ndarray) -> np.ndarray:
    """Binarize a grayscale image for line detection."""
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - th

def detect_table_boxes_on_image(img_rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect table bounding boxes in pixel coordinates from an image."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bin_inv = _binarize(gray)

    # Use morphology to find horizontal and vertical lines
    h_scale = max(10, w // 100)
    v_scale = max(10, h // 100)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_scale, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_scale))

    horiz = cv2.erode(bin_inv, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)
    vert = cv2.erode(bin_inv, vert_kernel, iterations=1)
    vert = cv2.dilate(vert, vert_kernel, iterations=1)

    # Combine lines and find contours
    lines = cv2.bitwise_or(horiz, vert)
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = (w * h) * 0.01  # Filter out very small areas
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh > min_area:
            roi = lines[y:y+hh, x:x+ww]
            # Check for a minimum density of lines within the contour
            if (roi > 0).mean() > 0.03:
                boxes.append((x, y, x + ww, y + hh))
    
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def draw_regions(page: fitz.Page, scale: float, tables: list):
    """Draw colored rectangles for each region type on the PDF page."""
    # Red for tables
    for box_px in tables:
        rect = fitz.Rect(box_px) / scale
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0), fill_opacity=0.5, overlay=True)

def extract_text_from_pdf(pdf_path, output_dir):
    doc = fitz.open(pdf_path)
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pdf_filename = os.path.basename(pdf_path)
    base_filename = os.path.splitext(pdf_filename)[0]
    output_text_path = os.path.join(output_dir, base_filename + "_extracted_text.txt")
    output_table_text_path = os.path.join(output_dir, base_filename + "_extracted_tables.txt")
    output_visual_pdf_path = os.path.join(output_dir, base_filename + "_regions.pdf")

    # Create a new PDF for visual output
    output_visual_doc = fitz.open()

    with open(output_text_path, "w") as out_file, open(output_table_text_path, "w") as table_out_file:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            
            # Make a copy of the page for visual annotations
            page_for_visual = output_visual_doc.new_page(width=page.rect.width, height=page.rect.height)
            page_for_visual.show_pdf_page(page_for_visual.rect, doc, page_num)

            # Rasterize for image-based detection and OCR
            pil_img, scale = rasterize_page(page)
            cv_img = np.array(pil_img)

            # Detect tables
            table_boxes_px = detect_table_boxes_on_image(cv_img)

            # Convert table pixel coordinates to PDF coordinates
            table_rects_pdf = []
            for box_px in table_boxes_px:
                x0, y0, x1, y1 = [coord / scale for coord in box_px]
                table_rects_pdf.append(fitz.Rect(x0, y0, x1, y1))
            
            # Draw regions on the visual PDF page
            draw_regions(page_for_visual, scale, table_boxes_px)

            # --- Perform full-page OCR and filter words ---
            ocr_data = pytesseract.image_to_data(pil_img, output_type=Output.DICT)
            
            non_table_words = []
            # Initialize a list of lists, one for each detected table
            per_table_words = [[] for _ in table_rects_pdf]

            # Iterate through all detected words from full-page OCR
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                if int(ocr_data['conf'][i]) > 50: # Filter out low confidence words
                    x, y, w, h = ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]
                    word_rect_px = fitz.Rect(x, y, x + w, y + h)
                    # Convert word's pixel bounding box to PDF coordinates
                    word_rect_pdf = word_rect_px / scale
                    word_text = ocr_data['text'][i]

                    is_in_table = False
                    # Expand table rect slightly for more robust intersection check
                    for table_idx, table_rect in enumerate(table_rects_pdf):
                        # Create a padded rect directly from coordinates
                        padded_table_rect = fitz.Rect(table_rect.x0 - 10, table_rect.y0 - 10, table_rect.x1 + 10, table_rect.y1 + 10) # Increased padding
                        if word_rect_pdf.intersects(padded_table_rect) and word_text.strip(): # Check intersection and if word is not empty
                            per_table_words[table_idx].append((word_rect_pdf.x0, word_rect_pdf.y0, word_text)) # Store x,y-coords
                            is_in_table = True
                            break
                    
                    if not is_in_table and word_text.strip(): # Add to non-table if not in table and not empty
                        non_table_words.append((word_rect_pdf.x0, word_rect_pdf.y0, word_text)) # Store x,y-coords
            
            # Reconstruct text with line and space handling
            def reconstruct_text_from_words(words):
                if not words:
                    return ""
                
                # Sort words primarily by y-coordinate, then by x-coordinate
                words.sort(key=lambda x: (x[1], x[0]))
                
                reconstructed_lines = []
                if words:
                    current_line_words = []
                    current_line_y = words[0][1]

                    for i, (x_coord, y_coord, word) in enumerate(words):
                        # Check for a new line (significant jump in y)
                        if abs(y_coord - current_line_y) > 10: # y-threshold for new line
                            reconstructed_lines.append(" ".join(current_line_words))
                            current_line_words = [word]
                            current_line_y = y_coord
                        else:
                            current_line_words.append(word)
                    reconstructed_lines.append(" ".join(current_line_words)) # Add the last line
                
                return "\n".join(reconstructed_lines)

            out_file.write(f"--- Page {page_num + 1} ---\n")
            out_file.write(reconstruct_text_from_words(non_table_words))
            out_file.write("\n\n")

            # Reconstruct and write table text
            if table_boxes_px:
                table_out_file.write(f"--- Tables from Page {page_num + 1} ---\n")
                for i, table_words_list in enumerate(per_table_words):
                    table_out_file.write(f"Table {i + 1} (Region: {table_rects_pdf[i].x0:.0f},{table_rects_pdf[i].y0:.0f},{table_rects_pdf[i].x1:.0f},{table_rects_pdf[i].y1:.0f}):\n")
                    table_out_file.write(reconstruct_text_from_words(table_words_list))
                    table_out_file.write("\n---\n")
                table_out_file.write("\n\n")

    doc.close()
    output_visual_doc.save(output_visual_pdf_path, garbage=4, deflate=True, clean=True)
    output_visual_doc.close()
    print(f"Saved annotated PDF to: {output_visual_pdf_path}")
    return output_text_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from a PDF using OCR.")
    parser.add_argument("pdf_path", type=str, nargs="?", default="/home/rober/synth-reader/input_data/synthetic_data/gold_pdfs/doc_1.pdf",
                        help="Path to the input PDF file. Defaults to input_data/synthetic_data/gold_pdfs/doc_1.pdf")
    parser.add_argument("--output_dir", type=str, default="output/",
                        help="Directory to save the extracted text file. Defaults to 'output/'.")
    args = parser.parse_args()

    extracted_file = extract_text_from_pdf(args.pdf_path, args.output_dir)
    print(f"Text extracted from {args.pdf_path} and saved to {extracted_file}")
