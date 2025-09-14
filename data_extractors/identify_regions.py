import argparse
from pathlib import Path
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image

# --- Region Identification Logic ---

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

def draw_regions(page: fitz.Page, scale: float, tables: list, figures: list, text_blocks: list):
    """Draw colored rectangles for each region type on the PDF page."""
    # Red for tables
    for box_px in tables:
        rect = fitz.Rect(box_px) / scale
        page.draw_rect(rect, color=(1, 0, 0), fill=(1, 0, 0), fill_opacity=0.5, overlay=True)

    # Blue for figures
    for box in figures:
        page.draw_rect(box, color=(0, 0, 1), fill=(0, 0, 1), fill_opacity=0.5, overlay=True)

    # Green for text blocks
    for box in text_blocks:
        page.draw_rect(box, color=(0, 1, 0), fill=(0, 1, 0), fill_opacity=0.5, overlay=True)

def process_pdf(pdf_path: Path, output_path: Path):
    """Identify and draw regions on a new PDF."""
    doc = fitz.open(pdf_path)
    
    for page in doc:
        # 1. Rasterize for image-based detection of tables
        pil_img, scale = rasterize_page(page)
        cv_img = np.array(pil_img)
        table_boxes_px = detect_table_boxes_on_image(cv_img)
        
        # 2. Use PyMuPDF's native functions for figures and text (more direct)
        figure_boxes = [page.get_image_bbox(img) for img in page.get_images(full=True)]
        # Filter for text blocks (type=0)
        text_boxes = [fitz.Rect(b[:4]) for b in page.get_text("blocks") if b[6] == 0] 

        # 3. Draw all identified regions on the page
        draw_regions(page, scale, table_boxes_px, figure_boxes, text_boxes)

    # 4. Save the annotated PDF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path), garbage=4, deflate=True, clean=True)
    print(f"Saved annotated PDF to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Identify and visualize regions in a PDF.")
    parser.add_argument(
        "input_pdf",
        type=Path,
        nargs="?",
        default=Path("input_data/synthetic_data/gold_pdfs/doc_1.pdf"),
        help="Path to the input PDF file. Defaults to doc_1.pdf in the gold data folder."
    )
    args = parser.parse_args()

    output_dir = Path("output_visual")
    output_pdf_path = output_dir / f"{args.input_pdf.stem}_regions.pdf"

    if not args.input_pdf.exists():
        print(f"Error: Input file not found at {args.input_pdf}")
        return

    process_pdf(args.input_pdf, output_pdf_path)

if __name__ == "__main__":
    main()



