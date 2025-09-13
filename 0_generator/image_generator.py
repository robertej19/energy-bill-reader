from config import local_params as lp
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import os


def pdf_to_image_pdf(input_pdf_path, output_pdf_path):
    doc = fitz.open(input_pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    if images:
        images[0].save(output_pdf_path, save_all=True, append_images=images[1:], resolution=300)
    doc.close()


def main():
    input_dir = lp.latex_pdf_directory
    output_dir = lp.image_pdf_directory
    output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file in input_dir.glob("*.pdf"):
        output_pdf = output_dir / f"image_{pdf_file.name}"
        print(f"Converting {pdf_file} -> {output_pdf}")
        pdf_to_image_pdf(pdf_file, output_pdf)
    print("Done.")

if __name__ == "__main__":
    main()
