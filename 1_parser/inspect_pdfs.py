import pdfplumber
from config import local_params as lp
from pathlib import Path
import csv

output_csv = lp.output_directory / "pdf_summary.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

rows = []

def process_pdfs(pdf_dir, source_label):
    for pdf_file in pdf_dir.glob("*.pdf"):
        all_text = []
        all_objects = set()
        total_images = 0
        total_lines = 0
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                all_text.append(text)
                all_objects.update(page.objects.keys())
                total_images += len(page.images)
                total_lines += len(page.lines)
        row = {
            "filename": pdf_file.name,
            "source": source_label,
            "metadata": " ".join(all_text).replace("\n", " ").strip(),
            "objects": ", ".join(sorted(all_objects)),
            "images": total_images,
            "lines": total_lines,
        }
        rows.append(row)

process_pdfs(lp.latex_pdf_directory, "latex")
process_pdfs(lp.image_pdf_directory, "image")

with open(output_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "source", "metadata", "objects", "images", "lines"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote summary to {output_csv}")