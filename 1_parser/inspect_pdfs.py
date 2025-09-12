import pdfplumber
from config import local_params as lp
from pathlib import Path
import csv

output_csv = lp.output_directory / "pdf_summary.csv"
output_csv.parent.mkdir(parents=True, exist_ok=True)

rows = []

for pdf_file in lp.pdf_directory.glob("*.pdf"):
    all_text = []
    all_objects = set()
    total_images = 0
    total_lines = 0
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Get metadata (concatenate all page texts)
            text = page.extract_text() or ""
            all_text.append(text)
            # Get objects (collect all unique object types)
            all_objects.update(page.objects.keys())
            # Get images (count total images)
            total_images += len(page.images)
            # Get lines (count total lines)
            total_lines += len(page.lines)
    row = {
        "filename": pdf_file.name,
        "metadata": " ".join(all_text).replace("\n", " ").strip(),
        "objects": ", ".join(sorted(all_objects)),
        "images": total_images,
        "lines": total_lines,
    }
    rows.append(row)

with open(output_csv, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "metadata", "objects", "images", "lines"])
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Wrote summary to {output_csv}")