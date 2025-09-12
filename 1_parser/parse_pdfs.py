# parse_pdfs_page_as_image.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Any
import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import pytesseract

def parse_args():
    p = argparse.ArgumentParser(description="OCR each PDF page as a single image (no region slicing).")
    p.add_argument(
        "--input_path",
        type=str,
        default="pdfs/example_2.pdf",
        help="PDF file or directory containing PDFs (default: pdfs/example_2.pdf)",
    )
    p.add_argument("--out", type=str, default="out_ocr", help="Output dir")
    p.add_argument("--dpi", type=int, default=350, help="Rasterization DPI")
    p.add_argument("--lang", type=str, default="eng", help="Tesseract languages, e.g. 'eng' or 'eng+spa'")
    p.add_argument("--psm", type=int, default=3, help="Tesseract PSM (3=auto page; 4=single column; 6=single block)")
    p.add_argument(
        "--page-as-image",
        action="store_true",
        default=True,
        help="OCR the entire page bitmap (recommended)."
    )
    p.add_argument(
        "--skip-images",
        action="store_true",
        default=True,
        help="Skip extracting embedded images."
    )
    p.add_argument(
        "--save-page-text",
        action="store_true",
        help="Also save page.get_text('text') for reference."
    )
    return p.parse_args()


def rasterize_page(page: fitz.Page, dpi: int = 350) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # full page
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_page_to_df(pil_img: Image.Image, lang: str, psm: int) -> pd.DataFrame:
    df = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DATAFRAME
    )
    if df.empty:
        return pd.DataFrame(columns=["text","conf","left","top","width","height"])
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"] != "") & (df["conf"] > 0)]
    keep = ["text","conf","left","top","width","height"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.Series(dtype="float64" if c=="conf" else "int64")
    return df[keep].reset_index(drop=True)

def discover_pdfs(input_path: Path) -> List[Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".pdf":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.pdf"))
    return []

def process_pdf(pdf_path: Path, out_root: Path, dpi: int, lang: str, psm: int,
                page_as_image: bool, skip_images: bool, save_page_text: bool) -> Dict[str, Any]:
    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    if save_page_text:
        (out_dir / "page_text").mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    frames = []
    meta = []
    for i, page in enumerate(doc):
        if save_page_text:
            (out_dir / "page_text" / f"page_{i:04d}.txt").write_text(page.get_text("text") or "", encoding="utf-8")

        # IMPORTANT: do not slice regions or images; OCR the full page bitmap once
        if page_as_image:
            pil = rasterize_page(page, dpi=dpi)
            df = ocr_page_to_df(pil, lang=lang, psm=psm)
        else:
            # (Not recommended here) fallback could mix embedded text + OCR
            pil = rasterize_page(page, dpi=dpi)
            df = ocr_page_to_df(pil, lang=lang, psm=psm)

        if not df.empty:
            df.insert(0, "page_index", i)
            df.insert(0, "pdf_name", pdf_path.name)
            frames.append(df)

        # We still report image count but skip extraction to avoid any per-image OCR logic
        num_imgs = len(page.get_images(full=True)) if not skip_images else 0
        meta.append({"page_index": i, "num_embedded_images": num_imgs})

    doc.close()

    ocr_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["pdf_name","page_index","text","conf","left","top","width","height"]
    )
    # Write outputs
    ocr_df.to_parquet(out_dir / f"{pdf_path.stem}_ocr_tokens.parquet", index=False)
    ocr_df.to_csv(out_dir / f"{pdf_path.stem}_ocr_tokens.csv", index=False)

    summary = {
        "pdf_path": str(pdf_path),
        "num_pages": len(meta),
        "pages": meta,
        "outputs": {
            "ocr_tokens_parquet": str(out_dir / f"{pdf_path.stem}_ocr_tokens.parquet"),
            "ocr_tokens_csv": str(out_dir / f"{pdf_path.stem}_ocr_tokens.csv"),
            "page_text_dir": str(out_dir / "page_text") if save_page_text else None,
        }
    }
    (out_dir / f"{pdf_path.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def main():
    args = parse_args()
    in_path = Path(args.input_path)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdfs(in_path)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {in_path}")

    index = []
    for pdf in pdfs:
        print(f"[+] {pdf}")
        index.append(process_pdf(
            pdf, out_root, dpi=args.dpi, lang=args.lang, psm=args.psm,
            page_as_image=args.page_as_image, skip_images=args.skip_images,
            save_page_text=args.save_page_text
        ))
    (out_root / "run_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[✓] Done -> {out_root.resolve()}")

if __name__ == "__main__":
    main()
