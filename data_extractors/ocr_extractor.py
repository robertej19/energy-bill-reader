"""
OCR-based text extraction from PDFs.

This module provides functionality to extract text from PDFs using OCR,
handling layout reconstruction and text post-processing.
"""

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract

from config import local_params as lp


def setup_tesseract_path():
    """Configure Tesseract path from environment variable if set."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd


def rasterize_page(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Convert a PDF page to PIL Image for OCR processing."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def ocr_tokens_df(pil_img: Image.Image, lang: str = "eng", psm: int = 3) -> pd.DataFrame:
    """Get OCR tokens as a DataFrame with position information."""
    df = pytesseract.image_to_data(
        pil_img,
        lang=lang,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DATAFRAME
    )
    
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "level","page_num","block_num","par_num","line_num","word_num",
            "left","top","width","height","conf","text"
        ])
    
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"].str.strip() != "") & (df["conf"] > 0)]
    return df


def reconstruct_lines(df: pd.DataFrame, max_spaces: int = 12) -> List[Dict[str, Any]]:
    """Reconstruct text lines from OCR tokens, preserving layout."""
    if df.empty:
        return []
    
    need = {"block_num","par_num","line_num","left","top","width","height","text"}
    for c in need:
        if c not in df.columns:
            df[c] = 0
    
    df = df.sort_values(["block_num","par_num","line_num","left"]).copy()

    lines: List[Dict[str, Any]] = []
    for (blk, par, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
        g = g.sort_values("left")
        
        def charw(row) -> float:
            t = row["text"]
            n = max(1, len(t.strip()))
            return float(row["width"]) / n
        
        cw_med = float(g.apply(charw, axis=1).median()) if len(g) else 8.0
        cw_med = max(cw_med, 2.0)

        parts: List[str] = []
        prev_right = None
        
        for _, r in g.iterrows():
            word = str(r["text"])
            left = float(r["left"])
            width = float(r["width"])
            
            if prev_right is None:
                parts.append(word)
            else:
                gap_px = max(0.0, left - prev_right)
                spaces = int(round(gap_px / cw_med))
                spaces = 1 if spaces <= 0 else min(spaces, max_spaces)
                parts.append(" " * spaces + word)
            
            prev_right = left + width

        y0 = float(g["top"].min())
        y1 = float((g["top"] + g["height"]).max())
        text = "".join(parts)
        lines.append({"y0": y0, "y1": y1, "text": text, "block": int(blk), "par": int(par), "line": int(ln)})

    lines.sort(key=lambda L: (L["block"], L["par"], L["y0"], L["line"]))
    return lines


def merge_hyphenation(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge lines that are hyphenated at line breaks."""
    if not lines:
        return lines
    
    out: List[Dict[str, Any]] = []
    i = 0
    
    while i < len(lines):
        cur = lines[i]["text"]
        if i+1 < len(lines):
            nxt = lines[i+1]["text"]
            if re.search(r"[A-Za-z0-9]-\s*$", cur) and re.match(r"^[a-zA-Z]", nxt):
                merged = lines[i].copy()
                merged["text"] = re.sub(r"-\s*$", "", cur) + nxt.lstrip()
                merged["y1"] = max(lines[i]["y1"], lines[i+1]["y1"])
                out.append(merged)
                i += 2
                continue
        out.append(lines[i])
        i += 1
    
    return out


def _strip_soft_hyphen(s: str) -> str:
    """Remove soft hyphens from text."""
    return s.replace("\u00ad", "")


def group_paragraphs(lines: List[Dict[str, Any]]) -> List[str]:
    """Group lines into paragraphs based on spacing and structure."""
    if not lines: 
        return []
    
    heights = [L["y1"] - L["y0"] for L in lines]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h

    paras: List[List[str]] = []
    cur: List[str] = [ _strip_soft_hyphen(lines[0]["text"]) ]

    for prev, cur_line in zip(lines, lines[1:]):
        new_para = False
        if (cur_line["block"] != prev["block"]) or (cur_line["par"] != prev["par"]):
            new_para = True
        else:
            gap = cur_line["y0"] - prev["y1"]
            if gap > gap_thr:
                new_para = True
        
        if new_para:
            paras.append(cur)
            cur = [ _strip_soft_hyphen(cur_line["text"]) ]
        else:
            cur.append(_strip_soft_hyphen(cur_line["text"]))
    
    if cur: 
        paras.append(cur)

    return [ "\n".join(seg) for seg in paras if "".join(seg).strip() ]


def extract_text_with_ocr(pdf_path: Union[str, Path], 
                         dpi: int = 350, 
                         lang: str = "eng", 
                         psm: int = 3,
                         max_spaces: int = 12, 
                         preserve_page_breaks: bool = True) -> str:
    """
    Extract text from PDF using OCR.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rasterization
        lang: Tesseract language code
        psm: Tesseract page segmentation mode
        max_spaces: Maximum spaces between words
        preserve_page_breaks: Whether to insert page markers
        
    Returns:
        Extracted text as string
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    all_parts: List[str] = []

    for i, page in enumerate(doc):
        pil = rasterize_page(page, dpi=dpi)
        df = ocr_tokens_df(pil, lang=lang, psm=psm)

        if df.empty:
            page_text = pytesseract.image_to_string(
                pil, lang=lang, config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
            ).strip()
        else:
            lines = reconstruct_lines(df, max_spaces=max_spaces)
            lines = merge_hyphenation(lines)
            paragraphs = group_paragraphs(lines)
            page_text = "\n\n".join(paragraphs).strip()

        if page_text:
            if preserve_page_breaks and all_parts:
                all_parts.append(f"\n\n--- Page {i+1} ---\n\n")
            all_parts.append(page_text)

    doc.close()

    final_text = ("\n\n".join(all_parts)).rstrip() + ("\n" if all_parts else "")
    return final_text


def process_pdf_to_file(pdf_path: Path, 
                       out_dir: Path, 
                       dpi: int = 350,
                       lang: str = "eng", 
                       psm: int = 3,
                       max_spaces: int = 12, 
                       preserve_page_breaks: bool = True) -> Path:
    """Process PDF and save to text file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"{pdf_path.stem}.txt"
    
    text = extract_text_with_ocr(
        pdf_path, dpi=dpi, lang=lang, psm=psm,
        max_spaces=max_spaces, preserve_page_breaks=preserve_page_breaks
    )
    
    out_txt.write_text(text, encoding="utf-8")
    return out_txt


def get_output_path(input_path: Path) -> Path:
    """Convert input_data path to corresponding output_data path."""
    input_data_path = Path("input_data")
    output_data_path = Path("output_data")

    # Convert to relative path from input_data
    try:
        relative_path = input_path.relative_to(input_data_path)
        # Create output path by replacing input_data with output_data
        output_path = output_data_path / relative_path
        # Change extension from .pdf to .txt
        output_path = output_path.with_suffix('.txt')
        return output_path
    except ValueError:
        # If path is not under input_data, use output_data directly
        output_path = output_data_path / input_path.name
        return output_path.with_suffix('.txt')


if __name__ == "__main__":
    setup_tesseract_path()

    parser = argparse.ArgumentParser(description="Extract text from PDFs using OCR")
    parser.add_argument("inputs", nargs="*", help="Input PDF files or directories containing PDFs (default: gold input data location)")
    args = parser.parse_args()

    # If no inputs provided, use gold input data location
    if not args.inputs:
        args.inputs = [str(lp.gold_input_data_location)]

    # Collect all PDF files from inputs
    pdf_files = []
    for input_path in args.inputs:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.pdf':
            pdf_files.append(path)
        elif path.is_dir():
            pdf_files.extend(path.glob("**/*.pdf"))
        else:
            print(f"Warning: {input_path} is not a valid file or directory")

    if not pdf_files:
        print("No PDF files found to process")
        exit(1)

    print(f"Processing {len(pdf_files)} PDF files...")

    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        try:
            # Get output path automatically
            output_file = get_output_path(pdf_path)

            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Extract text and save
            text = extract_text_with_ocr(pdf_path)
            output_file.write_text(text, encoding="utf-8")

            print(f"  Saved to {output_file}")
        except Exception as e:
            print(f"  Error processing {pdf_path}: {e}")

    print("OCR extraction complete!")
