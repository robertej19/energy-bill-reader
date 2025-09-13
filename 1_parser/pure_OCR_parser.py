#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import fitz  # PyMuPDF
from PIL import Image
import pandas as pd
import numpy as np
import pytesseract
import re

# ---------------- Args ----------------

def parse_args():
    p = argparse.ArgumentParser(description="Pure OCR (per-page image) → layout-aware text.")
    p.add_argument("--input_path", type=str, required=True, help="PDF file or directory of PDFs")
    p.add_argument("--out", type=str, default="out_txt", help="Output directory for .txt files")
    p.add_argument("--dpi", type=int, default=350, help="Rasterization DPI for OCR (300–400 typical)")
    p.add_argument("--lang", type=str, default="eng", help="Tesseract languages (e.g. 'eng' or 'eng+spa')")
    p.add_argument("--psm", type=int, default=3, help="Tesseract PSM: 3=auto, 4=single column, 6=single block")
    p.add_argument("--max-spaces", type=int, default=12, help="Cap inter-word spaces when reconstructing lines")
    p.add_argument("--preserve-page-breaks", action="store_true", help="Insert '--- Page N ---' between pages")
    return p.parse_args()

# ---------------- OCR helpers ----------------

def rasterize_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def ocr_tokens_df(pil_img: Image.Image, lang: str, psm: int) -> pd.DataFrame:
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
    # Clean
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    # Keep meaningful tokens only (positive conf and non-empty text)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"].str.strip() != "") & (df["conf"] > 0)]
    return df

# ---------------- Reconstruction ----------------

def _strip_soft_hyphen(s: str) -> str:
    # Remove Unicode soft hyphen characters if any
    return s.replace("\u00ad", "")

def reconstruct_lines(df: pd.DataFrame, max_spaces: int = 12) -> List[Dict[str, Any]]:
    """
    Build line strings with reflective spacing using word gaps.
    Returns a list of dicts: {"y0": float, "y1": float, "text": str, "block": int, "par": int, "line": int}
    """
    if df.empty:
        return []

    need_cols = {"block_num","par_num","line_num","left","top","width","height","text"}
    for c in need_cols:
        if c not in df.columns:
            df[c] = 0

    # Sort stable by reading order as given by Tesseract
    df = df.sort_values(["block_num", "par_num", "line_num", "left"]).copy()

    lines: List[Dict[str, Any]] = []
    for (blk, par, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
        g = g.sort_values("left")
        # Estimate a character width for this line (median over words)
        def charw(row) -> float:
            t = row["text"]
            n = max(1, len(t.strip()))
            return float(row["width"]) / n
        cw_series = g.apply(charw, axis=1)
        char_w = float(cw_series.median()) if len(cw_series) else 8.0
        char_w = max(char_w, 2.0)

        # Build the line with spaces proportional to word gaps
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
                spaces = int(round(gap_px / char_w))
                if spaces <= 0:
                    spaces = 1  # at least a space
                spaces = min(spaces, max_spaces)
                parts.append(" " * spaces + word)
            prev_right = left + width

        y0 = float(g["top"].min())
        y1 = float((g["top"] + g["height"]).max())
        text = "".join(parts)
        lines.append({"y0": y0, "y1": y1, "text": text, "block": int(blk), "par": int(par), "line": int(ln)})

    # Sort by block/par/line as primary reading order (Tesseract already tries to split columns)
    lines.sort(key=lambda L: (L["block"], L["par"], L["y0"], L["line"]))
    return lines

def merge_hyphenation(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If a line ends with a hyphen and the next line starts with a lowercase/letter,
    join them by removing the hyphen (soft wrap).
    """
    if not lines:
        return lines
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        cur = lines[i]["text"]
        if i+1 < len(lines):
            nxt = lines[i+1]["text"]
            # hyphen at end of current (but not double-dash) and next starts with letter
            if re.search(r"[A-Za-z0-9]-\s*$", cur) and re.match(r"^[a-zA-Z]", nxt):
                cur = re.sub(r"-\s*$", "", cur) + nxt.lstrip()
                # Merge bounding box vertically
                merged = lines[i].copy()
                merged["text"] = cur
                merged["y1"] = max(lines[i]["y1"], lines[i+1]["y1"])
                out.append(merged)
                i += 2
                continue
        out.append(lines[i])
        i += 1
    return out

def group_paragraphs(lines: List[Dict[str, Any]]) -> List[str]:
    """
    Group lines into paragraphs using:
      - paragraph or block boundaries from Tesseract
      - vertical gap threshold (relative to median line height)
    Return list of paragraph strings (with their internal line breaks preserved).
    """
    if not lines:
        return []

    # Estimate a median line height to decide paragraph gaps
    heights = [L["y1"] - L["y0"] for L in lines]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h  # tweakable

    paras: List[List[str]] = []
    cur: List[str] = [ _strip_soft_hyphen(lines[0]["text"]) ]

    for prev, cur_line in zip(lines, lines[1:]):
        new_para = False

        # Paragraph/block boundary from Tesseract
        if (cur_line["block"] != prev["block"]) or (cur_line["par"] != prev["par"]):
            new_para = True
        else:
            # Vertical gap heuristic
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

    # Join lines within a paragraph with newline to preserve within-paragraph breaks
    return [ "\n".join(seg) for seg in paras if "".join(seg).strip() ]

# ---------------- Per-PDF ----------------

def ocr_pdf_to_text(pdf_path: Path, out_dir: Path, dpi: int, lang: str, psm: int,
                    max_spaces: int, preserve_page_breaks: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"{pdf_path.stem}.txt"

    doc = fitz.open(pdf_path)
    all_parts: List[str] = []

    for i, page in enumerate(doc):
        pil = rasterize_page(page, dpi=dpi)
        df = ocr_tokens_df(pil, lang=lang, psm=psm)

        if df.empty:
            # fallback: plain image_to_string (less layout-aware)
            page_text = pytesseract.image_to_string(
                pil, lang=lang, config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
            ).strip()
            if page_text:
                if preserve_page_breaks and all_parts:
                    all_parts.append(f"\n\n--- Page {i+1} ---\n\n")
                all_parts.append(page_text)
            continue

        lines = reconstruct_lines(df, max_spaces=max_spaces)
        lines = merge_hyphenation(lines)
        paragraphs = group_paragraphs(lines)

        page_text = "\n\n".join(paragraphs).strip()
        if page_text:
            if preserve_page_breaks and all_parts:
                all_parts.append(f"\n\n--- Page {i+1} ---\n\n")
            all_parts.append(page_text)

    doc.close()

    final_text = ("\n\n".join(all_parts)).rstrip() + "\n"
    out_txt.write_text(final_text, encoding="utf-8")
    return out_txt

# ---------------- Batch ----------------

def discover_pdfs(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.pdf"))
    return []

def main():
    args = parse_args()
    in_path = Path(args.input_path)
    out_root = Path(args.out)
    pdfs = discover_pdfs(in_path)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {in_path}")

    out_root.mkdir(parents=True, exist_ok=True)
    for p in pdfs:
        print(f"[+] OCR {p}")
        out_txt = ocr_pdf_to_text(
            pdf_path=p, out_dir=out_root, dpi=args.dpi, lang=args.lang, psm=args.psm,
            max_spaces=args.max_spaces, preserve_page_breaks=args.preserve_page_breaks
        )
        print(f"    -> {out_txt}")

if __name__ == "__main__":
    main()
