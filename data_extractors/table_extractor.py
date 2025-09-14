#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import fitz  # PyMuPDF

# Import config for default input location
from config import local_params as lp
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract

# ---------- OCR helpers (reuse your patterns) ----------

def setup_tesseract_path():
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

def rasterize_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_tokens_df(pil_img: Image.Image, lang: str, psm: int) -> pd.DataFrame:
    df = pytesseract.image_to_data(
        pil_img, lang=lang,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
        output_type=pytesseract.Output.DATAFRAME
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["left","top","width","height","conf","text","line_num","block_num","par_num"])
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"].str.strip() != "") & (df["conf"] > 0)]
    return df

# ---------- Table detection (line morphology) ----------

def _binarize(gray: np.ndarray) -> np.ndarray:
    # Contrast-normalized Otsu; invert so lines are white
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    return th_inv

def detect_table_boxes(img_rgb: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Return list of table bounding boxes in pixel coords: (x0,y0,x1,y1).
    Uses horizontal + vertical morphology to find ruled tables.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    binv = _binarize(gray)

    # Kernels proportional to image dims
    h_scale = max(10, w // 120)     # horizontal line length
    v_scale = max(10, h // 120)     # vertical line length
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_scale, 1))
    vert_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_scale))

    # Extract lines
    horiz = cv2.erode(binv, horiz_kernel, iterations=1)
    horiz = cv2.dilate(horiz, horiz_kernel, iterations=1)
    vert  = cv2.erode(binv, vert_kernel, iterations=1)
    vert  = cv2.dilate(vert, vert_kernel, iterations=1)

    lines = cv2.bitwise_or(horiz, vert)

    # Find connected components as table candidates
    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int,int,int,int]] = []
    min_area = (w*h) * 0.002  # ignore tiny
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww*hh < min_area: 
            continue
        # Require some line density inside
        roi = lines[y:y+hh, x:x+ww]
        density = (roi > 0).mean()
        if density < 0.02: 
            continue
        boxes.append((x,y,x+ww,y+hh))

    # sort by top-left
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def extract_grid_lines(img_rgb: np.ndarray, box: Tuple[int,int,int,int]) -> Tuple[List[int], List[int]]:
    """
    Inside a detected table box, return sorted vertical and horizontal line positions (pixel x/y).
    """
    x0,y0,x1,y1 = box
    crop = img_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    binv = _binarize(gray)

    h, w = binv.shape
    # Stronger kernels inside the crop
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//20)))

    horiz = cv2.erode(binv, hk, iterations=1)
    horiz = cv2.dilate(horiz, hk, iterations=1)
    vert  = cv2.erode(binv, vk, iterations=1)
    vert  = cv2.dilate(vert, vk, iterations=1)

    # Sum projections to find line positions
    ys = (horiz > 0).sum(axis=1)   # for each y, count white pixels
    xs = (vert  > 0).sum(axis=0)

    def _peaks(arr, thr_ratio=0.4, min_gap=5):
        thr = arr.max() * thr_ratio
        idx = np.where(arr >= thr)[0]
        if idx.size == 0:
            return []
        # Merge runs into centers
        groups = []
        start = idx[0]; prev = idx[0]
        for i in idx[1:]:
            if i - prev > min_gap:
                groups.append((start, prev))
                start = i
            prev = i
        groups.append((start, prev))
        return [int((a+b)//2) for a,b in groups]

    h_lines = _peaks(ys, thr_ratio=0.35, min_gap=max(3, h//200))
    v_lines = _peaks(xs, thr_ratio=0.35, min_gap=max(3, w//200))

    # Require at least 2 lines per axis to form a grid
    return v_lines, h_lines

# ---------- Cell OCR given grid ----------

def ocr_table_cells(pil_page: Image.Image,
                    table_box: Tuple[int,int,int,int],
                    v_lines: List[int], h_lines: List[int],
                    lang: str, psm_cell: int = 6) -> Tuple[List[List[str]], List[Tuple[int,int,int,int]]]:
    """
    OCR each cell by cropping the cell rectangle. Returns (grid_text, cell_boxes_px).
    cell_boxes_px are absolute page-pixel coords aligned to page raster.
    """
    px0, py0, px1, py1 = table_box
    grid_text: List[List[str]] = []
    cell_boxes: List[Tuple[int,int,int,int]] = []

    # Build cell rectangles from consecutive line pairs
    if len(v_lines) < 2 or len(h_lines) < 2:
        return [], []

    # ensure sorted unique
    vx = sorted(set(v_lines)); hy = sorted(set(h_lines))

    for r in range(len(hy)-1):
        row_text = []
        y_a = py0 + hy[r]
        y_b = py0 + hy[r+1]
        for c in range(len(vx)-1):
            x_a = px0 + vx[c]
            x_b = px0 + vx[c+1]
            # pad slightly inside to avoid lines
            pad = 2
            xa, ya = max(px0, x_a+pad), max(py0, y_a+pad)
            xb, yb = min(px1, x_b-pad), min(py1, y_b-pad)
            if xa >= xb or ya >= yb:
                row_text.append("")
                cell_boxes.append((xa, ya, xb, yb))
                continue
            crop = pil_page.crop((xa, ya, xb, yb))
            txt = pytesseract.image_to_string(
                crop, lang=lang, config=f"--oem 3 --psm {psm_cell} -c preserve_interword_spaces=1"
            )
            row_text.append(txt.strip())
            cell_boxes.append((xa, ya, xb, yb))
        grid_text.append(row_text)
    return grid_text, cell_boxes

# ---------- Fallback when grid lines are missing ----------

def fallback_table_from_tokens(pil_page: Image.Image,
                               table_box: Tuple[int,int,int,int],
                               lang: str) -> List[List[str]]:
    """
    Token-based fallback: group by line (y) into rows; within each row, sort by x.
    Produces a ragged TSV-like grid (column counts can vary).
    """
    x0,y0,x1,y1 = table_box
    crop = pil_page.crop((x0,y0,x1,y1))
    df = ocr_tokens_df(crop, lang=lang, psm=6)
    if df.empty: 
        return []

    # row grouping by y center gaps
    df["y_center"] = df["top"] + df["height"]/2
    df = df.sort_values(["y_center","left"])

    rows: List[List[str]] = []
    current: List[Tuple[float,str]] = []
    # dynamic threshold based on median height
    med_h = float(np.median(df["height"])) if len(df) else 12.0
    thr = 0.6 * med_h
    last_y = None

    for _, r in df.iterrows():
        yc = float(r["y_center"])
        if last_y is None or abs(yc - last_y) <= thr:
            current.append((float(r["left"]), str(r["text"])))
        else:
            # flush previous row
            current.sort(key=lambda t: t[0])
            rows.append([" ".join([t[1] for t in current]).strip()])
            current = [(float(r["left"]), str(r["text"]))]
        last_y = yc

    if current:
        current.sort(key=lambda t:t[0])
        rows.append([" ".join([t[1] for t in current]).strip()])

    # Split row strings into columns by multiple spaces or tabs (heuristic)
    grid: List[List[str]] = []
    for row in rows:
        s = row[0]
        cols = [c.strip() for c in re.split(r"\s{2,}|\t+", s) if c.strip()]
        grid.append(cols if cols else [s])
    return grid

# ---------- Page-level pipeline ----------

def extract_tables_from_page(page: fitz.Page, dpi: int, lang: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts per table:
      {
        "bbox_px": (x0,y0,x1,y1),
        "rows": R, "cols": C,
        "cells": [[...], ...]     # strings, may be ragged if fallback
      }
    """
    pil = rasterize_page(page, dpi=dpi)
    img = np.array(pil)
    boxes = detect_table_boxes(img)

    results = []
    for b in boxes:
        vx, hy = extract_grid_lines(img, b)
        if len(vx) >= 2 and len(hy) >= 2:
            grid, _cells = ocr_table_cells(pil, b, vx, hy, lang=lang, psm_cell=6)
            if grid and all(isinstance(row, list) for row in grid):
                results.append({
                    "bbox_px": b, "rows": len(grid), "cols": len(grid[0]) if grid else 0,
                    "cells": grid, "source": "grid_lines"
                })
                continue
        # fallback
        grid_fb = fallback_table_from_tokens(pil, b, lang=lang)
        if grid_fb:
            cols = max(len(r) for r in grid_fb)
            results.append({
                "bbox_px": b, "rows": len(grid_fb), "cols": cols,
                "cells": grid_fb, "source": "token_fallback"
            })
    return results

# ---------- Saving ----------

def save_tables(doc_stem: str, page_idx: int, tables: List[Dict[str,Any]], out_root: Path):
    page_dir = out_root / f"{doc_stem}_page_{page_idx:03d}"
    page_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for t_idx, t in enumerate(tables):
        cells = t["cells"]
        csv_path = page_dir / f"table_{t_idx:02d}.csv"
        # Ragged rows → pad
        max_c = max(len(r) for r in cells) if cells else 0
        norm = [r + [""]*(max_c-len(r)) for r in cells]
        pd.DataFrame(norm).to_csv(csv_path, index=False, header=False)

        manifest.append({
            "table_index": t_idx,
            "bbox_px": t["bbox_px"],
            "rows": t["rows"], "cols": t["cols"],
            "csv": str(csv_path),
            "source": t.get("source","")
        })

    (page_dir / "tables_manifest.json").write_text(
        json.dumps({"page_index": page_idx, "tables": manifest}, indent=2),
        encoding="utf-8"
    )

# ---------- CLI ----------

def process_pdf(pdf_path: Path, out_root: Path, dpi: int, lang: str):
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        tables = extract_tables_from_page(page, dpi=dpi, lang=lang)
        save_tables(pdf_path.stem, i, tables, out_root)
    doc.close()

def main():
    ap = argparse.ArgumentParser(description="Extract tables from PDFs (image-based detection + OCR).")
    ap.add_argument("--input_path", type=str, default=str(lp.gold_input_data_location),
                    help="PDF file or directory (default: gold_input_data_location)")
    ap.add_argument("--out", type=str, default="out_tables", help="Output directory")
    ap.add_argument("--dpi", type=int, default=350, help="Rasterization DPI")
    ap.add_argument("--lang", type=str, default="eng", help="OCR language, e.g., 'eng' or 'eng+spa'")
    args = ap.parse_args()

    setup_tesseract_path()
    in_path = Path(args.input_path)
    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)

    pdfs: List[Path] = []
    if in_path.is_file() and in_path.suffix.lower()==".pdf":
        pdfs = [in_path]
    elif in_path.is_dir():
        pdfs = sorted(in_path.rglob("*.pdf"))
    else:
        raise SystemExit(f"Invalid input_path: {in_path}")

    for p in pdfs:
        print(f"[+] {p}")
        process_pdf(p, out_root, dpi=args.dpi, lang=args.lang)
    print(f"[✓] Done -> {out_root.resolve()}")

if __name__ == "__main__":
    main()
