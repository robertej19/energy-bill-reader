#!/usr/bin/env python3
"""
Unified PDF extractor: detect regions (text / tables / figures), extract text & table data.

Improvements vs previous:
- Figures: use page.get_image_rects(xref) and de-duplicate by IoU
- Text OCR: run on original page image; exclude tokens inside non-text boxes (no masking artifacts)
- Tesseract conf filter relaxed (>= 0)
- Stable PSM defaults: text psm=3, table cells psm=6
- Table acceptance: require >= 2 rows and >= 2 cols
"""
from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union

import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image

# Optional: load config if available
try:
    from config import local_params as lp
except Exception:
    lp = None

# ---------------- Tesseract setup ----------------

def setup_tesseract_path():
    """Configure Tesseract path from env variable if provided."""
    cmd = os.getenv("TESSERACT_CMD")
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd

# ---------------- Utilities ----------------

def iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter + 1e-9)

def dedupe_boxes(boxes: List[Tuple[int,int,int,int]], thr: float = 0.8) -> List[Tuple[int,int,int,int]]:
    kept: List[Tuple[int,int,int,int]] = []
    for b in sorted(boxes, key=lambda x: (x[1], x[0])):  # sort top-left
        if all(iou(b, k) < thr for k in kept):
            kept.append(b)
    return kept

# ---------------- Rasterization ----------------

def rasterize_page(page: fitz.Page, dpi: int = 300) -> Tuple[Image.Image, float]:
    """
    Render PDF page to a PIL image. Also return 'scale' = pixels per PDF point (72 dpi).
    """
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom

# ---------------- OCR tokens ----------------

def ocr_tokens_df(pil_img: Image.Image, lang: str = "eng", psm: int = 3) -> pd.DataFrame:
    """
    OCR tokens with positions from Tesseract as a DataFrame.
    Keep tokens with conf >= 0 (0 can be valid).
    """
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
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str)
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"].str.strip() != "") & (df["conf"] >= 0)]
    for c in ("block_num","par_num","line_num","left","top","width","height"):
        if c not in df.columns:
            df[c] = 0
    return df

def filter_tokens_outside_boxes(df: pd.DataFrame, boxes: List[Tuple[int,int,int,int]]) -> pd.DataFrame:
    """Return subset of tokens whose centers are NOT inside any box."""
    if df.empty or not boxes:
        return df
    bx = np.array(boxes, dtype=np.float32)
    # compute centers
    cx = df["left"].values + df["width"].values / 2.0
    cy = df["top"].values  + df["height"].values / 2.0
    keep = np.ones(len(df), dtype=bool)
    for (x0, y0, x1, y1) in bx:
        inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        keep &= ~inside
    return df.loc[keep]

# ---------------- Text line & paragraph reconstruction ----------------

def _strip_soft_hyphen(s: str) -> str:
    return s.replace("\u00ad", "")

def reconstruct_lines(df: pd.DataFrame, max_spaces: int = 12) -> List[Dict[str, Any]]:
    """
    Reconstruct text lines from OCR tokens, preserve layout and return bbox per line.
    """
    if df.empty:
        return []
    df = df.sort_values(["block_num", "par_num", "line_num", "left"]).copy()

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

        x0 = float(g["left"].min())
        y0 = float(g["top"].min())
        x1 = float((g["left"] + g["width"]).max())
        y1 = float((g["top"] + g["height"]).max())

        text = "".join(parts)
        lines.append({
            "text": text,
            "bbox_px": (x0, y0, x1, y1),
            "block": int(blk), "par": int(par), "line": int(ln)
        })

    lines.sort(key=lambda L: (L["block"], L["par"], L["bbox_px"][1], L["line"]))
    return lines

def merge_hyphenation(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge hyphenated line breaks."""
    if not lines:
        return lines
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(lines):
        cur = lines[i]["text"]
        if i + 1 < len(lines):
            nxt = lines[i+1]["text"]
            if re.search(r"[A-Za-z0-9]-\s*$", cur) and re.match(r"^[A-Za-z]", nxt):
                merged = lines[i].copy()
                merged["text"] = re.sub(r"-\s*$", "", cur) + nxt.lstrip()
                # expand bbox
                x0a,y0a,x1a,y1a = lines[i]["bbox_px"]
                x0b,y0b,x1b,y1b = lines[i+1]["bbox_px"]
                merged["bbox_px"] = (min(x0a,x0b), min(y0a,y0b), max(x1a,x1b), max(y1a,y1b))
                out.append(merged)
                i += 2
                continue
        out.append(lines[i]); i += 1
    return out

def group_paragraphs_with_boxes(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group consecutive lines into paragraphs; produce bbox per paragraph."""
    if not lines:
        return []
    heights = [L["bbox_px"][3] - L["bbox_px"][1] for L in lines]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h

    paras: List[Dict[str, Any]] = []
    cur_lines: List[Dict[str, Any]] = [lines[0]]
    for prev, cur in zip(lines, lines[1:]):
        new_para = False
        if (cur["block"] != prev["block"]) or (cur["par"] != prev["par"]):
            new_para = True
        else:
            gap = cur["bbox_px"][1] - prev["bbox_px"][3]  # y0_cur - y1_prev
            if gap > gap_thr:
                new_para = True
        if new_para:
            text = "\n".join(_strip_soft_hyphen(L["text"]) for L in cur_lines)
            if text.strip():
                x0 = min(L["bbox_px"][0] for L in cur_lines)
                y0 = min(L["bbox_px"][1] for L in cur_lines)
                x1 = max(L["bbox_px"][2] for L in cur_lines)
                y1 = max(L["bbox_px"][3] for L in cur_lines)
                paras.append({"type":"paragraph","bbox_px":(x0,y0,x1,y1),"text":text})
            cur_lines = [cur]
        else:
            cur_lines.append(cur)
    # flush last
    if cur_lines:
        text = "\n".join(_strip_soft_hyphen(L["text"]) for L in cur_lines)
        if text.strip():
            x0 = min(L["bbox_px"][0] for L in cur_lines)
            y0 = min(L["bbox_px"][1] for L in cur_lines)
            x1 = max(L["bbox_px"][2] for L in cur_lines)
            y1 = max(L["bbox_px"][3] for L in cur_lines)
            paras.append({"type":"paragraph","bbox_px":(x0,y0,x1,y1),"text":text})
    return paras

def extract_text_from_image(pil_img: Image.Image, lang: str, max_spaces: int = 12,
                            exclude_boxes_px: Optional[List[Tuple[int,int,int,int]]] = None,
                            psm_text: int = 3) -> List[Dict[str, Any]]:
    """
    OCR page image; filter tokens inside exclude boxes; build paragraphs with bboxes.
    """
    df = ocr_tokens_df(pil_img, lang=lang, psm=psm_text)
    if exclude_boxes_px:
        df = filter_tokens_outside_boxes(df, exclude_boxes_px)
    if df.empty:
        return []
    lines = reconstruct_lines(df, max_spaces=max_spaces)
    lines = merge_hyphenation(lines)
    return group_paragraphs_with_boxes(lines)

# ---------------- Table detection & extraction ----------------

def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - th  # invert so lines are white

def detect_table_boxes(img_rgb: np.ndarray) -> List[Tuple[int,int,int,int]]:
    """
    Detect ruled tables by morphological line extraction; return (x0,y0,x1,y1) pixel boxes.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    binv = _binarize(gray)

    h_scale = max(10, w // 120)
    v_scale = max(10, h // 120)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_scale, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_scale))

    horiz = cv2.dilate(cv2.erode(binv, hk, 1), hk, 1)
    vert  = cv2.dilate(cv2.erode(binv, vk, 1), vk, 1)
    lines = cv2.bitwise_or(horiz, vert)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int,int,int,int]] = []
    min_area = (w*h) * 0.002
    for c in contours:
        x,y,ww,hh = cv2.boundingRect(c)
        if ww*hh < min_area:
            continue
        roi = lines[y:y+hh, x:x+ww]
        if (roi > 0).mean() < 0.02:
            continue
        boxes.append((x,y,x+ww,y+hh))
    # Optional: NMS/merge duplicates
    return dedupe_boxes(boxes, thr=0.8)

def extract_grid_lines(img_rgb: np.ndarray, box: Tuple[int,int,int,int]) -> Tuple[List[int], List[int]]:
    x0,y0,x1,y1 = box
    crop = img_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    binv = _binarize(gray)
    h, w = binv.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//20)))

    horiz = cv2.dilate(cv2.erode(binv, hk, 1), hk, 1)
    vert  = cv2.dilate(cv2.erode(binv, vk, 1), vk, 1)

    ys = (horiz > 0).sum(axis=1)
    xs = (vert  > 0).sum(axis=0)

    def _peaks(arr, thr_ratio=0.35, min_gap=5):
        thr = arr.max() * thr_ratio
        idx = np.where(arr >= thr)[0]
        if idx.size == 0:
            return []
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
    return v_lines, h_lines

def ocr_table_cells(pil_page: Image.Image,
                    table_box: Tuple[int,int,int,int],
                    v_lines: List[int], h_lines: List[int],
                    lang: str, psm_cell: int = 6) -> Tuple[List[List[str]], List[Tuple[int,int,int,int]]]:
    px0, py0, px1, py1 = table_box
    if len(v_lines) < 2 or len(h_lines) < 2:
        return [], []
    vx = sorted(set(v_lines)); hy = sorted(set(h_lines))
    grid_text: List[List[str]] = []
    cell_boxes: List[Tuple[int,int,int,int]] = []

    for r in range(len(hy)-1):
        row_out: List[str] = []
        y_a = py0 + hy[r]; y_b = py0 + hy[r+1]
        for c in range(len(vx)-1):
            x_a = px0 + vx[c]; x_b = px0 + vx[c+1]
            pad = 2
            xa, ya = max(px0, x_a+pad), max(py0, y_a+pad)
            xb, yb = min(px1, x_b-pad), min(py1, y_b-pad)
            if xa >= xb or ya >= yb:
                row_out.append("")
                cell_boxes.append((xa,ya,xb,yb))
                continue
            crop = pil_page.crop((xa, ya, xb, yb))
            txt = pytesseract.image_to_string(
                crop, lang=lang, config=f"--oem 3 --psm {psm_cell} -c preserve_interword_spaces=1"
            )
            row_out.append(txt.strip())
            cell_boxes.append((xa,ya,xb,yb))
        grid_text.append(row_out)
    return grid_text, cell_boxes

def fallback_table_from_tokens(pil_page: Image.Image,
                               table_box: Tuple[int,int,int,int],
                               lang: str) -> List[List[str]]:
    x0,y0,x1,y1 = table_box
    crop = pil_page.crop((x0,y0,x1,y1))
    df = ocr_tokens_df(crop, lang=lang, psm=6)
    if df.empty:
        return []
    df["y_center"] = df["top"] + df["height"]/2.0
    df = df.sort_values(["y_center","left"])
    rows: List[List[str]] = []
    current: List[Tuple[float,str]] = []
    med_h = float(np.median(df["height"])) if len(df) else 12.0
    thr = 0.6 * med_h
    last_y = None
    for _, r in df.iterrows():
        yc = float(r["y_center"])
        if last_y is None or abs(yc - last_y) <= thr:
            current.append((float(r["left"]), str(r["text"])))
        else:
            current.sort(key=lambda t: t[0])
            rows.append([" ".join([t[1] for t in current]).strip()])
            current = [(float(r["left"]), str(r["text"]))]
        last_y = yc
    if current:
        current.sort(key=lambda t: t[0])
        rows.append([" ".join([t[1] for t in current]).strip()])
    grid: List[List[str]] = []
    for row in rows:
        s = row[0]
        cols = [c.strip() for c in re.split(r"\s{2,}|\t+", s) if c.strip()]
        grid.append(cols if cols else [s])
    return grid

def extract_tables_from_image(pil_img: Image.Image, lang: str) -> List[Dict[str, Any]]:
    """
    Detect and extract tables. Use line grid when possible; fallback to token grouping.
    Only accept as table if rows >= 2 and cols >= 2.
    """
    img = np.array(pil_img)
    boxes = detect_table_boxes(img)
    results: List[Dict[str, Any]] = []
    for b in boxes:
        vx, hy = extract_grid_lines(img, b)
        grid, source = [], ""
        if len(vx) >= 2 and len(hy) >= 2:
            grid, _ = ocr_table_cells(pil_img, b, vx, hy, lang=lang, psm_cell=6)
            source = "grid_lines"
        if not grid:
            grid = fallback_table_from_tokens(pil_img, b, lang=lang)
            source = "token_fallback"
        if not grid:
            continue
        max_cols = max(len(r) for r in grid) if grid else 0
        if len(grid) < 2 or max_cols < 2:
            continue
        normalized = [r + [""]*(max_cols - len(r)) for r in grid]
        results.append({"type":"table","bbox_px": b, "data": normalized, "source": source})
    return results

# ---------------- Figure extraction ----------------

def extract_figures_from_page(page: fitz.Page, scale: float) -> List[Dict[str, Any]]:
    """
    Extract embedded raster figures (vector graphics are not captured here).
    """
    boxes: List[Tuple[int,int,int,int]] = []
    for info in page.get_images(full=True):
        xref = info[0]
        for rect in page.get_image_rects(xref):
            x0, y0, x1, y1 = int(rect.x0 * scale), int(rect.y0 * scale), int(rect.x1 * scale), int(rect.y1 * scale)
            boxes.append((x0, y0, x1, y1))
    boxes = dedupe_boxes(boxes, thr=0.8)
    return [{"type":"figure","bbox_px": b, "source":"embedded"} for b in boxes]

# ---------------- Page orchestration ----------------

def process_page(page: fitz.Page, dpi: int, lang: str,
                 save_debug_images: bool = False, debug_root: Optional[Path] = None,
                 doc_stem: Optional[str] = None, psm_text: int = 3) -> Dict[str, Any]:
    """
    Process a page: detect tables & figures, extract text paragraphs outside those boxes.
    """
    pil_img, scale = rasterize_page(page, dpi=dpi)
    cv_img = np.array(pil_img)

    # 1) tables
    tables = extract_tables_from_image(pil_img, lang=lang)

    # 2) figures (embedded rasters)
    figures = extract_figures_from_page(page, scale)

    # 3) text on original image, exclude non-text boxes
    exclude = [t["bbox_px"] for t in tables] + [f["bbox_px"] for f in figures]
    paragraphs = extract_text_from_image(pil_img, lang=lang, exclude_boxes_px=exclude, psm_text=psm_text)

    # 4) debug overlays
    if save_debug_images and debug_root and doc_stem is not None:
        ann = cv_img.copy()
        # tables: blue
        for t in tables:
            x0,y0,x1,y1 = map(int, t["bbox_px"])
            cv2.rectangle(ann, (x0,y0), (x1,y1), (255,0,0), 2)
        # figures: green
        for f in figures:
            x0,y0,x1,y1 = map(int, f["bbox_px"])
            cv2.rectangle(ann, (x0,y0), (x1,y1), (0,255,0), 2)
        # paragraphs: red
        for p in paragraphs:
            x0,y0,x1,y1 = map(int, p["bbox_px"])
            cv2.rectangle(ann, (x0,y0), (x1,y1), (0,0,255), 2)
        dbg = Image.fromarray(ann)
        dbg_path = debug_root / f"{doc_stem}_page_{page.number:03d}_annotated.png"
        dbg.save(dbg_path)

    # 5) build page output; convert px -> PDF points by dividing by scale
    content: List[Dict[str, Any]] = []
    for t in tables:
        x0,y0,x1,y1 = t["bbox_px"]
        t["bbox"] = [round(x0/scale,2), round(y0/scale,2), round(x1/scale,2), round(y1/scale,2)]
        content.append(t)
    for f in figures:
        x0,y0,x1,y1 = f["bbox_px"]
        f["bbox"] = [round(x0/scale,2), round(y0/scale,2), round(x1/scale,2), round(y1/scale,2)]
        content.append(f)
    for p in paragraphs:
        x0,y0,x1,y1 = p["bbox_px"]
        p["bbox"] = [round(x0/scale,2), round(y0/scale,2), round(x1/scale,2), round(y1/scale,2)]
        content.append(p)

    # reading order
    content.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    return {
        "page_index": page.number,
        "width": round(page.rect.width, 2),
        "height": round(page.rect.height, 2),
        "content": content
    }

# ---------------- Document orchestration ----------------

def process_pdf(pdf_path: Path, out_dir: Path, dpi: int, lang: str,
                save_debug_images: bool = False, psm_text: int = 3,
                save_tables_csv: bool = False):
    doc = fitz.open(pdf_path)

    out = {
        "pdf_path": str(pdf_path),
        "num_pages": len(doc),
        "pages": []
    }

    debug_root = out_dir if save_debug_images else None
    doc_stem = pdf_path.stem if save_debug_images else None
    out_dir.mkdir(parents=True, exist_ok=True)

    for page in doc:
        page_data = process_page(
            page, dpi, lang,
            save_debug_images=save_debug_images, debug_root=debug_root, doc_stem=doc_stem,
            psm_text=psm_text
        )
        out["pages"].append(page_data)

        # Optional: also dump table CSVs per page
        if save_tables_csv:
            page_dir = out_dir / f"{pdf_path.stem}_page_{page.number:03d}"
            page_dir.mkdir(exist_ok=True)
            t_idx = 0
            for item in page_data["content"]:
                if item.get("type") == "table":
                    data = item.get("data") or []
                    if data:
                        pd.DataFrame(data).to_csv(page_dir / f"table_{t_idx:02d}.csv", index=False, header=False)
                    t_idx += 1

    doc.close()

    out_file = out_dir / f"{pdf_path.stem}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[✓] Saved structured JSON -> {out_file}")

# ---------------- CLI ----------------

def main():
    setup_tesseract_path()
    ap = argparse.ArgumentParser(description="Unified extraction: text paragraphs, tables, embedded figures.")
    ap.add_argument("inputs", nargs="*", help="PDF files or directories. If empty, tries config local_params.gold_input_data_location")
    ap.add_argument("--out", type=str, default="output_unified", help="Output directory")
    ap.add_argument("--dpi", type=int, default=300, help="Rasterization DPI (300–400 typical)")
    ap.add_argument("--lang", type=str, default="eng", help="Tesseract languages, e.g. 'eng' or 'eng+spa'")
    ap.add_argument("--psm-text", type=int, default=3, help="Tesseract PSM for page text (3=auto is robust)")
    ap.add_argument("--debug", action="store_true", help="Save annotated debug PNGs per page")
    ap.add_argument("--tables-csv", action="store_true", help="Also save table CSVs per page")
    args = ap.parse_args()

    inputs = args.inputs
    if not inputs and lp is not None and getattr(lp, "gold_input_data_location", None):
        inputs = [str(lp.gold_input_data_location)]
    elif not inputs:
        raise SystemExit("No inputs provided and no config fallback available.")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    pdfs: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
        else:
            print(f"[warn] Skipping invalid input: {x}")

    print(f"[info] Found {len(pdfs)} PDF(s).")
    for pdf in pdfs:
        print(f"[+] {pdf}")
        try:
            process_pdf(pdf, out_dir, dpi=args.dpi, lang=args.lang,
                        save_debug_images=args.debug, psm_text=args.psm_text,
                        save_tables_csv=args.tables_csv)
        except Exception as e:
            print(f"[err] {pdf}: {e}")

    print("[done] Unified extraction complete.")

if __name__ == "__main__":
    main()
