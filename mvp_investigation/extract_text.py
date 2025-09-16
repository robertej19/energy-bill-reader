#!/usr/bin/env python3
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import os, argparse, csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# ======================= Rasterization =======================

def rasterize_page(page: fitz.Page, dpi: int = 300):
    """Rasterize a PDF page to PIL.Image and return (image, scale px/pt)."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, zoom

# ======================= Table detection =======================

def _binarize(gray: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 255 - th  # invert so lines are white

def detect_table_boxes_on_image(img_rgb: np.ndarray):
    """Detect table bounding boxes (x0,y0,x1,y1) in pixel coords."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    bin_inv = _binarize(gray)

    h_scale = max(10, w // 100)
    v_scale = max(10, h // 100)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (h_scale, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_scale))

    horiz = cv2.dilate(cv2.erode(bin_inv, hk, 1), hk, 1)
    vert  = cv2.dilate(cv2.erode(bin_inv, vk, 1), vk, 1)
    lines = cv2.bitwise_or(horiz, vert)

    contours, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = (w * h) * 0.005  # permissive
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        if ww * hh < min_area:
            continue
        roi = lines[y:y+hh, x:x+ww]
        if (roi > 0).mean() < 0.02:
            continue
        boxes.append((x, y, x + ww, y + hh))
    boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes

def _peaks_1d(arr, thr_ratio=0.35, min_gap=5):
    thr = arr.max() * thr_ratio
    idx = np.where(arr >= thr)[0]
    if idx.size == 0:
        return []
    groups, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            groups.append((start, prev))
            start = i
        prev = i
    groups.append((start, prev))
    return [int((a+b)//2) for a,b in groups]

def extract_grid_lines(img_rgb: np.ndarray, box):
    """Return (v_lines, h_lines) inside the box (pixel offsets relative to box origin)."""
    x0,y0,x1,y1 = box
    crop = img_rgb[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    binv = _binarize(gray)
    h,w = binv.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//20), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//20)))
    horiz = cv2.dilate(cv2.erode(binv, hk, 1), hk, 1)
    vert  = cv2.dilate(cv2.erode(binv, vk, 1), vk, 1)
    ys = (horiz > 0).sum(axis=1)
    xs = (vert  > 0).sum(axis=0)
    h_lines = _peaks_1d(ys, thr_ratio=0.35, min_gap=max(3, h//200))
    v_lines = _peaks_1d(xs, thr_ratio=0.35, min_gap=max(3, w//200))
    return v_lines, h_lines

# ======================= OCR (DataFrame, keeps line ids) =======================

def image_to_df(img: Image.Image, lang="eng", psm=3) -> pd.DataFrame:
    """Return pytesseract DATAFRAME (keeps block/par/line ids)."""
    df = pytesseract.image_to_data(
        img, lang=lang,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
        output_type=Output.DATAFRAME
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "level","page_num","block_num","par_num","line_num","word_num",
            "left","top","width","height","conf","text"
        ])
    df = df.dropna(subset=["text"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    # keep >= 0 (0 often valid); drop blanks
    df = df[(df["text"] != "") & (df["conf"] >= 0)]
    # ensure columns present
    for c in ("block_num","par_num","line_num","left","top","width","height"):
        if c not in df.columns:
            df[c] = 0
    return df

def filter_df_outside_boxes(df: pd.DataFrame, boxes_px):
    """Filter out tokens whose centers fall inside any excluded box."""
    if df.empty or not boxes_px:
        return df
    cx = df["left"].values + df["width"].values/2.0
    cy = df["top"].values  + df["height"].values/2.0
    keep = np.ones(len(df), dtype=bool)
    for x0,y0,x1,y1 in boxes_px:
        inside = (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)
        keep &= ~inside
    return df.loc[keep].copy()

# ======================= Line & paragraph reconstruction =======================

def reconstruct_lines_from_df(df: pd.DataFrame, max_spaces=12):
    """
    Reconstruct lines using Tesseract's line ids to avoid cross-row mixing.
    Returns list of dicts: {text, bbox_px}
    """
    if df.empty:
        return []

    # Sort by block / paragraph / line / x (left)
    df = df.sort_values(["block_num","par_num","line_num","left"]).copy()

    lines = []
    for (blk, par, ln), g in df.groupby(["block_num","par_num","line_num"], sort=True):
        g = g.sort_values("left")
        # char width estimate
        widths = g["width"].astype(float).values
        chars  = g["text"].str.len().replace(0,1).astype(float).values
        cw_med = max(2.0, float(np.median(widths / chars)) if len(g) else 8.0)

        parts = []
        prev_right = None
        for _, r in g.iterrows():
            word = r["text"]
            left = float(r["left"]); width = float(r["width"])
            if prev_right is None:
                parts.append(word)
            else:
                gap_px = max(0.0, left - prev_right)
                spaces = int(round(gap_px / cw_med))
                spaces = 1 if spaces <= 0 else min(spaces, max_spaces)
                parts.append(" " * spaces + word)
            prev_right = left + width

        x0 = float(g["left"].min()); y0 = float(g["top"].min())
        x1 = float((g["left"] + g["width"]).max()); y1 = float((g["top"] + g["height"]).max())
        lines.append({"bbox_px": (x0,y0,x1,y1), "text": "".join(parts)})

    # reading order among lines is preserved by sort above
    return lines

def join_lines_to_paragraphs(lines):
    """Group consecutive lines into paragraphs using vertical gaps."""
    if not lines:
        return []
    heights = [y1-y0 for (x0,y0,x1,y1) in (L["bbox_px"] for L in lines)]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h

    paras = []
    cur = [lines[0]]
    for prev, cur_line in zip(lines, lines[1:]):
        _, y0p, _, y1p = prev["bbox_px"]
        x0c, y0c, _, _ = cur_line["bbox_px"]
        if y0c - y1p > gap_thr:
            paras.append(cur); cur = [cur_line]
        else:
            cur.append(cur_line)
    if cur: paras.append(cur)

    out = []
    for pl in paras:
        x0 = min(L["bbox_px"][0] for L in pl)
        y0 = min(L["bbox_px"][1] for L in pl)
        x1 = max(L["bbox_px"][2] for L in pl)
        y1 = max(L["bbox_px"][3] for L in pl)
        text = "\n".join(L["text"].replace("\u00ad","") for L in pl).strip()
        if text:
            out.append((x0,y0,x1,y1,text))
    return out

# ======================= Table OCR (cell-aware + fallback) =======================

def ocr_table_by_grid(pil_img, table_box, v_lines, h_lines, lang="eng"):
    x0,y0,x1,y1 = table_box
    if len(v_lines) < 2 or len(h_lines) < 2:
        return []
    vx = sorted(set(v_lines)); hy = sorted(set(h_lines))
    grid = []
    for r in range(len(hy)-1):
        row = []
        ya = y0 + hy[r]; yb = y0 + hy[r+1]
        for c in range(len(vx)-1):
            xa = x0 + vx[c]; xb = x0 + vx[c+1]
            pad = 2
            xa, ya2 = min(xb-1, xa+pad), min(yb-1, ya+pad)
            xb2, yb2 = max(xa+1, xb-pad), max(ya2+1, yb-pad)
            crop = pil_img.crop((xa, ya2, xb2, yb2))
            txt = pytesseract.image_to_string(
                crop, lang=lang, config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
            ).strip()
            row.append(txt)
        grid.append(row)
    return grid

def ocr_table_fallback_tokens(pil_img, table_box, lang="eng"):
    x0,y0,x1,y1 = table_box
    crop = pil_img.crop((x0,y0,x1,y1))
    # Use DF here too to keep line structure if available
    df = image_to_df(crop, lang=lang, psm=6)
    if df.empty:
        return []
    # Rows by line_num primarily; fallback to y clustering if needed
    if "line_num" in df.columns and df["line_num"].nunique() >= 1:
        dfs = df.sort_values(["block_num","par_num","line_num","left"])
        rows = [g["text"].tolist() for _, g in dfs.groupby(["block_num","par_num","line_num"])]
        # columnization: simple multi-space split on joined strings (heuristic)
        grid = []
        for toks in rows:
            s = " ".join(toks)
            cols = [c.strip() for c in re_split_multi_space(s)]
            grid.append(cols if cols else [s])
        return grid
    else:
        # Projection fallback
        data = []
        for _, r in df.iterrows():
            data.append((int(r["left"]), int(r["top"]), int(r["width"]), int(r["height"]), r["text"], float(r["conf"])))
        # group rows
        data = sorted(data, key=lambda t: (t[1], t[0]))
        heights = [h for _,_,_,h,_,_ in data]
        med_h = float(np.median(heights)) if heights else 12.0
        gap_thr = 0.7 * med_h
        rows, cur = [], [data[0]]
        for prev, curtok in zip(data, data[1:]):
            _, yp, _, hp, _, _ = prev
            _, yc, _, _, _, _ = curtok
            if yc - (yp + hp) > gap_thr:
                rows.append(cur); cur = [curtok]
            else:
                cur.append(curtok)
        if cur: rows.append(cur)
        # columns via vertical projections
        W = x1 - x0
        xs = np.zeros(W, dtype=np.int32)
        for x,y,w,h,txt,_ in data:
            cx = int(x + w/2); cx = min(max(0, cx), W-1)
            xs[cx] += 1
        v_peaks = _peaks_1d(xs, thr_ratio=0.25, min_gap=max(5, W//80))
        if len(v_peaks) < 2:
            return [[" ".join([t[4] for t in row])] for row in rows]
        v_peaks = sorted(v_peaks)
        cuts = [0] + [int((a+b)/2) for a,b in zip(v_peaks, v_peaks[1:])] + [W]
        grid = []
        for row in rows:
            cols = [[] for _ in range(len(cuts)-1)]
            for x,_,w,_,txt,_ in sorted(row, key=lambda t: t[0]):
                cx = x + w/2
                for j in range(len(cuts)-1):
                    if cuts[j] <= cx < cuts[j+1]:
                        cols[j].append(txt); break
            grid.append([" ".join(c).strip() for c in cols])
        return grid

import re
def re_split_multi_space(s: str):
    return re.split(r"\s{2,}|\t+", s)

# ======================= Drawing =======================

def draw_regions(page: fitz.Page, scale: float, tables_px):
    """Overlay semi-transparent red rectangles for tables."""
    for x0,y0,x1,y1 in tables_px:
        rect = fitz.Rect(x0/scale, y0/scale, x1/scale, y1/scale)
        page.draw_rect(rect, color=(1,0,0), fill=(1,0,0), fill_opacity=0.3, overlay=True)

# ======================= Main per-PDF routine =======================

def extract_text_from_pdf(pdf_path, output_dir,
                          lang="eng", dpi=300,
                          save_table_csv=True):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    base = Path(pdf_path).stem
    out_text = Path(output_dir, base + "_extracted_text.txt")     # includes placeholders
    out_tabs = Path(output_dir, base + "_extracted_tables.txt")   # debug: raw table dumps
    out_vis  = Path(output_dir, base + "_regions.pdf")

    vis_doc = fitz.open()

    with open(out_text, "w", encoding="utf-8") as f_text, open(out_tabs, "w", encoding="utf-8") as f_tab:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            vis_page = vis_doc.new_page(width=page.rect.width, height=page.rect.height)
            vis_page.show_pdf_page(vis_page.rect, doc, pno)

            pil, scale = rasterize_page(page, dpi=dpi)
            img = np.array(pil)

            # 1) detect tables
            table_boxes_px = detect_table_boxes_on_image(img)
            draw_regions(vis_page, scale, table_boxes_px)

            # 2) page OCR DF for non-table text (keeps line ids)
            df_page = image_to_df(pil, lang=lang, psm=3)
            df_page = filter_df_outside_boxes(df_page, table_boxes_px)
            lines = reconstruct_lines_from_df(df_page)
            paras = join_lines_to_paragraphs(lines)  # [(x0,y0,x1,y1,text)]

            # 3) per-table OCR & CSV; build table items with label + bbox
            table_items = []  # [{bbox_px, label, csv_path}]
            if table_boxes_px:
                f_tab.write(f"--- Tables from Page {pno+1} ---\n")
                page_dir = Path(output_dir, f"{base}_page_{pno:03d}")
                if save_table_csv:
                    page_dir.mkdir(parents=True, exist_ok=True)

                for t_idx, box in enumerate(table_boxes_px):
                    v_lines, h_lines = extract_grid_lines(img, box)
                    grid = ocr_table_by_grid(pil, box, v_lines, h_lines, lang=lang)
                    source = "grid"
                    if not grid:
                        grid = ocr_table_fallback_tokens(pil, box, lang=lang)
                        source = "fallback"

                    label = f"table_{t_idx:02d}.csv"
                    csv_path = page_dir / label if save_table_csv else None

                    # Debug dump + CSV
                    f_tab.write(f"{label} [{source}] (px: {box})\n")
                    max_cols = max((len(r) for r in grid), default=0)
                    for row in grid:
                        f_tab.write("\t".join(row) + "\n")
                    f_tab.write("---\n")

                    if save_table_csv and grid:
                        m = max_cols
                        norm = [r + [""]*(m-len(r)) for r in grid]
                        with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                            cw = csv.writer(cf)
                            cw.writerows(norm)

                    table_items.append({
                        "bbox_px": box,
                        "label": label,
                        "csv_path": (str(csv_path) if csv_path else None),
                    })
                f_tab.write("\n")

            # 4) Merge paragraphs and table placeholders by reading order (y0,x0)
            stream_items = []
            for x0,y0,x1,y1,text in paras:
                stream_items.append(("para", (x0,y0,x1,y1), text))
            for ti in table_items:
                x0,y0,x1,y1 = ti["bbox_px"]
                stream_items.append(("table", (x0,y0,x1,y1), ti["label"]))
            stream_items.sort(key=lambda it: (it[1][1], it[1][0]))

            # 5) Write merged stream (placeholders inline)
            f_text.write(f"--- Page {pno+1} ---\n")
            for kind, bbox, payload in stream_items:
                if kind == "para":
                    f_text.write(payload + "\n")
                else:
                    f_text.write(payload + "\n")
            f_text.write("\n")

    doc.close()
    vis_doc.save(out_vis, garbage=4, deflate=True, clean=True)
    vis_doc.close()
    return str(out_text)

# ======================= CLI =======================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract text + tables with correct line order and inline table placeholders.")
    ap.add_argument("pdf_path", type=str, nargs="?", default="input_data/synthetic_data/gold_pdfs/doc_1.pdf")
    ap.add_argument("--output_dir", type=str, default="output/")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--no-csv", action="store_true", help="Do not save per-table CSV files")
    args = ap.parse_args()

    out_file = extract_text_from_pdf(
        args.pdf_path, args.output_dir,
        lang=args.lang, dpi=args.dpi,
        save_table_csv=(not args.no_csv)
    )
    print(f"[✓] Text (with placeholders, correct line order) saved to {out_file}")
