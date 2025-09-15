#!/usr/bin/env python3
import fitz  # PyMuPDF
import pytesseract
from pytesseract import Output
from PIL import Image
import os, argparse
import cv2
import numpy as np
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
    min_area = (w * h) * 0.005  # slightly more permissive than 0.01
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

# ======================= OCR helpers =======================

def parse_conf(c):
    try:
        return float(c)
    except Exception:
        return -1.0

def image_to_tokens(img: Image.Image, lang="eng", psm=3, conf_thr=0.0):
    """Return list of tokens (x,y,w,h,text,conf) from pytesseract.image_to_data."""
    data = pytesseract.image_to_data(
        img, lang=lang,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
        output_type=Output.DICT
    )
    out = []
    n = len(data["text"])
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        conf = parse_conf(data["conf"][i])
        if conf < conf_thr:   # KEEP low conf by default (>= 0)
            continue
        x, y = int(data["left"][i]), int(data["top"][i])
        w, h = int(data["width"][i]), int(data["height"][i])
        out.append((x, y, w, h, text, conf))
    return out

def tokens_filter_outside_boxes(tokens, boxes_px):
    """Keep tokens whose centers are not inside any table box."""
    if not boxes_px:
        return tokens
    kept = []
    for x,y,w,h,txt,conf in tokens:
        cx, cy = x + w/2.0, y + h/2.0
        inside = False
        for X0,Y0,X1,Y1 in boxes_px:
            if (X0 <= cx <= X1) and (Y0 <= cy <= Y1):
                inside = True
                break
        if not inside:
            kept.append((x,y,w,h,txt,conf))
    return kept

# ======================= Text reconstruction =======================

def reconstruct_lines_from_tokens(tokens, max_spaces=12):
    """Build lines with bboxes; dynamic line threshold from median height."""
    if not tokens:
        return []

    # sort top-left
    tokens = sorted(tokens, key=lambda t: (t[1], t[0]))
    heights = [h for _,_,_,h,_,_ in tokens]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h

    # group to tentative lines by y
    lines = []
    cur = [tokens[0]]
    for prev, cur_tok in zip(tokens, tokens[1:]):
        _, y0p, _, hp, _, _ = prev
        _, y0c, _, hc, _, _ = cur_tok
        y1p = y0p + hp
        if y0c - y1p > gap_thr:
            lines.append(cur)
            cur = [cur_tok]
        else:
            cur.append(cur_tok)
    if cur:
        lines.append(cur)

    # within each line, sort by x and compute space via char width heuristic
    out = []
    for line in lines:
        line = sorted(line, key=lambda t: t[0])
        widths = [w/max(1,len(txt)) for _,_,w,_,txt,_ in line]
        cw = max(2.0, float(np.median(widths)) if widths else 8.0)
        parts = []
        prev_right = None
        for x,y,w,h,txt,_ in line:
            if prev_right is None:
                parts.append(txt)
            else:
                gap_px = max(0.0, x - prev_right)
                spaces = int(round(gap_px / cw))
                spaces = 1 if spaces <= 0 else min(spaces, max_spaces)
                parts.append(" " * spaces + txt)
            prev_right = x + w
        x0 = min(x for x,_,_,_,_,_ in line)
        y0 = min(y for _,y,_,_,_,_ in line)
        x1 = max(x+w for x,_,w,_,_,_ in line)
        y1 = max(y+h for _,y,_,h,_,_ in line)
        out.append((x0,y0,x1,y1,"".join(parts)))
    return out  # list of (x0,y0,x1,y1,text)

def join_lines_to_paragraphs(lines):
    """Group consecutive lines into paragraphs using vertical gaps."""
    if not lines:
        return []
    heights = [(y1-y0) for x0,y0,x1,y1,_ in lines]
    med_h = float(np.median(heights)) if heights else 10.0
    gap_thr = 1.4 * med_h
    paras = []
    cur = [lines[0]]
    for prev, cur_line in zip(lines, lines[1:]):
        _, _, _, y1p, _ = prev
        _, y0c, _, _, _ = cur_line
        if y0c - y1p > gap_thr:
            paras.append(cur); cur = [cur_line]
        else:
            cur.append(cur_line)
    if cur: paras.append(cur)
    out = []
    for plines in paras:
        x0 = min(l[0] for l in plines); y0 = min(l[1] for l in plines)
        x1 = max(l[2] for l in plines); y1 = max(l[3] for l in plines)
        text = "\n".join(l[4].replace("\u00ad","") for l in plines).strip()
        if text:
            out.append((x0,y0,x1,y1,text))
    return out

# ======================= Table OCR (cell-aware) =======================

def ocr_table_by_grid(pil_img, table_box, v_lines, h_lines, lang="eng"):
    """OCR each cell (using v/h lines). Return 2D list of cell strings."""
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

def ocr_table_fallback_tokens(pil_img, table_box, lang="eng", conf_thr=0.0):
    """No clear grid; OCR tokens on crop, build rows by Y and columns via vertical projections."""
    x0,y0,x1,y1 = table_box
    crop = pil_img.crop((x0,y0,x1,y1))
    toks = image_to_tokens(crop, lang=lang, psm=6, conf_thr=conf_thr)
    if not toks:
        return []

    # Group to rows (by y)
    toks = sorted(toks, key=lambda t: (t[1], t[0]))
    heights = [h for _,_,_,h,_,_ in toks]
    med_h = float(np.median(heights)) if heights else 12.0
    gap_thr = 0.7 * med_h
    rows = []
    cur = [toks[0]]
    for prev, curtok in zip(toks, toks[1:]):
        _, yp, _, hp, _, _ = prev
        _, yc, _, _, _, _ = curtok
        if yc - (yp + hp) > gap_thr:
            rows.append(cur); cur = [curtok]
        else:
            cur.append(curtok)
    if cur: rows.append(cur)

    # Estimate columns via vertical projection on centers across whole crop
    W = x1 - x0
    xs = np.zeros(W, dtype=np.int32)
    for x,y,w,h,txt,_ in toks:
        cx = int(x + w/2)
        cx = min(max(0, cx), W-1)
        xs[cx] += 1
    v_peaks = _peaks_1d(xs, thr_ratio=0.25, min_gap=max(5, W//80))
    if len(v_peaks) < 2:
        # fallback: treat each row as single string
        return [[" ".join([t[4] for t in row])] for row in rows]

    # Derive column boundaries as midpoints between peaks
    v_peaks = sorted(v_peaks)
    cuts = [0] + [int((a+b)/2) for a,b in zip(v_peaks, v_peaks[1:])] + [W]

    # Assign tokens to columns by center
    grid = []
    for row in rows:
        cols = [[] for _ in range(len(cuts)-1)]
        for x,y,w,h,txt,_ in sorted(row, key=lambda t: t[0]):
            cx = x + w/2
            # find interval
            for j in range(len(cuts)-1):
                if cuts[j] <= cx < cuts[j+1]:
                    cols[j].append(txt)
                    break
        grid.append([" ".join(c).strip() for c in cols])
    return grid

# ======================= Drawing =======================

def draw_regions(page: fitz.Page, scale: float, tables_px):
    """Overlay semi-transparent red rectangles for tables."""
    for x0,y0,x1,y1 in tables_px:
        rect = fitz.Rect(x0/scale, y0/scale, x1/scale, y1/scale)
        page.draw_rect(rect, color=(1,0,0), fill=(1,0,0), fill_opacity=0.3, overlay=True)

# ======================= Main per-PDF routine =======================

def extract_text_from_pdf(pdf_path, output_dir,
                          lang="eng", dpi=300,
                          conf_thr=0.0, # keep 0-confidence tokens too
                          save_table_csv=True):
    doc = fitz.open(pdf_path)
    os.makedirs(output_dir, exist_ok=True)

    base = Path(pdf_path).stem
    out_text = Path(output_dir, base + "_extracted_text.txt")
    out_tabs = Path(output_dir, base + "_extracted_tables.txt")
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

            # 2) page OCR for non-table text
            page_tokens = image_to_tokens(pil, lang=lang, psm=3, conf_thr=conf_thr)
            non_table_tokens = tokens_filter_outside_boxes(page_tokens, table_boxes_px)
            text_lines = reconstruct_lines_from_tokens(non_table_tokens)
            paras = join_lines_to_paragraphs(text_lines)

            f_text.write(f"--- Page {pno+1} ---\n")
            for _,_,_,_, s in paras:
                f_text.write(s + "\n")
            f_text.write("\n")

            # 3) table OCR per table (grid -> cells; else fallback tokens)
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
                        grid = ocr_table_fallback_tokens(pil, box, lang=lang, conf_thr=conf_thr)
                        source = "fallback"

                    # Require at least 2x2 to count as a table; otherwise print best-effort
                    max_cols = max((len(r) for r in grid), default=0)
                    f_tab.write(f"Table {t_idx+1} [{source}] (px: {box})\n")
                    for row in grid:
                        f_tab.write("\t".join(row) + "\n")
                    f_tab.write("---\n")

                    if save_table_csv and grid:
                        # Normalize ragged rows
                        m = max_cols
                        norm = [r + [""]*(m-len(r)) for r in grid]
                        import csv
                        with open(page_dir / f"table_{t_idx:02d}.csv", "w", newline="", encoding="utf-8") as cf:
                            cw = csv.writer(cf)
                            cw.writerows(norm)
                f_tab.write("\n")

    doc.close()
    vis_doc.save(out_vis, garbage=4, deflate=True, clean=True)
    vis_doc.close()
    return str(out_text)

# ======================= CLI =======================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract text + tables from a PDF using OCR.")
    ap.add_argument("pdf_path", type=str,
                    nargs="?", default="input_data/synthetic_data/gold_pdfs/doc_1.pdf")
    ap.add_argument("--output_dir", type=str, default="output/")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--lang", type=str, default="eng")
    ap.add_argument("--conf-thr", type=float, default=0.0,
                    help="Min Tesseract confidence to keep a token (default keeps 0).")
    ap.add_argument("--no-csv", action="store_true", help="Do not save per-table CSV files")
    args = ap.parse_args()

    os.environ.setdefault("TESSERACT_CMD", os.environ.get("TESSERACT_CMD",""))  # optional

    out_file = extract_text_from_pdf(
        args.pdf_path, args.output_dir,
        lang=args.lang, dpi=args.dpi,
        conf_thr=args.conf_thr,
        save_table_csv=(not args.no_csv)
    )
    print(f"[✓] Text saved to {out_file}")
