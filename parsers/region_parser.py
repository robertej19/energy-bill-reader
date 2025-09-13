#!/usr/bin/env python3
# regions_parser.py
# A pragmatic, layout-aware PDF -> regions & per-region extraction pipeline.

from __future__ import annotations
import argparse, hashlib, json, uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image
import numpy as np
import pandas as pd
import pytesseract

# -------- Optional integrations (graceful degradation) --------
try:
    import camelot  # vector tables on digital PDFs
    HAVE_CAMELOT = True
except Exception:
    HAVE_CAMELOT = False

try:
    import layoutparser as lparser  # ML fallback for scans
    HAVE_LP = True
except Exception:
    HAVE_LP = False

try:
    # PaddleOCR "PPStructure" Table engine (scanned tables)
    from paddleocr import PPStructure, draw_structure_result, save_structure_res
    HAVE_PADDLE_TABLE = True
except Exception:
    HAVE_PADDLE_TABLE = False

# -------- Optional project config (input directories) --------
try:
    from config import local_params as lp
    HAVE_CFG = True
except Exception:
    HAVE_CFG = False


# ==========================
# Args
# ==========================
def parse_args():
    p = argparse.ArgumentParser(description="Region typing + per-region extraction for PDFs.")
    p.add_argument("--input_path", type=str, default=None,
                   help="PDF file or directory (default: config lp.latex_pdf_directory & lp.image_pdf_directory if present)")
    p.add_argument("--out", type=str, default="out_regions", help="Output root directory")
    p.add_argument("--dpi", type=int, default=350, help="Rasterization DPI (for OCR/ML/fig thumbs)")
    p.add_argument("--lang", type=str, default="eng", help="Tesseract languages (e.g., 'eng' or 'eng+spa')")
    p.add_argument("--psm", type=int, default=3, help="Tesseract PSM (3=auto; 4=single column; 6=single block)")
    p.add_argument("--min-text-chars", type=int, default=30,
                   help="If embedded (digital) text on a page < threshold, treat as scanned page")
    p.add_argument("--enable-camelot", action="store_true", help="Enable Camelot (vector table detection/extraction)")
    p.add_argument("--enable-ml-layout", action="store_true", help="Enable ML fallback (layoutparser PubLayNet)")
    p.add_argument("--enable-paddle-table", action="store_true", help="Enable PaddleOCR table extraction for scans")
    p.add_argument("--save-page-text", action="store_true", help="Dump page.get_text('text') for QA")
    return p.parse_args()


# ==========================
# Utilities
# ==========================
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def discover_pdfs(paths: List[Path]) -> List[Path]:
    pdfs: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_file() and p.suffix.lower() == ".pdf":
            pdfs.append(p)
        elif p.is_dir():
            pdfs.extend(sorted(p.rglob("*.pdf")))
    return pdfs

def rect_area(b: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = b
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)

def iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (rect_area(a) + rect_area(b) - inter + 1e-9)

def clamp_bbox(b: Tuple[float, float, float, float], w: float, h: float):
    x0, y0, x1, y1 = b
    return (float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1)))


# ==========================
# Rendering / OCR
# ==========================
def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def crop_image(page: fitz.Page, bbox_pts: Tuple[float, float, float, float], dpi: int) -> Image.Image:
    """Crop a region by PDF points → PIL image at given dpi."""
    zoom = dpi / 72.0
    rect = fitz.Rect(*bbox_pts)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False, clip=rect)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_region_tokens(pil_img: Image.Image, lang: str, psm: int) -> pd.DataFrame:
    df = pytesseract.image_to_data(
        pil_img, lang=lang, config=f"--oem 3 --psm {psm}",
        output_type=pytesseract.Output.DATAFRAME
    )
    if df.empty:
        return pd.DataFrame(columns=["text", "conf", "left", "top", "width", "height"])
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df["conf"] = pd.to_numeric(df["conf"], errors="coerce").fillna(-1.0)
    df = df[(df["text"] != "") & (df["conf"] > 0)]
    keep = ["text", "conf", "left", "top", "width", "height"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.Series(dtype="float64" if c == "conf" else "int64")
    return df[keep].reset_index(drop=True)

def ocr_tokens_to_page_coords(tokens_df: pd.DataFrame, bbox_pts: Tuple[float, float, float, float], dpi: int) -> pd.DataFrame:
    """Convert OCR pixel coords (within crop) back to PDF points (page space)."""
    zoom = dpi / 72.0
    x0, y0, _, _ = bbox_pts
    out = tokens_df.copy()
    out["x0"] = out["left"] / zoom + x0
    out["y0"] = out["top"] / zoom + y0
    out["x1"] = (out["left"] + out["width"]) / zoom + x0
    out["y1"] = (out["top"] + out["height"]) / zoom + y0
    return out.drop(columns=["left", "top", "width", "height"])


# ==========================
# Page typing / Regions
# ==========================
def get_raw_regions(page: fitz.Page) -> List[Dict[str, Any]]:
    """Prefer 'blocks' for digital text; fall back to 'words'; use image rects for figures."""
    regs: List[Dict[str, Any]] = []

    # ---- A) TEXT via blocks
    try:
        blocks = page.get_text("blocks") or []
        # blocks: (x0, y0, x1, y1, text, block_no, block_type)
        for b in blocks:
            if len(b) >= 5:
                x0, y0, x1, y1, txt = b[:5]
                if (txt or "").strip():
                    regs.append({"label": "text", "bbox": (float(x0), float(y0), float(x1), float(y1)),
                                 "score": 1.0, "source": "pdf_blocks"})
    except Exception:
        pass

    # ---- B) TEXT fallback via words (cluster into coarse blocks)
    if not any(r["label"] == "text" for r in regs):
        words = page.get_text("words") or []
        # words: (x0,y0,x1,y1,"word", block_no, line_no, word_no)
        if words:
            # simple single-region fallback: cover all words
            x0 = min(w[0] for w in words); y0 = min(w[1] for w in words)
            x1 = max(w[2] for w in words); y1 = max(w[3] for w in words)
            regs.append({"label": "text", "bbox": (float(x0), float(y0), float(x1), float(y1)),
                         "score": 0.8, "source": "pdf_words_all"})

            # (optional) smarter: cluster words into multiple regions by column/line

    # ---- C) IMAGES via image rects (more robust than rawdict)
    try:
        for xref, *_ in page.get_images(full=True):
            for r in page.get_image_rects(xref) or []:
                bbox = (float(r.x0), float(r.y0), float(r.x1), float(r.y1))
                regs.append({"label": "figure", "bbox": bbox, "score": 0.7, "source": "pdf_image_rect"})
    except Exception:
        pass

    return regs


def drawings_as_figures(page: fitz.Page) -> List[Dict[str, Any]]:
    regs: List[Dict[str, Any]] = []
    try:
        for d in page.get_drawings() or []:
            rect = d.get("rect", None)
            if rect is None: continue
            bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
            if rect_area(bbox) > 5000:
                regs.append({"label":"figure","bbox":bbox,"score":0.5,"source":"vector_draw"})
    except Exception:
        pass
    return regs

def camelot_table_regions(pdf_path: Path, page_index: int) -> List[Dict[str, Any]]:
    regs: List[Dict[str, Any]] = []
    if not HAVE_CAMELOT:
        return regs
    # Try lattice then stream
    for flavor, score in [("lattice", 0.9), ("stream", 0.6)]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_index+1), flavor=flavor)
            for t in tables:
                bbox = getattr(t, "_bbox", None)
                if bbox and len(bbox) == 4:
                    regs.append({"label":"table","bbox":tuple(bbox),"score":score,"source":f"camelot_{flavor}"})
        except Exception:
            continue
        if regs:
            break
    return regs

def ml_layout_regions(pil_page: Image.Image, dpi: int) -> List[Dict[str, Any]]:
    """layoutparser PubLayNet model → text/table/figure boxes. Optional heavy dependency."""
    regs: List[Dict[str, Any]] = []
    if not HAVE_LP:
        return regs
    try:
        model = lparser.Detectron2LayoutModel(
            config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
            label_map={0:"Text",1:"Title",2:"List",3:"Table",4:"Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
            device="cpu"
        )
        layout = model.detect(pil_page)
        zoom = dpi / 72.0
        for l in layout:
            x0, y0, x1, y1 = float(l.block.x_1), float(l.block.y_1), float(l.block.x_2), float(l.block.y_2)
            bbox_pts = (x0/zoom, y0/zoom, x1/zoom, y1/zoom)
            label = l.type.lower()
            if label in ("title","list"): label = "text"
            if label in {"text","table","figure"}:
                regs.append({"label":label,"bbox":bbox_pts,"score":float(l.score),"source":"ml_publaynet"})
    except Exception:
        pass
    return regs

def merge_regions(a: List[Dict[str, Any]], b: List[Dict[str, Any]], iou_thr: float = 0.3) -> List[Dict[str, Any]]:
    order = {"table": 3, "figure": 2, "text": 1, "other": 0}
    out = list(a)
    for r in b:
        keep = True
        for q in out:
            if iou(tuple(r["bbox"]), tuple(q["bbox"])) > iou_thr and order.get(q["label"],0) >= order.get(r["label"],0):
                keep = False; break
        if keep: out.append(r)
    return out


# ==========================
# Digital text slicing
# ==========================
def extract_text_words_in_bbox(page: fitz.Page, bbox: Tuple[float,float,float,float]) -> str:
    """Use page.get_text('words') and aggregate words inside bbox into lines."""
    words = page.get_text("words") or []
    x0, y0, x1, y1 = bbox
    sel = [w for w in words if (w[0] >= x0 and w[1] >= y0 and w[2] <= x1 and w[3] <= y1)]
    if not sel:
        return ""
    # sort by (y, x); group by line proximity
    sel.sort(key=lambda w: (w[1], w[0]))
    lines: List[List[str]] = []
    cur_y = None
    cur_line: List[str] = []
    line_tol = 3.0  # points tolerance to join words on same line
    for w in sel:
        wy = w[1]
        text = w[4]
        if cur_y is None or abs(wy - cur_y) <= line_tol:
            cur_line.append(text)
            cur_y = wy if cur_y is None else (cur_y + wy)/2
        else:
            lines.append(cur_line)
            cur_line = [text]
            cur_y = wy
    if cur_line:
        lines.append(cur_line)
    return "\n".join(" ".join(tokens) for tokens in lines)


# ==========================
# Tables extraction
# ==========================
def extract_table_digital(pdf_path: Path, page_index: int, region_bbox: Tuple[float,float,float,float]) -> Tuple[Optional[pd.DataFrame], str, Optional[float]]:
    """Use Camelot to extract a single table overlapping the region bbox. Returns (df, method, quality_score)."""
    if not HAVE_CAMELOT:
        return None, "camelot_unavailable", None

    best_df, best_method, best_score = None, None, None
    for flavor, score in [("lattice", 0.9), ("stream", 0.6)]:
        try:
            tables = camelot.read_pdf(str(pdf_path), pages=str(page_index+1), flavor=flavor)
            for t in tables:
                bbox = getattr(t, "_bbox", None)
                if not bbox or len(bbox) != 4:  # cannot match without bbox
                    continue
                if iou(tuple(bbox), region_bbox) > 0.2:
                    df = t.df  # pandas DataFrame
                    if not df.empty and (best_df is None or score > (best_score or 0)):
                        best_df, best_method, best_score = df, f"camelot_{flavor}", score
        except Exception:
            continue
    return best_df, (best_method or "camelot_no_match"), best_score

def extract_table_scanned(pil_region: Image.Image) -> Tuple[Optional[pd.DataFrame], str, Optional[float]]:
    """Use PaddleOCR PPStructure 'table' if available; else return None."""
    if not HAVE_PADDLE_TABLE:
        return None, "paddle_unavailable", None
    try:
        table_engine = PPStructure(show_log=False, image_orientation=False, det=True, rec=True, table=True, ocr=True)
        # Run
        res = table_engine(np.array(pil_region))
        # Convert to DataFrame (very naive: join cells row-wise)
        # PPStructure returns list of dicts per element; for table, each has 'res' with 'html' and cells
        # Here we return a CSV-like rendering from 'html' if present.
        for el in res:
            if el.get("type") == "table" and "res" in el and "html" in el["res"]:
                # A simple HTML-to-rows fallback: strip tags and split lines (for demo)
                text = el["res"]["html"]
                # crude extraction: replace </tr> with newline and strip tags -> demo only
                import re
                rows = [re.sub("<[^<]+?>", "", r).strip() for r in text.split("</tr>") if r.strip()]
                rows = [r for r in rows if r]
                if rows:
                    df = pd.DataFrame([row.split() for row in rows])
                    return df, "paddle_table", 0.7
        return None, "paddle_no_table", None
    except Exception:
        return None, "paddle_error", None


# ==========================
# Per-PDF processing
# ==========================
def process_pdf(pdf_path: Path, out_root: Path, dpi: int, lang: str, psm: int,
                min_text_chars: int, enable_camelot: bool, enable_ml_layout: bool,
                enable_paddle_table: bool, save_page_text: bool) -> Dict[str, Any]:

    out_dir = out_root / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    text_dir = out_dir / "page_text"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    if save_page_text: text_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    doc = fitz.open(pdf_path)
    doc_id = sha256_file(pdf_path)[:16]

    regions_rows: List[Dict[str, Any]] = []
    text_rows: List[Dict[str, Any]] = []
    token_rows: List[Dict[str, Any]] = []
    table_rows: List[Dict[str, Any]] = []
    figure_rows: List[Dict[str, Any]] = []
    page_meta: List[Dict[str, Any]] = []

    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height

        embedded_text = page.get_text("text") or ""
        if save_page_text:
            (text_dir / f"page_{i:04d}.txt").write_text(embedded_text, encoding="utf-8")

        is_scanned = len(embedded_text.strip()) < min_text_chars

        # --- Detect regions
        regs_raw = get_raw_regions(page)
        regs_draw = drawings_as_figures(page)
        regs = merge_regions(regs_raw, regs_draw, iou_thr=0.3)

        if enable_camelot and HAVE_CAMELOT and not is_scanned:
            regs_tab = camelot_table_regions(pdf_path, i)
            regs = merge_regions(regs_tab, regs, iou_thr=0.2)  # tables win

        if (is_scanned or not regs) and enable_ml_layout and HAVE_LP:
            pil = render_page(page, dpi=dpi)
            regs_ml = ml_layout_regions(pil, dpi=dpi)
            regs = merge_regions(regs_ml, regs, iou_thr=0.3)

        # clamp + filter tiny
        clamped = []
        for r in regs:
            x0, y0, x1, y1 = clamp_bbox(tuple(r["bbox"]), w, h)
            if rect_area((x0,y0,x1,y1)) > 1500:
                r2 = dict(r)
                r2["bbox"] = (x0,y0,x1,y1)
                clamped.append(r2)
        regs = clamped

        # --- Extract per region
        for k, r in enumerate(regs):
            rid = f"{i}_{k}"
            label = r["label"]
            x0, y0, x1, y1 = r["bbox"]
            score = r.get("score", None)
            source = r.get("source", "heuristic")

            regions_rows.append({
                "doc_id": doc_id, "pdf_name": pdf_path.name,
                "page_index": i, "region_id": rid,
                "label": label, "score": score, "source": source,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "is_scanned_page": bool(is_scanned),
            })

            # TEXT
            if label == "text":
                if not is_scanned:
                    content = extract_text_words_in_bbox(page, (x0,y0,x1,y1))
                    text_rows.append({
                        "doc_id": doc_id, "pdf_name": pdf_path.name,
                        "page_index": i, "region_id": rid,
                        "source": "pdf",
                        "text": content
                    })
                else:
                    # OCR the text region
                    pil_reg = crop_image(page, (x0,y0,x1,y1), dpi=dpi)
                    toks = ocr_region_tokens(pil_reg, lang=lang, psm=psm)
                    if not toks.empty:
                        toks_page = ocr_tokens_to_page_coords(toks, (x0,y0,x1,y1), dpi=dpi)
                        for row in toks_page.itertuples():
                            token_rows.append({
                                "doc_id": doc_id, "pdf_name": pdf_path.name,
                                "page_index": i, "region_id": rid, "source": "ocr",
                                "text": row.text, "conf": float(row.conf),
                                "x0": float(row.x0), "y0": float(row.y0),
                                "x1": float(row.x1), "y1": float(row.y1)
                            })
                        # also concatenate tokens to region text
                        content = " ".join(toks_page["text"].tolist())
                        text_rows.append({
                            "doc_id": doc_id, "pdf_name": pdf_path.name,
                            "page_index": i, "region_id": rid,
                            "source": "ocr",
                            "text": content
                        })

            # TABLE
            elif label == "table":
                df_table: Optional[pd.DataFrame] = None
                method = "unknown"; qscore = None
                if not is_scanned and enable_camelot and HAVE_CAMELOT:
                    df_table, method, qscore = extract_table_digital(pdf_path, i, (x0,y0,x1,y1))
                if df_table is None and enable_paddle_table and HAVE_PADDLE_TABLE:
                    pil_reg = crop_image(page, (x0,y0,x1,y1), dpi=dpi)
                    df_table, method, qscore = extract_table_scanned(pil_reg)

                csv_path = None
                if isinstance(df_table, pd.DataFrame) and not df_table.empty:
                    csv_path = tables_dir / f"{pdf_path.stem}_p{i:04d}_r{rid}_table.csv"
                    df_table.to_csv(csv_path, index=False, header=False)

                table_rows.append({
                    "doc_id": doc_id, "pdf_name": pdf_path.name,
                    "page_index": i, "region_id": rid,
                    "method": method, "quality_score": qscore,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "csv_path": str(csv_path) if csv_path else None
                })

                # For retrieval, you might also store a text rendering (TSV-ish)
                if csv_path is None and df_table is not None:
                    # write a minimal TSV-ish text for RAG
                    tsv_text = "\n".join(["\t".join(map(str, row)) for row in df_table.values.tolist()])
                    text_rows.append({
                        "doc_id": doc_id, "pdf_name": pdf_path.name,
                        "page_index": i, "region_id": rid,
                        "source": method,
                        "text": tsv_text
                    })

            # FIGURE
            elif label == "figure":
                # We only save a thumbnail and index; no OCR by default
                pil_thumb = crop_image(page, (x0,y0,x1,y1), dpi=min(dpi, 200))
                thumb_path = figures_dir / f"{pdf_path.stem}_p{i:04d}_r{rid}_figure.png"
                pil_thumb.save(thumb_path)
                figure_rows.append({
                    "doc_id": doc_id, "pdf_name": pdf_path.name,
                    "page_index": i, "region_id": rid,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "thumb_path": str(thumb_path)
                })

        page_meta.append({
            "page_index": i,
            "width_pt": w, "height_pt": h,
            "embedded_text_chars": len(embedded_text),
            "is_scanned_page": bool(is_scanned),
            "n_regions": len(regs),
        })

    # ---------- Persist ----------
    regions_df = pd.DataFrame(regions_rows, columns=[
        "doc_id","pdf_name","page_index","region_id","label","score","source","x0","y0","x1","y1","is_scanned_page"
    ])
    text_df = pd.DataFrame(text_rows, columns=[
        "doc_id","pdf_name","page_index","region_id","source","text"
    ])
    tokens_df = pd.DataFrame(token_rows, columns=[
        "doc_id","pdf_name","page_index","region_id","source","text","conf","x0","y0","x1","y1"
    ])
    tables_df = pd.DataFrame(table_rows, columns=[
        "doc_id","pdf_name","page_index","region_id","method","quality_score","x0","y0","x1","y1","csv_path"
    ])
    figures_df = pd.DataFrame(figure_rows, columns=[
        "doc_id","pdf_name","page_index","region_id","x0","y0","x1","y1","thumb_path"
    ])

    out_dir.mkdir(parents=True, exist_ok=True)
    # Parquet (primary)
    regions_df.to_parquet(out_dir / f"{pdf_path.stem}_regions.parquet", index=False)
    text_df.to_parquet(out_dir / f"{pdf_path.stem}_text_regions.parquet", index=False)
    tokens_df.to_parquet(out_dir / f"{pdf_path.stem}_tokens.parquet", index=False)
    tables_df.to_parquet(out_dir / f"{pdf_path.stem}_tables.parquet", index=False)
    figures_df.to_parquet(out_dir / f"{pdf_path.stem}_figures.parquet", index=False)
    # CSV copies (helpful for quick peeks)
    regions_df.to_csv(out_dir / f"{pdf_path.stem}_regions.csv", index=False)
    text_df.to_csv(out_dir / f"{pdf_path.stem}_text_regions.csv", index=False)
    tables_df.to_csv(out_dir / f"{pdf_path.stem}_tables.csv", index=False)
    figures_df.to_csv(out_dir / f"{pdf_path.stem}_figures.csv", index=False)

    summary = {
        "doc_id": doc_id,
        "pdf_path": str(pdf_path),
        "num_pages": len(doc),
        "pages": page_meta,
        "outputs": {
            "regions_parquet": str(out_dir / f"{pdf_path.stem}_regions.parquet"),
            "text_regions_parquet": str(out_dir / f"{pdf_path.stem}_text_regions.parquet"),
            "tokens_parquet": str(out_dir / f"{pdf_path.stem}_tokens.parquet"),
            "tables_parquet": str(out_dir / f"{pdf_path.stem}_tables.parquet"),
            "figures_parquet": str(out_dir / f"{pdf_path.stem}_figures.parquet"),
            "tables_dir": str(tables_dir),
            "figures_dir": str(figures_dir),
            "page_text_dir": str(out_dir / "page_text") if save_page_text else None,
        },
        "features": {
            "camelot": bool(enable_camelot and HAVE_CAMELOT),
            "ml_layout": bool(enable_ml_layout and HAVE_LP),
            "paddle_table": bool(enable_paddle_table and HAVE_PADDLE_TABLE),
        }
    }
    (out_dir / f"{pdf_path.stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    doc.close()
    return summary


# ==========================
# Main
# ==========================
def main():
    read_real = True
    args = parse_args()
    if args.input_path:
        in_paths = [Path(args.input_path)]
    elif read_real:
        in_paths = [lp.real_pdf_directory,]
    else:
        if HAVE_CFG:
            in_paths = [lp.latex_pdf_directory, lp.image_pdf_directory, lp.real_pdf_directory]
        else:
            in_paths = [Path("pdfs")]

    out_root = Path(args.out); out_root.mkdir(parents=True, exist_ok=True)
    pdfs = discover_pdfs(in_paths)
    if not pdfs:
        raise SystemExit(f"No PDFs found under: {in_paths}")

    index = []
    for pdf in pdfs:
        print(f"[+] {pdf}")
        summary = process_pdf(
            pdf_path=pdf,
            out_root=out_root,
            dpi=args.dpi,
            lang=args.lang,
            psm=args.psm,
            min_text_chars=args.min_text_chars,
            enable_camelot=args.enable_camelot,
            enable_ml_layout=args.enable_ml_layout,
            enable_paddle_table=args.enable_paddle_table,
            save_page_text=args.save_page_text,
        )
        index.append(summary)

    (out_root / "run_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[✓] Done -> {out_root.resolve()}")


if __name__ == "__main__":
    main()
