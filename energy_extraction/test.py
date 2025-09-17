#!/usr/bin/env python3
import argparse, glob, os
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np
import layoutparser as lp

DEFAULT_IN_DIR = "docs"
DEFAULT_OUT_DIR = "out_annotated"
DEFAULT_DPI = 300
DEFAULT_SCORE = 0.50
DEFAULT_DEVICE = "cpu"
DEFAULT_WEIGHTS = ".cache/publaynet/model_final.pth"  # <-- local path you just downloaded

COLOR_MAP = {
    "Text":   ((0.10, 0.45, 0.85), (0.10, 0.45, 0.85)),
    "Title":  ((0.10, 0.75, 0.85), (0.10, 0.75, 0.85)),
    "Table":  ((0.13, 0.70, 0.30), (0.13, 0.70, 0.30)),
    "Figure": ((0.90, 0.20, 0.25), (0.90, 0.20, 0.25)),
}
FILL_OPACITY = 0.18
BORDER_WIDTH = 1.5

def load_layout_model(device: str, weight_path: str):
    weight_path = str(Path(weight_path).expanduser().absolute())
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weights not found at: {weight_path}")

    # Keep the config as an alias (layoutparser resolves this one reliably)
    cfg_alias = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"

    model = lp.Detectron2LayoutModel(
        config_path=cfg_alias,
        model_path=weight_path,  # <-- concrete .pth file, not an alias
        label_map={0:"Text", 1:"Title", 2:"List", 3:"Table", 4:"Figure"},
        extra_config=[
            # low threshold in cfg; we’ll filter after detect()
            "MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.05,
            "MODEL.DEVICE", device,
        ],
    )
    return model

def page_to_image(page: fitz.Page, dpi: int) -> np.ndarray:
    mat = fitz.Matrix(dpi/72, dpi/72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    return img

def px_to_pt_rect(block, dpi: int) -> fitz.Rect:
    s = 72.0 / dpi
    x1, y1, x2, y2 = block.block.x_1*s, block.block.y_1*s, block.block.x_2*s, block.block.y_2*s
    return fitz.Rect(x1, y1, x2, y2)

def draw_annots(page: fitz.Page, boxes: List[Tuple[fitz.Rect, str]]):
    for rect, label in boxes:
        stroke, fill = COLOR_MAP.get(label, ((0, 0, 0), (0, 0, 0)))
        annot = page.add_rect_annot(rect)
        annot.set_colors(stroke=stroke, fill=fill)
        try: annot.set_opacity(FILL_OPACITY)
        except Exception: pass
        try: annot.set_border(width=BORDER_WIDTH)
        except Exception: pass
        annot.info["title"] = label
        annot.update()

def annotate_pdf(pdf_path: Path, out_path: Path, model, dpi: int, score_thresh: float):
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        img = page_to_image(page, dpi)
        layout = model.detect(img)
        keep = [b for b in layout if (b.type in {"Text","Title","Table","Figure"}) and ((b.score or 0) >= score_thresh)]
        rects = [(px_to_pt_rect(b, dpi), b.type) for b in keep]
        draw_annots(page, rects)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    doc.close()

def main():
    ap = argparse.ArgumentParser(description="Annotate PDFs (Detectron2 PubLayNet) with local weights.")
    ap.add_argument("--in_dir", default=DEFAULT_IN_DIR)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument("--score", type=float, default=DEFAULT_SCORE)
    ap.add_argument("--device", choices=["cpu","cuda"], default=DEFAULT_DEVICE)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to model_final.pth")
    args = ap.parse_args()

    pdfs = sorted(glob.glob(str(Path(args.in_dir) / "*.pdf")))
    if not pdfs:
        print(f"[!] No PDFs in {Path(args.in_dir).resolve()}")
        return

    print(f"[+] Loading model on {args.device} with weights {args.weights}")
    model = load_layout_model(args.device, args.weights)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[+] Annotating {len(pdfs)} PDF(s)…")
    for p in pdfs:
        p = Path(p)
        out = out_dir / f"{p.stem}_annotated.pdf"
        print(f"    - {p.name} -> {out.name}")
        annotate_pdf(p, out, model, args.dpi, args.score)
    print("[✓] Done.")

if __name__ == "__main__":
    main()
