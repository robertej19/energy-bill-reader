# extract_pdf.py
from pathlib import Path
from typing import List, Dict, Any
import json

# IMPORT your script's functions:
from extract_text import (
    extract_text_from_pdf,    # we’ll not use the txt output, but we reuse internals
    rasterize_page, image_to_df, filter_df_outside_boxes,
    reconstruct_lines_from_df, join_lines_to_paragraphs,
    detect_table_boxes_on_image, extract_grid_lines,
    ocr_table_by_grid, ocr_table_fallback_tokens
)
import fitz, numpy as np
from PIL import Image

def extract_text_and_tables_jsonl(pdf_path: str, out_jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Returns list of dicts: 
      - para: {kind:"para", page:int, bbox_px:(x0,y0,x1,y1), text:str}
      - table:{kind:"table",page:int,bbox_px, label, caption?, headers?, rows:[[...]]}
    Also writes them to JSONL.
    """
    items: List[Dict[str,Any]] = []
    doc = fitz.open(pdf_path)
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for pno in range(doc.page_count):
            page = doc.load_page(pno)
            pil, _scale = rasterize_page(page, dpi=300)
            img = np.array(pil)

            # tables (pixel coords)
            table_boxes_px = detect_table_boxes_on_image(img)

            # non-table text lines/paragraphs
            df_page = image_to_df(pil, lang="eng", psm=3)
            df_page = filter_df_outside_boxes(df_page, table_boxes_px)
            lines = reconstruct_lines_from_df(df_page)
            paras = join_lines_to_paragraphs(lines)

            # paragraphs
            for (x0,y0,x1,y1,text) in paras:
                rec = {"kind":"para", "page": pno+1, "bbox_px":[float(x0),float(y0),float(x1),float(y1)], "text": text}
                items.append(rec); f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # tables
            for t_idx, box in enumerate(table_boxes_px):
                v_lines, h_lines = extract_grid_lines(img, box)
                grid = ocr_table_by_grid(pil, box, v_lines, h_lines, lang="eng")
                if not grid:
                    grid = ocr_table_fallback_tokens(pil, box, lang="eng")

                headers = grid[0] if grid else []
                rows = grid[1:] if len(grid) > 1 else []
                rec = {
                    "kind":"table", "page": pno+1, "bbox_px":[float(b) for b in box],
                    "label": f"table_{t_idx:02d}",
                    "caption": None,             # could add via heuristics later
                    "headers": headers, "rows": rows
                }
                items.append(rec); f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    doc.close()
    return items
