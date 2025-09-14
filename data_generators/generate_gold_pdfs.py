# generate_gold_pdfs.py
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.units import inch
from pathlib import Path
import textwrap, json, math, random
from PIL import Image
import fitz  # PyMuPDF
from data_generators.image_generator import pdf_to_image_pdf
from config import local_params
from data_generators.schemas import Document

PAGE_SIZES = {"LETTER": LETTER, "A4": A4}

def wrap_text(text, font, size, max_width, c):
    lines = []
    for raw in text.split("\n"):
        if not raw.strip():
            lines.append("")
            continue
        words = raw.split()
        line = ""
        for w in words:
            trial = (line + " " + w).strip()
            if stringWidth(trial, font, size) <= max_width:
                line = trial
            else:
                if line: lines.append(line)
                line = w
        if line: lines.append(line)
    return lines

def draw_paragraph(c, text, x, y, w, line_h, font, size):
    lines = wrap_text(text, font, size, w, c)
    bboxes = []
    cur_y = y
    for line in lines:
        if line:
            c.setFont(font, size)
            c.drawString(x, cur_y, line)
            ww = stringWidth(line, font, size)
            bboxes.append((x, cur_y, x+ww, cur_y+line_h))
        cur_y -= line_h
    # paragraph bbox (union)
    if bboxes:
        x0 = min(b[0] for b in bboxes); y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes); y1 = max(b[3] for b in bboxes)
        return cur_y, (x0, y0, x1, y1)
    else:
        return cur_y, (x, y-line_h, x, y)

def draw_table(c, rows, x, y, col_widths, row_h, font, size, pad=4):
    ncols = max(len(r) for r in rows)
    # auto col widths = equal
    if col_widths == "auto":
        cw = [max(stringWidth(str(cell), font, size) for cell in col)+2*pad
              for col in zip(*([r + [""]*(ncols-len(r))] for r in rows))]
        col_w = [max(cw)]*ncols  # simple equalization
    elif isinstance(col_widths, list):
        col_w = col_widths
    else:
        col_w = [90]*ncols

    table_w = sum(col_w)
    cells_bboxes = []
    cur_y = y
    for r_idx, row in enumerate(rows):
        cx = x
        for c_idx in range(ncols):
            text = str(row[c_idx]) if c_idx < len(row) else ""
            # cell rect
            c.rect(cx, cur_y - row_h, col_w[c_idx], row_h)
            # text
            c.setFont(font, size)
            c.drawString(cx + pad, cur_y - row_h + (row_h - size) / 2, text)
            cells_bboxes.append({
                "r": r_idx, "c": c_idx,
                "bbox": (cx, cur_y - row_h, cx + col_w[c_idx], cur_y)
            })
            cx += col_w[c_idx]
        cur_y -= row_h
    table_bbox = (x, cur_y, x + table_w, y)
    return cur_y, table_bbox, cells_bboxes

def draw_figure_placeholder(c, caption, x, y, w, h, font, size):
    c.rect(x, y - h, w, h)
    if caption:
        c.setFont(font, size)
        c.drawString(x, y - h - size - 2, caption)
    # caption bbox (rough)
    cap_bbox = (x, y - h - size - 2, x + stringWidth(caption or "", font, size), y - h)
    return y - h - size - 6, (x, y - h, x + w, y), cap_bbox

def layout_columns(page_w, page_h, margins, ncols, gutter=18):
    ml, mt, mr, mb = margins
    usable_w = page_w - ml - mr
    col_w = (usable_w - gutter*(ncols-1)) / ncols
    cols = []
    for i in range(ncols):
        x = ml + i*(col_w + gutter)
        y_top = page_h - mt
        cols.append((x, y_top, col_w))
    return cols

def render_spec_to_pdf(doc: Document, out_dir="gold_out", filename="doc"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = doc.meta
    styles = doc.styles
    base_font = styles.base_font
    base_size = styles.base_size
    heading_font = styles.heading_font
    heading_sizes = styles.heading_sizes.model_dump()
    page_size = PAGE_SIZES.get(meta.page_size, LETTER)
    ncols = meta.columns
    margins = meta.margins_pt
    line_gap = getattr(styles, 'line_gap', 1.2)  # line_gap might not be in schema

    pdf_path = out / f"{filename}.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=page_size)
    W, H = page_size

    gold = {"meta": meta.model_dump(), "pages": []}

    for p_idx, page in enumerate(doc.pages):
        # reset to first column on each page
        cols = layout_columns(W, H, margins, ncols)
        col_i = 0; x, y_top, col_w = cols[col_i]
        cursor_y = y_top
        line_h = base_size * line_gap

        page_gold = {"size": [W, H], "elements": []}

        for el in page.elements:
            et = el.type

            # Move to next column/page if needed
            def ensure_space(height_needed):
                nonlocal col_i, x, y_top, col_w, cursor_y, cols, page_gold
                if cursor_y - height_needed < margins[3]:
                    # next column or page
                    if col_i + 1 < len(cols):
                        col_i += 1
                        x, y_top, col_w = cols[col_i]
                        cursor_y = y_top
                    else:
                        gold["pages"].append(page_gold)
                        c.showPage()
                        # new page
                        cols = layout_columns(W, H, margins, ncols)
                        col_i = 0; x, y_top, col_w = cols[col_i]
                        cursor_y = y_top
                        page_gold = {"size":[W,H], "elements":[]}

            if et == "heading":
                lvl = el.level
                size = heading_sizes.get(lvl, base_size+2)
                text = el.text
                ensure_space(size*2)
                c.setFont(heading_font, size)
                c.drawString(x, cursor_y, text)
                bbox = (x, cursor_y, x + stringWidth(text, heading_font, size), cursor_y + size)
                page_gold["elements"].append({"type":"heading","level":lvl,"text":text,"bbox":bbox})
                cursor_y -= size * 1.4

            elif et == "paragraph":
                text = el.text
                # rough estimate lines: wrap first, then ensure space
                lines = wrap_text(text, base_font, base_size, col_w, c)
                need_h = max(line_h * max(1,len(lines)), line_h*1.2)
                ensure_space(need_h)
                cursor_y, bbox = draw_paragraph(c, text, x, cursor_y, col_w, line_h, base_font, base_size)
                page_gold["elements"].append({"type":"paragraph","text":text,"bbox":bbox})
                cursor_y -= line_h * 0.5

            elif et == "table":
                rows = el.rows
                row_h = getattr(el, 'row_h', base_size*1.8)  # row_h not in schema, use default
                ensure_space(row_h * len(rows) + base_size*2)
                cursor_y, tbbox, cells = draw_table(
                    c, rows, x, cursor_y, el.col_widths, row_h,
                    base_font, base_size
                )
                page_gold["elements"].append({"type":"table","rows":len(rows),"cols":max(len(r) for r in rows),
                                              "bbox":tbbox,"cells":cells})
                cursor_y -= base_size

            elif et == "figure":
                fig_h = float(el.height_pt)
                ensure_space(fig_h + base_size*2)
                cursor_y, fbbox, cap_bbox = draw_figure_placeholder(
                    c, el.caption, x, cursor_y, col_w, fig_h,
                    base_font, base_size
                )
                page_gold["elements"].append({"type":"figure","bbox":fbbox,"caption_bbox":cap_bbox})
                cursor_y -= base_size

        gold["pages"].append(page_gold)
        c.showPage()

    c.save()
    gold_path = out / f"{filename}.gold.json"
    gold_path.write_text(json.dumps(gold, indent=2), encoding="utf-8")

    # Convert the PDF to an image-based PDF and overwrite the original
    pdf_to_image_pdf(pdf_path, pdf_path)

    return pdf_path, gold_path

# ---- Example run with 3 varied specs (pretend these came from an LLM) ----
if __name__ == "__main__":
    import random
    random.seed(42)
    specs = []
    for k in range(1):
        specs.append({
          "meta": {"title": f"Doc {k+1}", "page_size": "LETTER", "columns": 1 if k==0 else 2, "margins_pt": [72,72,72,72]},
          "styles": {"base_font": "Helvetica", "base_size": 10, "heading_font": "Helvetica-Bold",
                    "heading_sizes": {"h1": 16, "h2": 13}},
          "pages": [
            {"elements": [
                {"type": "heading", "level": "h1", "text": f"Report #{k+1} - Page 1"},
                {"type": "paragraph", "text": ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * (8+k)) + "\n" + ("Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. " * (5+k))},
                {"type": "table", "rows": [["Year","A","B","C"],["2023","10","11","12"],["2024","12","13","15"],["2025","14","15","16"]], "col_widths": "auto"},
                {"type": "paragraph", "text": "Aliquam erat volutpat. Pellentesque habitant morbi tristique senectus et netus et malesuada fames ac turpis egestas. " * (3+k)},
                {"type": "table", "rows": [["Qtr","X","Y"],["Q1","5","7"],["Q2","8","9"],["Q3","6","4"],["Q4","7","8"]], "col_widths": "auto"},
                {"type": "figure", "caption": "Example figure 1", "height_pt": 120 + 20*k},
                {"type": "paragraph", "text": "Conclusion: Vivamus sagittis lacus vel augue laoreet rutrum."}
            ]},
            {"elements": [
                {"type": "heading", "level": "h1", "text": f"Report #{k+1} - Page 2"},
                {"type": "paragraph", "text": ("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. " * (7+k))},
                {"type": "table", "rows": [["Item","Value"],["Alpha","100"],["Beta","200"],["Gamma","300"]], "col_widths": "auto"},
                {"type": "paragraph", "text": "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. " * (4+k)},
                {"type": "table", "rows": [["Month","Sales"],["Jan","1200"],["Feb","1100"],["Mar","1300"],["Apr","1250"]], "col_widths": "auto"},
                {"type": "figure", "caption": "Example figure 2", "height_pt": 100 + 15*k},
                {"type": "paragraph", "text": "End of page 2: Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."}
            ]}
          ]
        })
    out = local_params.gold_data_location
    out.mkdir(parents=True, exist_ok=True)
    for i, spec in enumerate(specs, 1):
        doc = Document(**spec)  # Convert dict to Document object
        pdf_path, gold_path = render_spec_to_pdf(doc, out_dir=str(out), filename=f"doc_{i}")
        print("Wrote:", pdf_path, "and", gold_path)
