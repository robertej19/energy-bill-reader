# app.py
import os
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from ingest import ingest_single_pdf
from indexing import Index

APP_DIR = Path(__file__).parent
STATIC_DIR = APP_DIR / "static"
OUT_DIR = APP_DIR / "output"
PDF_NAME = "doc.pdf"

# app.py (add near other imports)
import fitz
from functools import lru_cache

PDF_DPI = 300  # must match the dpi used during extraction
PDF_PATH = STATIC_DIR / PDF_NAME
PAGES_DIR = OUT_DIR / "pages"
PAGES_DIR.mkdir(parents=True, exist_ok=True)



app = FastAPI(title="PDF RAG MVP with Citations")

# serve frontend & pdf
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

index_obj: Optional[Index] = None

class SearchResponse(BaseModel):
    results: list

class AnswerResponse(BaseModel):
    answer: str
    hits: list

def _page_png_path(page: int) -> Path:
    return PAGES_DIR / f"page_{page:04d}_{PDF_DPI}dpi.png"

@lru_cache(maxsize=512)
def render_page_to_png(page: int, dpi: int = PDF_DPI) -> Path:
    """Render the given 1-based page to a PNG at dpi; cache on disk."""
    out = _page_png_path(page)
    if out.exists():
        return out
    doc = fitz.open(str(PDF_PATH))
    try:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"Page out of range: {page}")
        p = doc.load_page(page - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out))
        return out
    finally:
        doc.close()

@app.get("/page_image")
def page_image(page: int):
    """Return a rendered PNG of the page at 300dpi."""
    try:
        png_path = render_page_to_png(page, PDF_DPI)
        return FileResponse(str(png_path), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/", response_class=HTMLResponse)
def index_page():
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)

@app.post("/ingest")
def do_ingest():
    pdf_path = STATIC_DIR / PDF_NAME
    if not pdf_path.exists():
        return JSONResponse({"error": f"Missing {pdf_path}"}, status_code=400)
    info = ingest_single_pdf(str(pdf_path), str(OUT_DIR))
    global index_obj
    index_obj = Index(str(OUT_DIR/"embeddings.npz"), str(OUT_DIR/"meta.jsonl"), pdf_name=PDF_NAME)
    return JSONResponse({"status":"ok","counts":info["count"]})

@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(...), top_k: int = 10):
    global index_obj
    if index_obj is None:
        # lazy load
        emb = OUT_DIR/"embeddings.npz"; meta = OUT_DIR/"meta.jsonl"
        if not (emb.exists() and meta.exists()):
            return JSONResponse({"error":"Index not built. POST /ingest first."}, status_code=400)
        index_obj = Index(str(emb), str(meta), pdf_name=PDF_NAME)
    hits = index_obj.search(query, top_k=top_k)
    return {"results": hits}

@app.get("/answer", response_model=AnswerResponse)
def answer(query: str = Query(...), mode: str = "matches", top_k: int = 6):
    global index_obj
    if index_obj is None:
        emb = OUT_DIR/"embeddings.npz"; meta = OUT_DIR/"meta.jsonl"
        if not (emb.exists() and meta.exists()):
            return JSONResponse({"error":"Index not built. POST /ingest first."}, status_code=400)
        index_obj = Index(str(emb), str(meta), pdf_name=PDF_NAME)

    if mode == "matches":
        hits = index_obj.search(query, top_k=top_k)
        answer = "Top matches:\n" + "\n".join([f"- {h['kind']} (p.{h['page']}) {h['citation_url']}" for h in hits])
        return {"answer": answer, "hits": hits}
    else:
        pack = index_obj.compose_extractive_answer(query, top_k=top_k)
        return {"answer": pack["answer"], "hits": pack["hits"]}
