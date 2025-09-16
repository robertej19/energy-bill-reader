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

app = FastAPI(title="PDF RAG MVP with Citations")

# serve frontend & pdf
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

index_obj: Optional[Index] = None

class SearchResponse(BaseModel):
    results: list

class AnswerResponse(BaseModel):
    answer: str
    hits: list

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
