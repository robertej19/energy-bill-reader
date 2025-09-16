# app.py
import os
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, List, Dict, Any

import fitz  # PyMuPDF
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from llama_cpp import Llama

# ------------------------
# Paths & constants
# ------------------------
APP_DIR = Path(__file__).parent.resolve()
STATIC_DIR = APP_DIR / "static"
OUT_DIR = APP_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_NAME = "doc.pdf"  # put your single PDF at static/doc.pdf
PDF_PATH = STATIC_DIR / PDF_NAME

# rasterization for /page_image and must match extraction DPI
PDF_DPI = 300

# model path for local RAG
MODEL_PATH = "/home/rober/UNIVERSAL_UTILITIES/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"

# written by ingest.py (our extractor shim)
STREAM_PATH = OUT_DIR / "stream.jsonl"

# ------------------------
# Local modules
# ------------------------
from ingest import ingest_single_pdf  # noqa: E402
from indexing import Index            # noqa: E402

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="PDF RAG MVP with Citations & Highlights")

# serve /static (PDF, CSS, HTML assets etc.)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global in-memory index object
index_obj: Optional[Index] = None


# ------------------------
# Pydantic response models
# ------------------------
class SearchResponse(BaseModel):
    results: list


class AnswerResponse(BaseModel):
    answer: str
    hits: list


# ------------------------
# HTML index page
# ------------------------
@app.get("/", response_class=HTMLResponse)
def index_page():
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        return HTMLResponse("<h1>Missing static/index.html</h1>", status_code=500)
    return HTMLResponse(index_html.read_text(encoding="utf-8"))


# ------------------------
# Page image rendering (PNG @ 300 DPI)
# ------------------------
PAGES_DIR = OUT_DIR / "pages"
PAGES_DIR.mkdir(parents=True, exist_ok=True)


def _page_png_path(page: int) -> Path:
    return PAGES_DIR / f"page_{page:04d}_{PDF_DPI}dpi.png"


@lru_cache(maxsize=512)
def render_page_to_png(page: int, dpi: int = PDF_DPI) -> Path:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing PDF at {PDF_PATH}")
    out = _page_png_path(page)
    if out.exists():
        return out
    doc = fitz.open(str(PDF_PATH))
    try:
        if page < 1 or page > doc.page_count:
            raise ValueError(f"Page out of range: {page} (1..{doc.page_count})")
        p = doc.load_page(page - 1)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = p.get_pixmap(matrix=mat, alpha=False)
        pix.save(str(out))
        return out
    finally:
        doc.close()


@app.get("/page_image")
def page_image(page: int = Query(1, ge=1)):
    try:
        png_path = render_page_to_png(page, PDF_DPI)
        return FileResponse(str(png_path), media_type="image/png")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ------------------------
# Ingest (extract + embed + index)
# ------------------------
@app.post("/ingest")
def do_ingest():
    if not PDF_PATH.exists():
        return JSONResponse({"error": f"Missing {PDF_PATH}"}, status_code=400)
    info = ingest_single_pdf(str(PDF_PATH), str(OUT_DIR))

    global index_obj
    emb = OUT_DIR / "embeddings.npz"
    meta = OUT_DIR / "meta.jsonl"
    if not (emb.exists() and meta.exists()):
        return JSONResponse({"error": "Embedding or meta files not found after ingest."}, status_code=500)

    index_obj = Index(str(emb), str(meta), pdf_name=PDF_NAME)
    return JSONResponse({"status": "ok", "count": info.get("count", 0)})


# ------------------------
# Debug: sample paragraphs from stream.jsonl
# ------------------------
@lru_cache(maxsize=1)
def _load_stream_paras() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if STREAM_PATH.exists():
        with open(STREAM_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("kind") == "para" and obj.get("text"):
                    items.append({
                        "page": obj.get("page"),
                        "bbox_px": obj.get("bbox_px"),
                        "text": obj.get("text"),
                    })
    return items


@app.get("/debug/paras")
def debug_paras(n: int = 6, offset: int = 0, sample: bool = True):
    items = _load_stream_paras()
    if not items:
        return JSONResponse({"error": "No paragraphs found. Run /ingest first."}, status_code=404)
    if sample:
        step = max(1, len(items) // max(1, n))
        picked = [items[i] for i in range(0, min(len(items), step * n), step)]
    else:
        picked = items[offset:offset + n]
    return {"paras": picked, "total": len(items)}


# ------------------------
# Search & Answer (RAG)
# ------------------------
def _ensure_index() -> Optional[Index]:
    global index_obj
    if index_obj is not None:
        return index_obj
    emb = OUT_DIR / "embeddings.npz"
    meta = OUT_DIR / "meta.jsonl"
    if emb.exists() and meta.exists():
        index_obj = Index(str(emb), str(meta), pdf_name=PDF_NAME)
        return index_obj
    return None


@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(...), top_k: int = 10):
    idx = _ensure_index()
    if idx is None:
        return JSONResponse({"error": "Index not built. POST /ingest first."}, status_code=400)
    hits = idx.search(query, top_k=top_k)  # switch to idx.search_hybrid if you added it
    return {"results": hits}


# ---- RAG with local Qwen2.5 GGUF via llama-cpp-python ----
class RagLLM:
    _llm: Optional[Llama] = None

    @classmethod
    def get_llm(cls) -> Llama:
        if cls._llm is None:
            if not Path(MODEL_PATH).exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            cls._llm = Llama(
                model_path=MODEL_PATH,
                n_ctx=8192,                 # match model's trained context (32k)
                n_threads=1,                #os.cpu_count() or 4,
                n_gpu_layers=0,              # set >0 if using CUDA wheel
                chat_format="qwen",          # <-- FIX: use Qwen chat handler
                verbose=False
            )
        return cls._llm

    @staticmethod
    def _build_context(hits: List[Dict[str, Any]], max_chars: int = 6000) -> List[Dict[str, Any]]:
        ctx_items: List[Dict[str, Any]] = []
        total = 0
        for i, h in enumerate(hits, 1):
            t = (h.get("text") or "").strip()
            if not t:
                continue
            if len(t) > 800:
                t = t[:780].rstrip() + "…"
            url = h.get("citation_url") or f"/static/{PDF_NAME}#page={h.get('page', 1)}"
            s = f"[{i}] p.{h.get('page',1)}: {t}\n"
            if total + len(s) > max_chars:
                break
            ctx_items.append({"id": i, "text": t, "page": h.get("page", 1), "bbox_px": h.get("bbox_px"), "url": url})
            total += len(s)
        return ctx_items

    @staticmethod
    def _prompt_for_qwen(question: str, ctx_items: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        # Qwen uses the 'qwen' chat_format in llama-cpp; standard roles are fine.
        context_lines = [f"[{it['id']}] (page {it['page']}): {it['text']}" for it in ctx_items]
        context_text = "\n".join(context_lines) if context_lines else "(no context)"

        system_prompt = (
            "You are a careful research assistant. Answer using ONLY the provided context.\n"
            "Cite sources inline with bracketed footnotes like [^1], [^2] that refer to the numbered context items.\n"
            "If the context is insufficient, say so explicitly."
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context items:\n{context_text}\n\n"
            "Instructions:\n"
            "- Use only the information in the context items.\n"
            "- Append the matching citation markers like [^3] when you make claims.\n"
            "- Keep the answer concise and factual."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

    @staticmethod
    def _link_citations(answer: str, ctx_items: List[Dict[str, Any]]) -> str:
        url_by_id = {str(it["id"]): it["url"] for it in ctx_items}
        def _repl(m):
            k = m.group(1); url = url_by_id.get(k)
            return f"[^{k}]({url})" if url else f"[^{k}]"
        out = re.sub(r"\[\^(\d+)\]", _repl, answer)
        out = re.sub(r"\[(\d+)\](?!\()", _repl, out)
        return out

    def generate(self, question: str, hits: List[Dict[str, Any]], max_new_tokens: int = 512, temperature: float = 0.2) -> Dict[str, Any]:
        llm = self.get_llm()
        ctx_items = self._build_context(hits)
        msgs = self._prompt_for_qwen(question, ctx_items)
        resp = llm.create_chat_completion(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_new_tokens,
            stop=[],
        )
        text = resp["choices"][0]["message"]["content"]
        linked = self._link_citations(text, ctx_items)
        return {"answer": linked, "context_items": ctx_items}


@app.get("/answer", response_model=AnswerResponse)
def answer(query: str = Query(...), mode: str = "rag", top_k: int = 8):
    idx = _ensure_index()
    if idx is None:
        return JSONResponse({"error": "Index not built. POST /ingest first."}, status_code=400)

    if mode == "matches":
        hits = idx.search(query, top_k=top_k)
        answer_text = "Top matches:\n" + "\n".join([f"- {h['kind']} (p.{h['page']}) {h['citation_url']}" for h in hits])
        return {"answer": answer_text, "hits": hits}

    # RAG mode: pull extra candidates for better context
    hits = idx.search(query, top_k=max(12, top_k))
    rag = RagLLM()
    gen = rag.generate(query, hits)
    return {"answer": gen["answer"], "hits": hits}
