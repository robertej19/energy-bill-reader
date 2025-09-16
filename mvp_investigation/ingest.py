# ingest.py
#!/usr/bin/env python3
import json, os, re, math
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from extract_pdf import extract_text_and_tables_jsonl  # we'll define this shim below

CHUNK_TOKENS = 50
OVERLAP_TOKENS = 20

def basic_tokenize(s: str) -> List[str]:
    return re.findall(r"\w+|\S", s)

def sliding_chunks(text: str, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS):
    toks = basic_tokenize(text)
    if not toks:
        return []
    out = []
    i = 0
    while i < len(toks):
        j = min(len(toks), i + chunk_tokens)
        out.append("".join(toks[i:j]).replace(" ##", ""))  # keep simple join
        if j == len(toks):
            break
        i = j - overlap
    return out

def flatten_table_rows(table: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (row_as_text, row_meta)."""
    rows = table.get("rows", [])
    headers = table.get("headers", [])
    page = table["page"]
    bbox = table["bbox_px"]
    out = []
    for r_idx, row in enumerate(rows):
        kv_pairs = []
        for j, cell in enumerate(row):
            key = headers[j] if j < len(headers) and headers[j] else f"col{j}"
            val = cell.strip()
            if val:
                kv_pairs.append(f"{key}={val}")
        row_text = "; ".join(kv_pairs) if kv_pairs else " ".join([c.strip() for c in row if c.strip()])
        out.append((row_text, {"page": page, "bbox_px": bbox, "row_index": r_idx}))
    return out

def build_embeddings(items: List[Dict[str, Any]], model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [it["text"] for it in items]
    vecs = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=64)
    ids  = np.array([it["item_id"] for it in items], dtype=object)
    return ids, vecs

def ingest_single_pdf(pdf_path: str, output_dir: str = "output") -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    stream_path = Path(output_dir, "stream.jsonl")
    meta_path   = Path(output_dir, "meta.jsonl")
    emb_path    = Path(output_dir, "embeddings.npz")

    # 1) Run extraction (JSONL stream with page/bbox/kind/text)
    items = extract_text_and_tables_jsonl(pdf_path, stream_path)

    # 2) Build retrieval units (chunks + table rows)
    retr_items: List[Dict[str, Any]] = []
    meta_records: List[Dict[str, Any]] = []

    def add_unit(doc_id, kind, page, bbox, text, extra=None):
        item_id = f"{doc_id}|{kind}|p{page}|{len(retr_items)}"
        retr_items.append({"item_id": item_id, "doc_id": doc_id, "kind": kind, "page": page, "bbox_px": bbox, "text": text})
        meta = {"item_id": item_id, "doc_id": doc_id, "kind": kind, "page": page, "bbox_px": bbox}
        if extra: meta.update(extra)
        meta_records.append(meta)

    doc_id = Path(pdf_path).stem
    for rec in items:
        if rec["kind"] == "para":
            text = rec["text"].strip()
            if not text: continue
            chunks = sliding_chunks(text)
            for ch in chunks:
                add_unit(doc_id, "chunk", rec["page"], rec["bbox_px"], ch)
        elif rec["kind"] == "table":
            # make table-level unit
            cap = rec.get("caption", rec["label"])
            table_text = (cap or "table").strip()
            if table_text:
                add_unit(doc_id, "table", rec["page"], rec["bbox_px"], table_text, {"label": rec.get("label")})
            # row-level units
            for row_text, row_meta in flatten_table_rows(rec):
                if row_text.strip():
                    add_unit(doc_id, "row", row_meta["page"], rec["bbox_px"], row_text, {"row_index": row_meta["row_index"], "label": rec.get("label")})

    # 3) Write meta
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta_records:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 4) Embeddings
    ids, vecs = build_embeddings(retr_items)
    np.savez_compressed(emb_path, ids=ids, vecs=vecs)

    return {"stream_path": str(stream_path), "meta_path": str(meta_path), "emb_path": str(emb_path), "count": len(retr_items)}

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="static/doc.pdf")
    ap.add_argument("--out", default="output")
    args = ap.parse_args()
    info = ingest_single_pdf(args.pdf, args.out)
    print("[✓] Ingest complete:", info)
