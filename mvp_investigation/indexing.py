# indexing.py
import json, numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class Index:
    def __init__(self, emb_path: str, meta_path: str, pdf_name: str = "doc.pdf"):
        data = np.load(emb_path, allow_pickle=True)

        # Normalize ids to plain strings
        raw_ids = data["ids"]
        if isinstance(raw_ids, np.ndarray):
            self.ids = [x if isinstance(x, str) else x.item() for x in raw_ids]
        else:
            self.ids = [str(raw_ids)]

        self.vecs = data["vecs"].astype("float32")
        self.id2row = {self.ids[i]: i for i in range(len(self.ids))}

        # Load meta
        self.meta: Dict[str, Dict[str, Any]] = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                m = json.loads(line)
                self.meta[m["item_id"]] = m

        # ✅ Always initialize the encoder here
        self.model: SentenceTransformer | None = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.pdf_name = pdf_name

    def _ensure_model(self):
        if self.model is None:
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, top_k=10) -> List[Dict[str, Any]]:
        # Defensive: ensure encoder exists even if __init__ path ever changes
        self._ensure_model()

        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0].astype("float32")

        sims = self.vecs @ q
        if len(sims) == 0:
            return []

        k = min(top_k * 3, len(sims))
        idx = np.argpartition(-sims, range(k))[:k]
        idx = idx[np.argsort(-sims[idx])][:top_k]

        results = []
        for i in idx:
            item_id = self.ids[i]
            m = self.meta.get(item_id, {})
            page = int(m.get("page", 1))
            citation_url = f"/static/{self.pdf_name}#page={page}"
            results.append({
                "item_id": item_id,
                "score": float(sims[i]),
                "page": page,
                "bbox_px": m.get("bbox_px"),
                "kind": m.get("kind"),
                "label": m.get("label"),
                "row_index": m.get("row_index"),
                "citation_url": citation_url,
            })
        return results

    def compose_extractive_answer(self, query: str, top_k=6) -> Dict[str, Any]:
        hits = self.search(query, top_k=top_k)
        lines = []
        for h in hits:
            cite = f"[p.{h['page']}]({h['citation_url']})"
            label = h.get("label") or h.get("kind")
            lines.append(f"- Evidence {label or ''} {cite}")
        ans = "Here are the most relevant pieces of evidence (extractive MVP):\n" + "\n".join(lines)
        return {"answer": ans, "hits": hits}
