#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, unicodedata
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from difflib import SequenceMatcher

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate OCR text vs gold JSON (from ReportLab renderer).")
    p.add_argument("--gold_json", type=str, required=True, help="Path to *.gold.json from generate_gold_pdfs.py")
    p.add_argument("--ocr_txt", type=str, required=True, help="Path to OCR text produced by ocr_pdf_to_text.py")
    p.add_argument("--out", type=str, default="out_eval", help="Output directory")
    p.add_argument("--lower", action="store_true", help="Lowercase for scoring")
    p.add_argument("--strip_punct", action="store_true", help="Strip punctuation for WER scoring")
    p.add_argument("--assume_page_markers", action="store_true",
                   help="Split OCR text on '--- Page N ---' markers for page-level metrics.")
    return p.parse_args()

# ---------- normalization ----------
LIGATURES = {"\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl"}
def normalize_text(s: str, lower=False) -> str:
    if not s: return ""
    s = s.replace("\u00ad", "")
    for k,v in LIGATURES.items(): s = s.replace(k, v)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()
    return s.lower() if lower else s

def strip_punct_words(words: List[str]) -> List[str]:
    return [re.sub(r"[^\w\-']", "", w) for w in words if re.sub(r"[^\w\-']", "", w) != ""]

# ---------- edit distances ----------
def levenshtein_char(a: str, b: str) -> int:
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    if len(a) > len(b): a, b = b, a
    prev = list(range(len(a) + 1))
    for bj in b:
        curr = [prev[0] + 1]
        for i, ai in enumerate(a, start=1):
            ins = curr[i-1] + 1
            dele = prev[i] + 1
            sub = prev[i-1] + (ai != bj)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def levenshtein_words(a_words: List[str], b_words: List[str]) -> int:
    na, nb = len(a_words), len(b_words)
    if na == 0: return nb
    if nb == 0: return na
    prev = list(range(na + 1))
    for j in range(1, nb + 1):
        bj = b_words[j-1]
        curr = [j]
        for i in range(1, na + 1):
            cost = 0 if a_words[i-1] == bj else 1
            curr.append(min(curr[i-1] + 1, prev[i] + 1, prev[i-1] + cost))
        prev = curr
    return prev[-1]

# ---------- gold loading & linearization ----------
def load_gold(gold_path: Path) -> dict:
    return json.loads(gold_path.read_text(encoding="utf-8"))

def linearize_gold(gold: dict) -> Tuple[str, List[dict], List[str]]:
    """
    Returns:
      ref_text (str) - full linearized gold text (headings, paragraphs, tables (as TSV lines); figures ignored unless caption stored)
      elements (list) - per-element metadata with char spans in ref_text
      page_texts (list[str]) - concatenated text per page (same linearization)
    """
    all_text_parts: List[str] = []
    page_texts: List[str] = []
    elements: List[dict] = []

    for p_idx, page in enumerate(gold.get("pages", [])):
        page_parts: List[str] = []
        offset = sum(len(x) for x in all_text_parts)
        for e_idx, el in enumerate(page.get("elements", [])):
            et = el.get("type")
            if et == "heading":
                t = str(el.get("text","")).strip()
                if not t: continue
                start = offset + sum(len(x) for x in page_parts)
                page_parts.append(t + "\n")
                end = offset + sum(len(x) for x in page_parts)
                elements.append({"page":p_idx+1,"idx":e_idx,"type":"heading","text":t,"start":start,"end":end})
            elif et == "paragraph":
                t = str(el.get("text","")).strip()
                if not t: continue
                start = offset + sum(len(x) for x in page_parts)
                page_parts.append(t + "\n\n")
                end = offset + sum(len(x) for x in page_parts)
                elements.append({"page":p_idx+1,"idx":e_idx,"type":"paragraph","text":t,"start":start,"end":end})
            elif et == "table":
                rows = el.get("rows_data") or el.get("rows")
                if isinstance(rows, list) and rows:
                    t = "\n".join(["\t".join(map(str, r)) for r in rows])
                    start = offset + sum(len(x) for x in page_parts)
                    page_parts.append(t + "\n\n")
                    end = offset + sum(len(x) for x in page_parts)
                    elements.append({"page":p_idx+1,"idx":e_idx,"type":"table","text":t,"start":start,"end":end})
            elif et == "figure":
                cap = str(el.get("caption","")).strip()
                if cap:
                    start = offset + sum(len(x) for x in page_parts)
                    page_parts.append(cap + "\n\n")
                    end = offset + sum(len(x) for x in page_parts)
                    elements.append({"page":p_idx+1,"idx":e_idx,"type":"figure_caption","text":cap,"start":start,"end":end})
        page_text = "".join(page_parts)
        page_texts.append(page_text)
        all_text_parts.append(page_text)

    ref_text = "".join(all_text_parts).rstrip() + ("\n" if all_text_parts else "")
    return ref_text, elements, page_texts

# ---------- alignment ----------
def build_ref_to_hyp_map(ref: str, hyp: str):
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    return [(i1,i2,j1,j2,tag) for tag,i1,i2,j1,j2 in sm.get_opcodes()]

def project_span(ops, ref_start: int, ref_end: int, hyp_len: int) -> Tuple[int,int]:
    def map_pos(ref_pos: int) -> int:
        j_at = 0
        for i1,i2,j1,j2,tag in ops:
            if ref_pos < i1:
                return j1
            if i1 <= ref_pos < i2:
                if tag in ("equal","replace"):
                    rel = 0 if (i2-i1)==0 else (ref_pos - i1)/(i2 - i1)
                    return int(j1 + rel*(j2 - j1))
                elif tag == "delete":
                    return j1
                elif tag == "insert":
                    return j2
            j_at = j2
        return hyp_len
    a = max(0, min(hyp_len, map_pos(ref_start)))
    b = max(0, min(hyp_len, map_pos(ref_end)))
    if b < a: a, b = b, a
    pad = max(0, (b-a)//10)
    return max(0, a - pad), min(hyp_len, b + pad)

# ---------- scoring ----------
def score_pair(ref: str, hyp: str, lower=False, strip_punct=False) -> Dict[str, float]:
    ref_n = normalize_text(ref, lower=lower)
    hyp_n = normalize_text(hyp, lower=lower)
    cer_den = max(1, len(ref_n))
    cer = levenshtein_char(ref_n, hyp_n) / cer_den

    ref_w = ref_n.split()
    hyp_w = hyp_n.split()
    if strip_punct:
        ref_w = strip_punct_words(ref_w)
        hyp_w = strip_punct_words(hyp_w)
    wer_den = max(1, len(ref_w))
    wer = levenshtein_words(ref_w, hyp_w) / wer_den
    return {"cer": cer, "wer": wer, "ref_chars": cer_den, "ref_words": wer_den}

# ---------- main ----------
def main():
    args = parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    gold = json.loads(Path(args.gold_json).read_text(encoding="utf-8"))
    ref_text, elements, ref_pages = linearize_gold(gold)
    hyp_text = Path(args.ocr_txt).read_text(encoding="utf-8", errors="ignore")

    # Doc-level
    doc_scores = score_pair(ref_text, hyp_text, lower=args.lower, strip_punct=args.strip_punct)

    # Page-level
    if args.assume_page_markers:
        parts = re.split(r"\n?-{3,}\s*Page\s+\d+\s*-{3,}\n?", hyp_text)
        hyp_pages = [p.strip() for p in parts if p.strip()]
    else:
        # proportional split based on ref page lengths
        ref_lens = [max(1,len(normalize_text(t, lower=args.lower))) for t in ref_pages]
        total = sum(ref_lens) or 1
        cuts = []; cursor = 0
        for L in ref_lens[:-1]:
            cursor += int(round(L/total * len(hyp_text)))
            cuts.append(cursor)
        hyp_pages = []; prev = 0
        for c in cuts + [len(hyp_text)]:
            hyp_pages.append(hyp_text[prev:c]); prev = c

    per_page_rows = []
    for i, ref_p in enumerate(ref_pages):
        hyp_p = hyp_pages[i] if i < len(hyp_pages) else ""
        sc = score_pair(ref_p, hyp_p, lower=args.lower, strip_punct=args.strip_punct)
        per_page_rows.append({"page": i+1, **sc})
    df_pages = pd.DataFrame(per_page_rows)
    if not df_pages.empty:
        df_pages.to_csv(out_dir/"per_page.csv", index=False)

    # Element-level via alignment
    ops = build_ref_to_hyp_map(normalize_text(ref_text, lower=args.lower),
                               normalize_text(hyp_text, lower=args.lower))
    elem_rows = []
    for el in elements:
        t = el.get("text","").strip()
        if not t: continue
        a, b = project_span(ops, el["start"], el["end"], len(hyp_text))
        hyp_slice = hyp_text[a:b]
        sc = score_pair(t, hyp_slice, lower=args.lower, strip_punct=args.strip_punct)
        elem_rows.append({
            "page": el["page"], "idx": el["idx"], "type": el["type"],
            "ref_len_chars": sc["ref_chars"], "ref_len_words": sc["ref_words"],
            "cer": sc["cer"], "wer": sc["wer"],
            "hyp_start": a, "hyp_end": b
        })
    df_elems = pd.DataFrame(elem_rows)
    if not df_elems.empty:
        df_elems.to_csv(out_dir/"per_element.csv", index=False)

    summary = {"doc": doc_scores}
    (out_dir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Console summary for quick visibility
    print(f"[Doc] CER={doc_scores['cer']:.2%}  WER={doc_scores['wer']:.2%}")
    if not df_pages.empty:
        print(f"[Per-page mean] CER={df_pages['cer'].mean():.2%}  WER={df_pages['wer'].mean():.2%}")

if __name__ == "__main__":
    main()
