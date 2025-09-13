#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: pip install optuna
try:
    import optuna
    HAVE_OPTUNA = True
except Exception:
    HAVE_OPTUNA = False

# ---------------- Args ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Auto-tune OCR parameters to minimize CER/WER against gold.")
    p.add_argument("--gold_dir", type=str, default="./gold",
                   help="Directory containing *.pdf and matching *.gold.json (default: ./gold)")
    p.add_argument("--out", type=str, default="runs/exp1", help="Output directory for caches and reports")
    p.add_argument("--n_trials", type=int, default=20, help="Optuna trials (ignored for --grid)")
    p.add_argument("--n_jobs", type=int, default=4, help="Parallel docs per trial")
    p.add_argument("--grid", action="store_true", help="Use a small grid search instead of Optuna")
    p.add_argument("--train_frac", type=float, default=0.8, help="Train/val split fraction")
    p.add_argument("--lang", type=str, default="eng", help="Tesseract language(s)")
    p.add_argument("--lower", action="store_true", help="Lowercase before scoring")
    p.add_argument("--strip_punct", action="store_true", help="Strip punctuation before WER scoring")
    return p.parse_args()

# ---------------- Dataset ----------------
def discover_pairs(gold_dir: Path) -> List[Tuple[Path, Path]]:
    pdfs = sorted(gold_dir.glob("*.pdf"))
    pairs = []
    for pdf in pdfs:
        gj = pdf.with_suffix("").with_suffix(".gold.json")  # handle .pdf -> .gold.json
        if not gj.exists():
            gj = gold_dir / (pdf.stem + ".gold.json")
        if gj.exists():
            pairs.append((pdf, gj))
    return pairs

def split_train_val(pairs: List[Tuple[Path,Path]], frac: float) -> Tuple[List, List]:
    n_train = max(1, int(len(pairs) * frac))
    return pairs[:n_train], pairs[n_train:]

# ---------------- Hash key ----------------
def param_key(params: Dict) -> str:
    s = json.dumps(params, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

# ---------------- Calls to your scripts ----------------
def run_ocr(pdf: Path, ocr_out_dir: Path, dpi: int, psm: int, max_spaces: int, lang: str) -> Path:
    ocr_out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = ocr_out_dir / f"{pdf.stem}.txt"
    if txt_path.exists():
        return txt_path
    cmd = [
        sys.executable, "ocr_pdf_to_text.py",
        "--input_path", str(pdf),
        "--out", str(ocr_out_dir),
        "--dpi", str(dpi),
        "--psm", str(psm),
        "--max-spaces", str(max_spaces),
        "--lang", lang,
        "--preserve-page-breaks",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return txt_path

def eval_pair(gold_json: Path, ocr_txt: Path, eval_out_dir: Path, lower: bool, strip_punct: bool) -> Dict:
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "eval_gold_ocr_text.py",
        "--gold_json", str(gold_json),
        "--ocr_txt", str(ocr_txt),
        "--out", str(eval_out_dir),
        "--assume_page_markers",
    ]
    if lower: cmd.append("--lower")
    if strip_punct: cmd.append("--strip_punct")

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    summary = json.loads((eval_out_dir/"summary.json").read_text(encoding="utf-8"))
    return summary  # contains {"doc":{"cer":..,"wer":..}, ...}

# ---------------- Trial runner ----------------
def score_trial(pairs: List[Tuple[Path,Path]], cache_root: Path,
                dpi: int, psm: int, max_spaces: int, lang: str,
                n_jobs: int, lower: bool, strip_punct: bool) -> Tuple[float, List[Dict]]:
    """
    Returns (loss, per_doc_rows)
    loss = weighted mean of (0.6*CER + 0.4*WER) over docs
    """
    key = param_key({"dpi": dpi, "psm": psm, "max_spaces": max_spaces, "lang": lang})
    ocr_dir = cache_root / f"ocr_{key}"; ocr_dir.mkdir(parents=True, exist_ok=True)
    eval_dir = cache_root / f"eval_{key}"; eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    def process_one(pair):
        pdf, gold = pair
        try:
            txt = run_ocr(pdf, ocr_dir, dpi, psm, max_spaces, lang)
            ed = eval_dir / pdf.stem
            summary = eval_pair(gold, txt, ed, lower, strip_punct)
            cer = float(summary["doc"]["cer"])
            wer = float(summary["doc"]["wer"])
            loss = 0.6 * cer + 0.4 * wer
            return {"pdf": str(pdf), "cer": cer, "wer": wer, "loss": loss}
        except subprocess.CalledProcessError as e:
            return {"pdf": str(pdf), "cer": 1.0, "wer": 1.0, "loss": 1.0, "error": e.stderr.decode("utf-8", "ignore")}
        except Exception as e:
            return {"pdf": str(pdf), "cer": 1.0, "wer": 1.0, "loss": 1.0, "error": repr(e)}

    # parallel over docs for this trial
    with ThreadPoolExecutor(max_workers=max(1, n_jobs)) as ex:
        futs = [ex.submit(process_one, p) for p in pairs]
        for f in as_completed(futs):
            rows.append(f.result())

    # mean loss (exclude hard failures if you want)
    good = [r for r in rows if "error" not in r]
    if not good:
        mean_loss = 1.0
    else:
        mean_loss = sum(r["loss"] for r in good) / len(good)

    # persist per-doc results for this trial
    (eval_dir / "per_doc.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    return mean_loss, rows

# ---------------- Main ----------------
def main():
    args = parse_args()
    gold_dir = Path(args.gold_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = discover_pairs(gold_dir)
    if not pairs:
        raise SystemExit(f"No (pdf, gold.json) pairs found under {gold_dir}")

    train, val = split_train_val(pairs, args.train_frac)
    print(f"[data] train={len(train)}  val={len(val)}")

    cache_root = out_root / "cache"
    reports_root = out_root / "reports"
    cache_root.mkdir(exist_ok=True); reports_root.mkdir(exist_ok=True)

    def evaluate_params(tag: str, dpi: int, psm: int, max_spaces: int) -> Dict:
        train_loss, train_rows = score_trial(
            train, cache_root, dpi=dpi, psm=psm, max_spaces=max_spaces, lang=args.lang,
            n_jobs=args.n_jobs, lower=args.lower, strip_punct=args.strip_punct
        )
        val_loss, val_rows = score_trial(
            val, cache_root, dpi=dpi, psm=psm, max_spaces=max_spaces, lang=args.lang,
            n_jobs=args.n_jobs, lower=args.lower, strip_punct=args.strip_punct
        ) if val else (train_loss, train_rows)

        report = {
            "tag": tag,
            "params": {"dpi": dpi, "psm": psm, "max_spaces": max_spaces, "lang": args.lang},
            "train": {"loss": train_loss, "rows": train_rows},
            "val": {"loss": val_loss, "rows": val_rows},
        }
        (reports_root / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    best_report = None

    if args.grid or not HAVE_OPTUNA:
        print("[mode] grid search")
        grid_dpi = [275, 300, 325, 350, 375, 400]
        grid_psm = [3, 4, 6]
        grid_spaces = [8, 10, 12, 14, 16]
        k = 0
        for dpi in grid_dpi:
            for psm in grid_psm:
                for ms in grid_spaces:
                    k += 1
                    tag = f"grid_{k:03d}_dpi{dpi}_psm{psm}_ms{ms}"
                    rep = evaluate_params(tag, dpi, psm, ms)
                    if best_report is None or rep["val"]["loss"] < best_report["val"]["loss"]:
                        best_report = rep
    else:
        print("[mode] optuna")
        study = optuna.create_study(direction="minimize")
        def objective(trial):
            dpi = trial.suggest_int("dpi", 250, 450, step=25)
            psm = trial.suggest_categorical("psm", [3, 4, 6])
            ms = trial.suggest_int("max_spaces", 6, 20)
            tag = f"trial_{trial.number:03d}_dpi{dpi}_psm{psm}_ms{ms}"
            rep = evaluate_params(tag, dpi, psm, ms)
            # keep track of best
            nonlocal best_report
            if best_report is None or rep["val"]["loss"] < best_report["val"]["loss"]:
                best_report = rep
            return rep["val"]["loss"]
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        # Save study
        (out_root / "study_best.json").write_text(json.dumps(best_report, indent=2), encoding="utf-8")

    # Final summary
    if best_report:
        (out_root / "best_params.json").write_text(json.dumps(best_report["params"], indent=2), encoding="utf-8")
        print("[best] params:", best_report["params"])
        print(f"[best] train loss={best_report['train']['loss']:.4f}  val loss={best_report['val']['loss']:.4f}")
        # also write CSV for quick glance
        import pandas as pd
        def rows_to_df(rows):
            import pandas as pd
            return pd.DataFrame(rows)
        df_train = rows_to_df(best_report["train"]["rows"]); df_val = rows_to_df(best_report["val"]["rows"])
        df_train.to_csv(out_root/"best_train_per_doc.csv", index=False)
        df_val.to_csv(out_root/"best_val_per_doc.csv", index=False)

if __name__ == "__main__":
    main()
