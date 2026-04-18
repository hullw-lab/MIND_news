"""
run_final_analysis.py
---------------------
Runs the post-training analysis that didn't complete in the training notebook:
  1. Final evaluation of the best checkpoint -> final_metrics.json
  2. Cold-start error analysis by history bucket -> error_analysis_by_history.csv
  3. Popularity baseline -> popularity_baseline.json
  4. Final summary table -> final_summary.csv

Run from the project root:
    python run_final_analysis.py

Expects:
  - cache/preprocessed.pkl        (from notebook 02)
  - cache/embedding_matrix.npy    (from notebook 02)
  - cache/eval_samples.pkl        (from notebook 02)
  - results/hyperparameter_comparison.csv  (from the training runs that completed)
  - models/nrms_*_best.pt         (from the training runs that completed)
"""

import json
import os
import pickle
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import MINDEvalDataset, eval_collate
from evaluate import evaluate, mrr_score, ndcg_score
from model import NRMSModel


RESULTS_DIR = "results"
MODELS_DIR = "models"
CACHE_DIR = "cache"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # -------------------------------------------------------------------------
    # 1. Load everything we need
    # -------------------------------------------------------------------------
    print("[load] preprocessed bundle...")
    with open(os.path.join(CACHE_DIR, "preprocessed.pkl"), "rb") as f:
        bundle = pickle.load(f)
    with open(os.path.join(CACHE_DIR, "eval_samples.pkl"), "rb") as f:
        eval_samples = pickle.load(f)
    embedding_matrix = np.load(os.path.join(CACHE_DIR, "embedding_matrix.npy"))
    print(f"[load] eval samples: {len(eval_samples):,}")

    # -------------------------------------------------------------------------
    # 2. Find best run from the hyperparameter comparison
    # -------------------------------------------------------------------------
    comp_path = os.path.join(RESULTS_DIR, "hyperparameter_comparison.csv")
    comp = pd.read_csv(comp_path)
    best_row = comp.sort_values("AUC", ascending=False).iloc[0]
    best_run_label = best_row["Run"]
    best_auc_from_training = best_row["AUC"]

    label_to_key = {
        "Baseline (lr=1e-4, K=4, drop=0.2)": "nrms_baseline",
        "A: lr=3e-4":     "nrms_exp_a_lr3e4",
        "B: neg_k=8":     "nrms_exp_b_negk8",
        "C: dropout=0.3": "nrms_exp_c_drop03",
    }
    best_run_key = label_to_key[best_run_label]
    best_ckpt = os.path.join(MODELS_DIR, f"{best_run_key}_best.pt")
    print(f"[best run] {best_run_label}  (training AUC {best_auc_from_training:.4f})")
    print(f"[best ckpt] {best_ckpt}")

    # -------------------------------------------------------------------------
    # 3. Build eval loader
    # -------------------------------------------------------------------------
    eval_ds = MINDEvalDataset(
        eval_samples, bundle["news_encoded"],
        max_history=50, max_title_len=30,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=32, shuffle=False, num_workers=0,
        collate_fn=eval_collate,
        pin_memory=(device.type == "cuda"),
    )

    # -------------------------------------------------------------------------
    # 4. Build model and load weights
    # -------------------------------------------------------------------------
    print("[model] building NRMS and loading best checkpoint...")
    model = NRMSModel(embedding_matrix, num_heads=16, head_dim=16, dropout=0.2).to(device)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()

    # -------------------------------------------------------------------------
    # 5. Final evaluation -> final_metrics.json
    # -------------------------------------------------------------------------
    print("\n[eval] final evaluation of best model on full dev set...")
    final_metrics = evaluate(model, eval_loader, device, verbose=True)
    final_path = os.path.join(RESULTS_DIR, "final_metrics.json")
    with open(final_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[write] {final_path}")

    # -------------------------------------------------------------------------
    # 6. Error analysis by history length bucket
    # -------------------------------------------------------------------------
    print("\n[analysis] history-length bucket analysis...")

    buckets = {
        "0 (cold)":   (0, 0),
        "1-4":        (1, 4),
        "5-19":       (5, 19),
        "20+":        (20, 10_000),
    }
    results_by_bucket = {}

    with torch.no_grad():
        for batch in eval_loader:
            hist = batch["history"].to(device)
            mask = batch["hist_mask"].to(device)
            cands = [c.to(device) for c in batch["candidates"]]
            labels_list = batch["labels"]
            scores_list = model.score_variable_candidates(hist, cands, mask)

            hist_lens = mask.sum(dim=1).cpu().numpy()

            for hl, labels, scores in zip(hist_lens, labels_list, scores_list):
                if labels.sum() == 0 or labels.sum() == len(labels):
                    continue
                for name, (lo, hi) in buckets.items():
                    if lo <= hl <= hi:
                        results_by_bucket.setdefault(name, {"auc": [], "mrr": [], "n10": []})
                        results_by_bucket[name]["auc"].append(roc_auc_score(labels, scores))
                        results_by_bucket[name]["mrr"].append(mrr_score(labels, scores))
                        results_by_bucket[name]["n10"].append(ndcg_score(labels, scores, 10))
                        break

    rows = []
    for name in ["0 (cold)", "1-4", "5-19", "20+"]:
        d = results_by_bucket.get(name)
        if not d:
            continue
        rows.append({
            "History bucket": name,
            "N impressions":  len(d["auc"]),
            "AUC":     float(np.mean(d["auc"])),
            "MRR":     float(np.mean(d["mrr"])),
            "nDCG@10": float(np.mean(d["n10"])),
        })

    err_analysis = pd.DataFrame(rows).round(4)
    err_path = os.path.join(RESULTS_DIR, "error_analysis_by_history.csv")
    err_analysis.to_csv(err_path, index=False)
    print(err_analysis.to_string(index=False))
    print(f"[write] {err_path}")

    # -------------------------------------------------------------------------
    # 7. Popularity baseline
    # -------------------------------------------------------------------------
    print("\n[baseline] popularity baseline...")
    pop = Counter()
    for imps in bundle["train_beh"]["impressions"].dropna():
        for tok in imps.split():
            nid, label = tok.rsplit("-", 1)
            if label == "1":
                pop[nid] += 1

    aucs, mrrs, n10s = [], [], []
    for s in eval_samples:
        labels = np.asarray(s["labels"])
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        scores = np.asarray([pop.get(nid, 0) for nid in s["candidates"]], dtype=float)
        aucs.append(roc_auc_score(labels, scores))
        mrrs.append(mrr_score(labels, scores))
        n10s.append(ndcg_score(labels, scores, 10))

    pop_metrics = {
        "AUC":     float(np.mean(aucs)),
        "MRR":     float(np.mean(mrrs)),
        "nDCG@10": float(np.mean(n10s)),
    }
    print("Popularity baseline:")
    for k, v in pop_metrics.items():
        print(f"  {k:8s} = {v:.4f}")
    pop_path = os.path.join(RESULTS_DIR, "popularity_baseline.json")
    with open(pop_path, "w") as f:
        json.dump(pop_metrics, f, indent=2)
    print(f"[write] {pop_path}")

    # -------------------------------------------------------------------------
    # 8. Final summary
    # -------------------------------------------------------------------------
    summary = pd.DataFrame([
        {"Model": "Random",            "AUC": 0.5,                 "MRR": 0.2,                 "nDCG@10": 0.3},
        {"Model": "Popularity",        "AUC": pop_metrics["AUC"],  "MRR": pop_metrics["MRR"],  "nDCG@10": pop_metrics["nDCG@10"]},
        {"Model": f"NRMS ({best_run_label})", "AUC": final_metrics["AUC"], "MRR": final_metrics["MRR"], "nDCG@10": final_metrics["nDCG@10"]},
    ]).round(4)

    summary_path = os.path.join(RESULTS_DIR, "final_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("\nFinal summary:")
    print(summary.to_string(index=False))
    print(f"[write] {summary_path}")

    print("\n[done] all final analysis artifacts written.")


if __name__ == "__main__":
    main()