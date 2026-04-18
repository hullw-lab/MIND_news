"""
evaluate.py
-----------
Ranking metrics for the MIND benchmark + per-impression eval loop.
"""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader


def dcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    order = np.argsort(y_score)[::-1][:k]
    gains = np.asarray(y_true)[order]
    discounts = np.log2(np.arange(len(gains)) + 2)
    return float(np.sum(gains / discounts))


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int = 10) -> float:
    best = dcg_score(y_true, y_true, k)
    if best == 0:
        return 0.0
    return dcg_score(y_true, y_score, k) / best


def mrr_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)[::-1]
    y_sorted = np.asarray(y_true)[order]
    for i, v in enumerate(y_sorted):
        if v == 1:
            return 1.0 / (i + 1)
    return 0.0


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device,
             verbose: bool = True) -> Dict[str, float]:
    model.eval()
    aucs, mrrs, n5s, n10s = [], [], [], []

    for batch in loader:
        history = batch["history"].to(device)
        hist_mask = batch["hist_mask"].to(device)
        cand_list = [c.to(device) for c in batch["candidates"]]
        label_list = batch["labels"]

        scores_list = model.score_variable_candidates(history, cand_list, hist_mask)

        for scores, labels in zip(scores_list, label_list):
            if labels.sum() == 0 or labels.sum() == len(labels):
                continue
            aucs.append(roc_auc_score(labels, scores))
            mrrs.append(mrr_score(labels, scores))
            n5s.append(ndcg_score(labels, scores, 5))
            n10s.append(ndcg_score(labels, scores, 10))

    metrics = {
        "AUC":    float(np.mean(aucs)),
        "MRR":    float(np.mean(mrrs)),
        "nDCG@5": float(np.mean(n5s)),
        "nDCG@10": float(np.mean(n10s)),
        "n_impressions": len(aucs),
    }
    if verbose:
        print(f"  AUC     = {metrics['AUC']:.4f}")
        print(f"  MRR     = {metrics['MRR']:.4f}")
        print(f"  nDCG@5  = {metrics['nDCG@5']:.4f}")
        print(f"  nDCG@10 = {metrics['nDCG@10']:.4f}")
        print(f"  ({metrics['n_impressions']:,} impressions scored)")
    return metrics
