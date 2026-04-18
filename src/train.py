"""
train.py
--------
Training loop for NRMS with cross-entropy over (1 positive + K negatives).

Can be run from a notebook (import train_model) OR as a script:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_loader import (
    MINDTrainDataset, MINDEvalDataset, eval_collate,
    NewsTokenizer, build_eval_samples, build_train_samples,
    encode_all_news, load_behaviors, load_glove_matrix, load_news,
)
from evaluate import evaluate
from model import NRMSModel


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_or_load_preprocessed(data_dir: str,
                               cache_dir: str,
                               max_title_len: int,
                               min_word_freq: int) -> dict:
    """Build tokenizer, encoded news, and splits once; cache to disk."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "preprocessed.pkl")

    if os.path.exists(cache_path):
        print(f"[cache] loading preprocessed bundle from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("[preprocess] reading TSVs...")
    train_news = load_news(os.path.join(data_dir, "MINDsmall_train", "news.tsv"))
    dev_news   = load_news(os.path.join(data_dir, "MINDsmall_dev",   "news.tsv"))
    train_beh  = load_behaviors(os.path.join(data_dir, "MINDsmall_train", "behaviors.tsv"))
    dev_beh    = load_behaviors(os.path.join(data_dir, "MINDsmall_dev",   "behaviors.tsv"))

    # Union news from both splits so dev candidates are always encodable.
    all_news = (
        pd.concat([train_news, dev_news], ignore_index=True)
          .drop_duplicates("news_id")
          .reset_index(drop=True)
    )

    print("[preprocess] building vocab on TRAIN titles only...")
    tokenizer = NewsTokenizer(max_title_len=max_title_len, min_word_freq=min_word_freq)
    tokenizer.build_vocab(train_news["title"].tolist())

    print("[preprocess] encoding all news titles...")
    news_encoded = encode_all_news(all_news, tokenizer)

    bundle = {
        "tokenizer": tokenizer,
        "news_encoded": news_encoded,
        "train_beh": train_beh,
        "dev_beh": dev_beh,
        "all_news": all_news,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[cache] saved bundle to {cache_path}")
    return bundle


def train_model(
    data_dir: str = "data",
    cache_dir: str = "cache",
    models_dir: str = "models",
    results_dir: str = "results",
    glove_path: str | None = None,
    batch_size: int = 64,
    eval_batch_size: int = 32,
    lr: float = 1e-4,
    epochs: int = 5,
    neg_k: int = 4,
    max_history: int = 50,
    max_title_len: int = 30,
    min_word_freq: int = 2,
    num_heads: int = 16,
    head_dim: int = 16,
    dropout: float = 0.2,
    grad_clip: float = 1.0,
    seed: int = 42,
    num_workers: int = 0,
    run_name: str = "nrms_base",
    eval_every_epoch: bool = True,
    limit_train_samples: int | None = None,
) -> Dict:
    set_seed(seed)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    bundle = build_or_load_preprocessed(data_dir, cache_dir, max_title_len, min_word_freq)
    tokenizer: NewsTokenizer = bundle["tokenizer"]
    news_encoded = bundle["news_encoded"]

    # GloVe embedding matrix
    if glove_path is None:
        glove_path = os.path.join(data_dir, "glove", "glove.6B.300d.txt")
    emb_cache = os.path.join(cache_dir, "embedding_matrix.npy")
    if os.path.exists(emb_cache):
        embedding_matrix = np.load(emb_cache)
        print(f"[cache] loaded embedding matrix {embedding_matrix.shape}")
    else:
        embedding_matrix = load_glove_matrix(glove_path, tokenizer.word2idx, 300)
        np.save(emb_cache, embedding_matrix)

    # Build samples
    print("[samples] building train samples...")
    train_samples = build_train_samples(
        bundle["train_beh"], news_encoded,
        neg_k=neg_k, max_history=max_history, seed=seed,
    )
    if limit_train_samples is not None:
        train_samples = train_samples[:limit_train_samples]
        print(f"[samples] limited to {len(train_samples):,}")

    print("[samples] building eval samples...")
    eval_samples = build_eval_samples(
        bundle["dev_beh"], news_encoded, max_history=max_history,
    )

    train_ds = MINDTrainDataset(train_samples, news_encoded, max_history, max_title_len)
    eval_ds  = MINDEvalDataset(eval_samples, news_encoded, max_history, max_title_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=eval_batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=eval_collate,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = NRMSModel(embedding_matrix, num_heads, head_dim, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] trainable params: {n_params:,}")

    # History
    history: Dict[str, List] = {"epoch": [], "train_loss": [], "val_metrics": []}
    best_auc = -1.0
    best_path = os.path.join(models_dir, f"{run_name}_best.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, n_batches = 0.0, 0

        for batch in train_loader:
            history_ids = batch["history"].to(device, non_blocking=True)
            hist_mask   = batch["hist_mask"].to(device, non_blocking=True)
            candidates  = batch["candidates"].to(device, non_blocking=True)

            optimizer.zero_grad()
            scores = model(history_ids, candidates, hist_mask)  # (B, 1+K)
            # Positive is always index 0
            target = torch.zeros(scores.size(0), dtype=torch.long, device=device)
            loss = criterion(scores, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total += loss.item()
            n_batches += 1

        avg_loss = total / max(n_batches, 1)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{epochs}] train_loss={avg_loss:.4f} ({dt:.1f}s)")

        history["epoch"].append(epoch)
        history["train_loss"].append(avg_loss)

        if eval_every_epoch:
            print(f"[epoch {epoch}] evaluating on dev...")
            metrics = evaluate(model, eval_loader, device)
            history["val_metrics"].append(metrics)
            if metrics["AUC"] > best_auc:
                best_auc = metrics["AUC"]
                torch.save(model.state_dict(), best_path)
                print(f"[checkpoint] new best AUC={best_auc:.4f} -> {best_path}")

        # Always save last
        torch.save(model.state_dict(),
                   os.path.join(models_dir, f"{run_name}_last.pt"))

    # Persist history
    hist_path = os.path.join(results_dir, f"{run_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] history saved to {hist_path}")

    return {"history": history, "best_auc": best_auc, "best_ckpt": best_path}


# pandas is only imported inside build_or_load_preprocessed; import here too
import pandas as pd  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--neg-k", type=int, default=4)
    parser.add_argument("--run-name", type=str, default="nrms_base")
    parser.add_argument("--limit-train-samples", type=int, default=None)
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        neg_k=args.neg_k,
        run_name=args.run_name,
        limit_train_samples=args.limit_train_samples,
    )
