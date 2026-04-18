"""
data_loader.py
--------------
Handles everything from raw TSV to PyTorch DataLoaders:
  - downloading MIND-small + GloVe
  - NLTK tokenization and vocabulary building
  - GloVe embedding matrix construction
  - parsing behaviors.tsv into (history, candidates, labels) samples
  - train-time and eval-time Dataset classes
"""

from __future__ import annotations

import os
import pickle
import random
import urllib.request
import zipfile
from collections import Counter
from typing import Dict, List, Tuple

import nltk
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def _ensure_nltk() -> None:
    """Download NLTK tokenizer resources if missing. Safe to call repeatedly."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"[warn] nltk.download('{resource}') failed: {e}. "
                      f"If you see tokenization errors, run "
                      f"`python -c \"import nltk; nltk.download('{resource}')\"` manually.")


_ensure_nltk()


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

MIND_URLS = {
    "train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "dev":   "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"


def download_mind(data_dir: str = "data") -> None:
    """Download and extract MIND-small train + dev splits.

    NOTE: Microsoft disabled public access on the original Azure blob in 2024.
    If the download fails with HTTP 409, grab the dataset from Kaggle instead:
        https://www.kaggle.com/datasets/arashnic/mind-news-dataset
    Extract so you have data/MINDsmall_train/ and data/MINDsmall_dev/ each
    containing behaviors.tsv and news.tsv.
    """
    os.makedirs(data_dir, exist_ok=True)
    for split, url in MIND_URLS.items():
        out_dir = os.path.join(data_dir, f"MINDsmall_{split}")
        if os.path.exists(os.path.join(out_dir, "behaviors.tsv")):
            print(f"[skip] {split} already present at {out_dir}")
            continue
        zip_path = os.path.join(data_dir, f"MINDsmall_{split}.zip")
        print(f"[download] {split} -> {zip_path}")
        try:
            urllib.request.urlretrieve(url, zip_path)
        except urllib.error.HTTPError as e:
            raise RuntimeError(
                f"MIND download failed ({e}). Microsoft's public blob is offline.\n"
                f"Manual fix: download from Kaggle and extract into {data_dir}/\n"
                f"  https://www.kaggle.com/datasets/arashnic/mind-news-dataset\n"
                f"You need: {data_dir}/MINDsmall_train/{{behaviors.tsv,news.tsv}} "
                f"and {data_dir}/MINDsmall_dev/{{behaviors.tsv,news.tsv}}"
            ) from e
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        os.remove(zip_path)
        print(f"[done] extracted to {out_dir}")

def download_glove(data_dir: str = "data") -> str:
    """Download GloVe 6B 300d. Returns path to glove.6B.300d.txt."""
    glove_dir = os.path.join(data_dir, "glove")
    target = os.path.join(glove_dir, "glove.6B.300d.txt")
    if os.path.exists(target):
        print(f"[skip] GloVe already present at {target}")
        return target
    os.makedirs(glove_dir, exist_ok=True)
    zip_path = os.path.join(glove_dir, "glove.6B.zip")
    print(f"[download] GloVe 6B (this is ~820MB)")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        # Only extract the 300d file to save disk.
        z.extract("glove.6B.300d.txt", glove_dir)
    os.remove(zip_path)
    print(f"[done] extracted to {target}")
    return target


# ---------------------------------------------------------------------------
# Reading TSVs
# ---------------------------------------------------------------------------

NEWS_COLS = [
    "news_id", "category", "subcategory", "title",
    "abstract", "url", "title_entities", "abstract_entities",
]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def load_news(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=NEWS_COLS, quoting=3)


def load_behaviors(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", names=BEH_COLS, quoting=3)


# ---------------------------------------------------------------------------
# Tokenization + vocabulary
# ---------------------------------------------------------------------------

class NewsTokenizer:
    """NLTK word-level tokenizer with PAD=0, UNK=1 reserved."""

    PAD, UNK = 0, 1

    def __init__(self, max_title_len: int = 30, min_word_freq: int = 2):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx: Dict[str, int] = {"<PAD>": self.PAD, "<UNK>": self.UNK}

    def build_vocab(self, titles: List[str]) -> None:
        counts: Counter = Counter()
        for t in titles:
            if isinstance(t, str):
                counts.update(nltk.word_tokenize(t.lower()))
        for word, c in counts.items():
            if c >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f"[vocab] size = {len(self.word2idx):,} "
              f"(from {len(counts):,} unique tokens, min_freq={self.min_word_freq})")

    def encode_title(self, title: str) -> List[int]:
        if not isinstance(title, str):
            tokens: List[str] = []
        else:
            tokens = nltk.word_tokenize(title.lower())
        idx = [self.word2idx.get(t, self.UNK) for t in tokens]
        if len(idx) < self.max_title_len:
            idx = idx + [self.PAD] * (self.max_title_len - len(idx))
        else:
            idx = idx[: self.max_title_len]
        return idx

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "max_title_len": self.max_title_len,
                "min_word_freq": self.min_word_freq,
            }, f)

    @classmethod
    def load(cls, path: str) -> "NewsTokenizer":
        with open(path, "rb") as f:
            d = pickle.load(f)
        tok = cls(max_title_len=d["max_title_len"], min_word_freq=d["min_word_freq"])
        tok.word2idx = d["word2idx"]
        return tok


def load_glove_matrix(glove_path: str,
                      word2idx: Dict[str, int],
                      embed_dim: int = 300,
                      seed: int = 42) -> np.ndarray:
    """Build (vocab_size, embed_dim) matrix. OOV words get small random vectors."""
    rng = np.random.default_rng(seed)
    matrix = (rng.standard_normal((len(word2idx), embed_dim)) * 0.1).astype("float32")
    matrix[0] = 0.0  # PAD

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word2idx:
                matrix[word2idx[word]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"[glove] matched {found:,}/{len(word2idx):,} words "
          f"({100*found/len(word2idx):.1f}% coverage)")
    return matrix


# ---------------------------------------------------------------------------
# Behavior parsing
# ---------------------------------------------------------------------------

def encode_all_news(news_df: pd.DataFrame,
                    tokenizer: NewsTokenizer) -> Dict[str, List[int]]:
    """news_id -> list[int] of token indices (fixed length)."""
    out: Dict[str, List[int]] = {}
    for nid, title in zip(news_df["news_id"].values, news_df["title"].values):
        out[nid] = tokenizer.encode_title(title)
    return out


def build_train_samples(behaviors_df: pd.DataFrame,
                        news_encoded: Dict[str, List[int]],
                        neg_k: int = 4,
                        max_history: int = 50,
                        seed: int = 42) -> List[dict]:
    """
    One training sample per clicked article. For each positive, sample K negatives
    from the SAME impression (standard NRMS protocol).
    """
    rng = random.Random(seed)
    samples: List[dict] = []
    skipped_no_neg = 0

    for hist_str, imp_str in zip(behaviors_df["history"].values,
                                 behaviors_df["impressions"].values):
        # History
        if isinstance(hist_str, str) and hist_str.strip():
            history = [nid for nid in hist_str.split() if nid in news_encoded]
        else:
            history = []
        history = history[-max_history:]

        if not isinstance(imp_str, str):
            continue

        pos, neg = [], []
        for token in imp_str.split():
            nid, label = token.rsplit("-", 1)
            if nid not in news_encoded:
                continue
            (pos if label == "1" else neg).append(nid)

        if not pos or not neg:
            skipped_no_neg += 1
            continue

        for p in pos:
            if len(neg) >= neg_k:
                sampled = rng.sample(neg, neg_k)
            else:
                # With replacement when impression has few negatives
                sampled = [rng.choice(neg) for _ in range(neg_k)]
            samples.append({
                "history": history,
                "candidates": [p] + sampled,  # index 0 is positive
            })

    print(f"[train samples] {len(samples):,} built, {skipped_no_neg:,} impressions skipped")
    return samples


def build_eval_samples(behaviors_df: pd.DataFrame,
                       news_encoded: Dict[str, List[int]],
                       max_history: int = 50,
                       drop_zero_history: bool = True) -> List[dict]:
    """One sample per impression with FULL candidate list and labels.

    drop_zero_history: when True (default), drop impressions where the user has
    no clickable history. These are true cold-start cases and would cause
    all-masked attention rows. Cold-start is analyzed separately in
    notebook 03 by NOT dropping these, then handled via the AdditiveAttention
    fallback. Leave True for headline metrics to match standard NRMS protocol.
    """
    samples: List[dict] = []
    n_zero_hist, n_bad_labels = 0, 0
    for hist_str, imp_str in zip(behaviors_df["history"].values,
                                 behaviors_df["impressions"].values):
        if isinstance(hist_str, str) and hist_str.strip():
            history = [nid for nid in hist_str.split() if nid in news_encoded]
        else:
            history = []
        history = history[-max_history:]

        if drop_zero_history and len(history) == 0:
            n_zero_hist += 1
            continue

        if not isinstance(imp_str, str):
            continue

        cand_ids, labels = [], []
        for token in imp_str.split():
            nid, label = token.rsplit("-", 1)
            if nid in news_encoded:
                cand_ids.append(nid)
                labels.append(int(label))

        if not cand_ids or sum(labels) == 0 or sum(labels) == len(labels):
            n_bad_labels += 1
            continue  # AUC undefined

        samples.append({
            "history": history,
            "candidates": cand_ids,
            "labels": labels,
        })
    print(f"[eval samples] {len(samples):,} built "
          f"(dropped: {n_zero_hist:,} zero-history, {n_bad_labels:,} bad-labels)")
    return samples


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class MINDTrainDataset(Dataset):
    """Fixed-size tensors. history_mask is 1 for real items, 0 for padding."""

    def __init__(self, samples: List[dict],
                 news_encoded: Dict[str, List[int]],
                 max_history: int = 50,
                 max_title_len: int = 30):
        self.samples = samples
        self.news_encoded = news_encoded
        self.max_history = max_history
        self.max_title_len = max_title_len
        self.pad_title = [0] * max_title_len

    def __len__(self) -> int:
        return len(self.samples)

    def _pad_history(self, hist_ids: List[str]) -> Tuple[List[List[int]], List[int]]:
        hist = [self.news_encoded[nid] for nid in hist_ids]
        mask = [1] * len(hist)
        while len(hist) < self.max_history:
            hist.append(self.pad_title)
            mask.append(0)
        return hist[-self.max_history:], mask[-self.max_history:]

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        hist, hist_mask = self._pad_history(s["history"])
        cands = [self.news_encoded[nid] for nid in s["candidates"]]
        return {
            "history": torch.tensor(hist, dtype=torch.long),
            "hist_mask": torch.tensor(hist_mask, dtype=torch.long),
            "candidates": torch.tensor(cands, dtype=torch.long),
        }


class MINDEvalDataset(Dataset):
    """Variable candidate counts -> returned as Python lists (collate_fn handles it)."""

    def __init__(self, samples: List[dict],
                 news_encoded: Dict[str, List[int]],
                 max_history: int = 50,
                 max_title_len: int = 30):
        self.samples = samples
        self.news_encoded = news_encoded
        self.max_history = max_history
        self.max_title_len = max_title_len
        self.pad_title = [0] * max_title_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        hist = [self.news_encoded[nid] for nid in s["history"]]
        mask = [1] * len(hist)
        while len(hist) < self.max_history:
            hist.append(self.pad_title)
            mask.append(0)
        cands = [self.news_encoded[nid] for nid in s["candidates"]]
        return {
            "history": torch.tensor(hist[-self.max_history:], dtype=torch.long),
            "hist_mask": torch.tensor(mask[-self.max_history:], dtype=torch.long),
            "candidates": torch.tensor(cands, dtype=torch.long),
            "labels": np.asarray(s["labels"], dtype=np.int64),
        }


def eval_collate(batch: List[dict]) -> dict:
    """Stack fixed fields, keep candidates and labels as lists (variable length)."""
    return {
        "history": torch.stack([b["history"] for b in batch]),
        "hist_mask": torch.stack([b["hist_mask"] for b in batch]),
        "candidates": [b["candidates"] for b in batch],
        "labels": [b["labels"] for b in batch],
    }
