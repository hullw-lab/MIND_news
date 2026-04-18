"""
Smoke test with synthetic MIND-format data.
Does NOT require real data download.
Verifies:
  - TSV parsing
  - tokenizer + GloVe (random matrix stand-in)
  - sample construction
  - forward pass through NRMSModel
  - evaluation metrics
  - variable-length candidate eval path
"""
import os, sys, shutil, random

# Stub NLTK to avoid network download in sandbox. Real env uses the full tokenizer.
import nltk
nltk.word_tokenize = lambda s: s.lower().split()

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import (
    NewsTokenizer, encode_all_news,
    build_train_samples, build_eval_samples,
    MINDTrainDataset, MINDEvalDataset, eval_collate,
)
from model import NRMSModel
from evaluate import evaluate

SMOKE = "/tmp/smoke_mind"
if os.path.exists(SMOKE):
    shutil.rmtree(SMOKE)
os.makedirs(SMOKE)

# ---- Synthetic news ----
rng = random.Random(0)
words = ["the", "team", "wins", "market", "stocks", "rise", "movie", "review",
         "election", "debate", "players", "score", "tech", "company", "launch",
         "drama", "film", "star", "budget", "deal", "coach", "season", "game"]
cats  = ["sports", "finance", "entertainment", "news"]

news_rows = []
for i in range(200):
    n = rng.randint(5, 12)
    title = " ".join(rng.choices(words, k=n))
    news_rows.append((f"N{i}", rng.choice(cats), "sub", title, "abs", "http://x", "[]", "[]"))
pd.DataFrame(news_rows).to_csv(f"{SMOKE}/news.tsv", sep="\t", header=False, index=False)

# ---- Synthetic behaviors ----
beh_rows = []
imp_id = 0
for u in range(100):
    user = f"U{u}"
    hist = rng.sample([f"N{i}" for i in range(200)], k=rng.randint(0, 15))
    hist_str = " ".join(hist) if hist else ""
    # Each user has 2-4 impressions
    for _ in range(rng.randint(2, 4)):
        cand_ids = rng.sample([f"N{i}" for i in range(200)], k=rng.randint(10, 25))
        n_pos = rng.randint(1, 3)
        pos_set = set(rng.sample(cand_ids, k=n_pos))
        imp_str = " ".join(f"{c}-{1 if c in pos_set else 0}" for c in cand_ids)
        beh_rows.append((imp_id, user, "11/13/2019 8:36:57 AM", hist_str, imp_str))
        imp_id += 1
pd.DataFrame(beh_rows).to_csv(f"{SMOKE}/behaviors.tsv", sep="\t", header=False, index=False)

# ---- Load and process ----
from data_loader import load_news, load_behaviors
news_df = load_news(f"{SMOKE}/news.tsv")
beh_df  = load_behaviors(f"{SMOKE}/behaviors.tsv")
print(f"[smoke] news={len(news_df)}, beh={len(beh_df)}")

tok = NewsTokenizer(max_title_len=12, min_word_freq=1)
tok.build_vocab(news_df["title"].tolist())

# Fake GloVe matrix
embedding_matrix = (np.random.randn(len(tok.word2idx), 50) * 0.1).astype("float32")
embedding_matrix[0] = 0

news_encoded = encode_all_news(news_df, tok)

train_samples = build_train_samples(beh_df.iloc[:200], news_encoded, neg_k=4, max_history=10)
eval_samples  = build_eval_samples(beh_df.iloc[200:], news_encoded, max_history=10)

print(f"[smoke] train_samples={len(train_samples)}, eval_samples={len(eval_samples)}")

train_ds = MINDTrainDataset(train_samples, news_encoded, max_history=10, max_title_len=12)
eval_ds  = MINDEvalDataset(eval_samples, news_encoded, max_history=10, max_title_len=12)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
eval_loader  = DataLoader(eval_ds,  batch_size=4, shuffle=False, num_workers=0,
                          collate_fn=eval_collate)

device = torch.device("cpu")
# Use 4 heads × 8 head_dim = 32 to stay tiny; 50-d GloVe projects up to 32
model = NRMSModel(embedding_matrix, num_heads=4, head_dim=8, dropout=0.1).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
crit = torch.nn.CrossEntropyLoss()

# 2 quick training steps
model.train()
for i, batch in enumerate(train_loader):
    scores = model(batch["history"], batch["candidates"], batch["hist_mask"])
    target = torch.zeros(scores.size(0), dtype=torch.long)
    loss = crit(scores, target)
    opt.zero_grad(); loss.backward(); opt.step()
    if i >= 1: break
print(f"[smoke] training step ran. loss={loss.item():.4f}")

# Eval
metrics = evaluate(model, eval_loader, device, verbose=False)
print(f"[smoke] eval metrics: {metrics}")
assert 0.0 <= metrics["AUC"] <= 1.0
assert 0.0 <= metrics["MRR"] <= 1.0
print("[smoke] PASSED")
