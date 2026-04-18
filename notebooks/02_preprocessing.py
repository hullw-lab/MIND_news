# %% [markdown]
# # 02 — Preprocessing & Feature Engineering
#
# **Goal:** Convert raw TSV files into tensors ready for the model.
# Produces a cached bundle (`cache/preprocessed.pkl`) and a GloVe
# embedding matrix (`cache/embedding_matrix.npy`) so `03_training.ipynb`
# runs without redoing this work.

# %%
import os, sys, pickle, time
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import numpy as np
import pandas as pd

from data_loader import (
    download_mind, download_glove,
    load_news, load_behaviors,
    NewsTokenizer, load_glove_matrix,
    encode_all_news, build_train_samples, build_eval_samples,
)

os.makedirs("../cache", exist_ok=True)

# %% [markdown]
# ## 1. Ensure data is downloaded

# %%
download_mind("../data")
# NOTE: GloVe is ~820MB zipped. Uncomment to download automatically,
# or manually place glove.6B.300d.txt in data/glove/.
glove_path = download_glove("../data")
glove_path = "../data/glove/glove.6B.300d.txt"
print(f"GloVe path: {glove_path}")
print(f"Exists? {os.path.exists(glove_path)}")

# %% [markdown]
# ## 2. Load TSVs

# %%
train_news = load_news("../data/MINDsmall_train/news.tsv")
dev_news   = load_news("../data/MINDsmall_dev/news.tsv")
train_beh  = load_behaviors("../data/MINDsmall_train/behaviors.tsv")
dev_beh    = load_behaviors("../data/MINDsmall_dev/behaviors.tsv")

# Union news so dev candidates are always encodable
all_news = (
    pd.concat([train_news, dev_news], ignore_index=True)
      .drop_duplicates("news_id")
      .reset_index(drop=True)
)
print(f"Train news : {len(train_news):,}")
print(f"Dev news   : {len(dev_news):,}")
print(f"Union      : {len(all_news):,}")

# %% [markdown]
# ## 3. Build vocabulary from TRAIN titles only
#
# We build the vocabulary only from training titles to avoid leakage, but we will
# *encode* dev news titles with the same vocabulary (OOV tokens become `<UNK>`).

# %%
MAX_TITLE_LEN = 30
MIN_WORD_FREQ = 2

tokenizer = NewsTokenizer(max_title_len=MAX_TITLE_LEN, min_word_freq=MIN_WORD_FREQ)
tokenizer.build_vocab(train_news["title"].tolist())

# %% [markdown]
# ## 4. Build GloVe embedding matrix

# %%
if os.path.exists(glove_path):
    embedding_matrix = load_glove_matrix(glove_path, tokenizer.word2idx, embed_dim=300)
    np.save("../cache/embedding_matrix.npy", embedding_matrix)
    print(f"Saved embedding matrix: {embedding_matrix.shape}")
else:
    print("WARNING: GloVe file not found. Run download_glove() above, or the "
          "model will fall back to random initialization in training.")

# %% [markdown]
# ## 5. Encode every news title

# %%
t0 = time.time()
news_encoded = encode_all_news(all_news, tokenizer)
print(f"Encoded {len(news_encoded):,} articles in {time.time()-t0:.1f}s")
print("Example encoding for first article:")
first_id = all_news['news_id'].iloc[0]
print(f"  {first_id} -> {news_encoded[first_id][:10]}... (len={len(news_encoded[first_id])})")

# %% [markdown]
# ## 6. Build TRAIN samples (negative sampling)

# %%
NEG_K = 4
MAX_HISTORY = 50

train_samples = build_train_samples(
    train_beh, news_encoded, neg_k=NEG_K, max_history=MAX_HISTORY, seed=42
)

# Quick sanity check
ex = train_samples[0]
print(f"Example train sample:")
print(f"  history len = {len(ex['history'])}")
print(f"  candidates  = {len(ex['candidates'])} (1 positive + {NEG_K} negatives)")

# %% [markdown]
# ## 7. Build EVAL samples (full candidate lists)

# %%
eval_samples = build_eval_samples(dev_beh, news_encoded, max_history=MAX_HISTORY)

# Distribution of candidates per impression in eval
cand_counts = [len(s["candidates"]) for s in eval_samples]
print(f"Eval impressions        : {len(eval_samples):,}")
print(f"Avg candidates / imp    : {np.mean(cand_counts):.1f}")
print(f"Median candidates / imp : {np.median(cand_counts):.0f}")

# %% [markdown]
# ## 8. Cache the full preprocessed bundle

# %%
bundle = {
    "tokenizer": tokenizer,
    "news_encoded": news_encoded,
    "train_beh": train_beh,
    "dev_beh": dev_beh,
    "all_news": all_news,
}
with open("../cache/preprocessed.pkl", "wb") as f:
    pickle.dump(bundle, f)

# Save samples separately so you can rebuild training samples with a
# different seed or neg_k without re-tokenizing everything.
with open("../cache/train_samples.pkl", "wb") as f:
    pickle.dump(train_samples, f)
with open("../cache/eval_samples.pkl", "wb") as f:
    pickle.dump(eval_samples, f)

print("Preprocessing cache written to ../cache/")
for fn in os.listdir("../cache"):
    p = os.path.join("../cache", fn)
    print(f"  {fn:30s}  {os.path.getsize(p)/1e6:8.2f} MB")
