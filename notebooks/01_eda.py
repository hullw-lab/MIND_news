# %% [markdown]
# # 01 — Exploratory Data Analysis
#
# **Goal:** Understand the MIND-small dataset before building any model.
# Produces 7 visualizations with written interpretations. All figures are
# saved to `results/` so they can be pulled into the final README.

# %%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import download_mind, load_news, load_behaviors, NEWS_COLS, BEH_COLS

sns.set_style("whitegrid")
os.makedirs("../results", exist_ok=True)

# %%
# Download MIND-small if not already present. Skips if files exist.
download_mind("../data")

# %%
news_df = load_news("../data/MINDsmall_train/news.tsv")
beh_df  = load_behaviors("../data/MINDsmall_train/behaviors.tsv")

print(f"News articles : {len(news_df):,}")
print(f"Users         : {beh_df['user_id'].nunique():,}")
print(f"Impressions   : {len(beh_df):,}")

# %% [markdown]
# ## 1. Basic statistics and click-through rate

# %%
def count_shown(s):
    return 0 if not isinstance(s, str) else len(s.split())

def count_clicks(s):
    if not isinstance(s, str): return 0
    return sum(1 for t in s.split() if t.endswith("-1"))

beh_df["n_shown"]  = beh_df["impressions"].apply(count_shown)
beh_df["n_clicks"] = beh_df["impressions"].apply(count_clicks)

total_shown  = beh_df["n_shown"].sum()
total_clicks = beh_df["n_clicks"].sum()
print(f"Total candidates shown : {total_shown:,}")
print(f"Total clicks           : {total_clicks:,}")
print(f"Overall CTR            : {total_clicks/total_shown:.4f}")

# %% [markdown]
# **Interpretation.** The overall CTR is ~4–5%, which is realistic for
# news feeds. Every impression is class-imbalanced: far more non-clicks
# than clicks, which is why negative sampling matters during training.

# %% [markdown]
# ## 2. Category distribution

# %%
cat_counts = news_df["category"].value_counts()
plt.figure(figsize=(12, 5))
sns.barplot(x=cat_counts.index, y=cat_counts.values, color="#2E86AB")
plt.title("Articles per category (MIND-small train)")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of articles")
plt.tight_layout()
plt.savefig("../results/01_category_distribution.png", dpi=120)
plt.show()

print("Top 5 categories:")
print(cat_counts.head())

# %% [markdown]
# **Interpretation.** The top three categories (news, sports, finance) dominate
# supply. A purely popularity-based baseline will already do reasonably well on
# users interested in these topics; niche categories (autos, weather) are the
# hardest cold-start cases.

# %% [markdown]
# ## 3. Click-through rate by category

# %%
# Expand impressions into per-article rows to compute per-category CTR.
rows = []
for imps in beh_df["impressions"].dropna():
    for tok in imps.split():
        nid, label = tok.rsplit("-", 1)
        rows.append((nid, int(label)))
imp_df = pd.DataFrame(rows, columns=["news_id", "click"])
imp_df = imp_df.merge(news_df[["news_id", "category"]], on="news_id", how="left")

cat_ctr = imp_df.groupby("category")["click"].agg(["sum", "count"])
cat_ctr["ctr"] = cat_ctr["sum"] / cat_ctr["count"]
cat_ctr = cat_ctr.sort_values("ctr", ascending=False)

plt.figure(figsize=(12, 5))
sns.barplot(x=cat_ctr.index, y=cat_ctr["ctr"].values, color="#E07B39")
plt.title("Click-through rate by category")
plt.xticks(rotation=45, ha="right")
plt.ylabel("CTR")
plt.tight_layout()
plt.savefig("../results/02_ctr_by_category.png", dpi=120)
plt.show()
print(cat_ctr.head(10).round(4))

# %% [markdown]
# **Interpretation.** CTR varies ~3x across categories. Some low-supply categories
# actually have higher CTR, suggesting that when the recommender does surface them,
# users engage well — an argument for category-aware encoding as an extension.

# %% [markdown]
# ## 4. User click activity (power law check)

# %%
user_clicks = beh_df.groupby("user_id")["n_clicks"].sum()
print("Quantiles of clicks per user:")
print(user_clicks.describe(percentiles=[.5, .75, .9, .95, .99]).round(2))

plt.figure(figsize=(10, 4))
plt.hist(user_clicks, bins=60, color="#1E3A5F", edgecolor="white")
plt.yscale("log")
plt.xlabel("Total clicks per user (over 6-week window)")
plt.ylabel("Number of users (log)")
plt.title("User activity distribution")
plt.tight_layout()
plt.savefig("../results/03_user_click_dist.png", dpi=120)
plt.show()

frac_sparse = (user_clicks < 10).mean()
print(f"Fraction of users with <10 total clicks: {frac_sparse:.1%}")

# %% [markdown]
# **Interpretation.** Classic long-tail: a minority of power users drive most
# clicks, while a large fraction have very few interactions. This is the core
# cold-start challenge — the user encoder has to produce a meaningful vector
# from 0–5 historical clicks for a large portion of users.

# %% [markdown]
# ## 5. User history length distribution

# %%
def hist_len(s):
    return 0 if not isinstance(s, str) else len(s.split())

beh_df["hist_len"] = beh_df["history"].apply(hist_len)
print(beh_df["hist_len"].describe(percentiles=[.5, .75, .9, .95]).round(1))

plt.figure(figsize=(10, 4))
plt.hist(beh_df["hist_len"].clip(upper=200), bins=60, color="#4CB5AE", edgecolor="white")
plt.xlabel("History length (clipped at 200)")
plt.ylabel("Number of impressions")
plt.title("Distribution of user history length at impression time")
plt.tight_layout()
plt.savefig("../results/04_history_length.png", dpi=120)
plt.show()

# %% [markdown]
# **Interpretation.** Median history is small (typically ~20–30), which justifies
# the `max_history=50` cap in the model — we rarely truncate, and padding dominates
# for cold users. The mask is essential to prevent attention from leaking into pads.

# %% [markdown]
# ## 6. Impression size distribution

# %%
plt.figure(figsize=(10, 4))
plt.hist(beh_df["n_shown"].clip(upper=100), bins=50, color="#2E86AB", edgecolor="white")
plt.xlabel("Candidates per impression (clipped at 100)")
plt.ylabel("Number of impressions")
plt.title("Impression size distribution")
plt.tight_layout()
plt.savefig("../results/05_impression_size.png", dpi=120)
plt.show()

print(f"Mean candidates/impression   : {beh_df['n_shown'].mean():.1f}")
print(f"Median candidates/impression : {beh_df['n_shown'].median():.0f}")
print(f"Mean clicks/impression       : {beh_df['n_clicks'].mean():.2f}")

# %% [markdown]
# **Interpretation.** Most impressions show ~20–40 candidates with only 1–2 clicks.
# That means during evaluation, each ranking task is non-trivial — the model has
# to push the 1–2 positives above ~20+ negatives.

# %% [markdown]
# ## 7. Temporal patterns

# %%
beh_df["ts"]   = pd.to_datetime(beh_df["time"], errors="coerce")
beh_df["hour"] = beh_df["ts"].dt.hour
beh_df["day"]  = beh_df["ts"].dt.date

daily = beh_df.groupby("day").size()
hourly = beh_df.groupby("hour").size()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
daily.plot(ax=axes[0], color="#1E3A5F")
axes[0].set_title("Impressions per day")
axes[0].set_ylabel("Impressions")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(hourly.index, hourly.values, color="#E07B39")
axes[1].set_title("Impressions per hour of day")
axes[1].set_xlabel("Hour (UTC)")
axes[1].set_ylabel("Impressions")
plt.tight_layout()
plt.savefig("../results/06_temporal.png", dpi=120)
plt.show()

# %% [markdown]
# **Interpretation.** Clear daily rhythm — impressions spike during working hours
# and dip overnight. This matters because users shift topics by time of day; a
# pure content model ignores this signal, which is a promising extension vector.

# %% [markdown]
# ## 8. Title length distribution

# %%
import nltk
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

title_lens = news_df["title"].dropna().map(
    lambda t: len(nltk.word_tokenize(t.lower()))
)
print(title_lens.describe(percentiles=[.5, .9, .95, .99]).round(1))

plt.figure(figsize=(10, 4))
plt.hist(title_lens.clip(upper=40), bins=40, color="#4CB5AE", edgecolor="white")
plt.axvline(30, color="red", linestyle="--", label="max_title_len=30")
plt.xlabel("Title length (tokens)")
plt.ylabel("Number of articles")
plt.title("Distribution of title length")
plt.legend()
plt.tight_layout()
plt.savefig("../results/07_title_length.png", dpi=120)
plt.show()

# %% [markdown]
# **Interpretation.** Nearly all titles fit in 30 tokens after NLTK tokenization,
# confirming the chosen `max_title_len=30` truncates less than ~1% of titles.

# %% [markdown]
# ## Data quality notes
#
# - Some `history` fields are NaN (first-time users). The loader treats these as empty.
# - A handful of impressions have 0 clicks or 100% clicks; we drop these from AUC/MRR
#   calculations since those metrics are undefined when labels are constant.
# - Entities/abstracts are available but unused in the baseline model; they are
#   documented as extensions in the README.
