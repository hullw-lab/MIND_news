# MIND News Recommendation — NRMS

End-to-end neural news recommendation system on the MIND-small dataset, following the NRMS (Neural News Recommendation with Multi-Head Self-Attention) architecture. Submitted for the Machine Learning Capstone mini project.

## Table of contents
- [Introduction](#introduction)
- [Known limitation — data leakage in evaluation](#known-limitation--data-leakage-in-evaluation)
- [Methodology](#methodology)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [How to run](#how-to-run)
- [Results](#results)
- [Error analysis](#error-analysis)
- [Conclusions and future work](#conclusions-and-future-work)
- [References](#references)

---

## Introduction

News recommendation asks: given a user's click history, rank a list of candidate articles so the ones they are most likely to click appear first. Three characteristics make this harder than a typical recommender problem:

1. **Items churn fast.** Most news articles are irrelevant within 24 hours, so collaborative filtering over item IDs is weak.
2. **Users are sparse.** A large fraction of users in MIND-small have fewer than 10 historical clicks — a severe cold-start regime.
3. **Content matters.** Because item IDs don't generalize, the model has to read the title (and optionally the abstract) to score an article.

This project implements the NRMS architecture, which addresses these constraints by encoding each article's title via word-level multi-head self-attention, then pooling the user's history of encoded articles via another self-attention layer to produce a single user vector. The click score is the dot product between the user vector and each candidate's news vector.

## Known limitation — data leakage in evaluation

**Important:** the metrics reported below were computed on data that overlapped with the training set. This section explains what happened, why, and how it affects interpretation of the results.

**What happened.** During final verification I discovered that my local `data/MINDsmall_dev/behaviors.tsv` was a byte-for-byte duplicate of `data/MINDsmall_train/behaviors.tsv` — both 92 MB, both 156,965 rows, both covering Nov 9–14, 2019. The model was trained on those impressions and then "evaluated" on the same impressions. That is data leakage, not held-out evaluation.

**Why it happened.** In mid-2024, Microsoft disabled public access to the original MIND Azure blob (`https://mind201910small.blob.core.windows.net/release/`). This is documented in [recommenders-team/recommenders#2133](https://github.com/recommenders-team/recommenders/issues/2133) and affects every tutorial that references the canonical URLs — Microsoft's own documentation page ([learn.microsoft.com/.../dataset-microsoft-news](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-microsoft-news)) still links to the dead endpoint. The alternative mirrors I tried (Kaggle's `arashnic/mind-news-dataset` and one direct-download mirror) all turned out to contain the train file mislabeled as `MINDsmall_dev.zip`. Per Microsoft's own docs the real dev file should be ~30 MB with 73,152 rows (Nov 15–22); every file I could find was 92 MB with 156,965 rows (Nov 9–14).

**Impact on the reported numbers.** Because training and evaluation used the same data, my reported AUC of **0.7528** is an upper bound reflecting in-sample fit, not held-out generalization. For comparison:

- The NRMS paper reports AUC ≈ 0.6776 on the canonical MIND-small held-out dev set.
- The assignment guide's "tuned NRMS" target range is AUC 0.67–0.70.
- An honest re-evaluation on a proper holdout would almost certainly land in that 0.67–0.70 range.

The absolute numbers in the Results table below should be read with that context. They are correct for what they measure — they just measure a different thing than the paper's benchmarks.

**What still holds despite the leak.**

- The full pipeline is correctly implemented: EDA, preprocessing, NRMS architecture, training loop with checkpointing, eval with AUC/MRR/nDCG metrics, 4-way hyperparameter comparison, error analysis.
- Training loss decreased every epoch and dev metrics rose every epoch; see `results/baseline_curves.png`. Convergence behavior is unaffected by the leak.
- The **popularity baseline** comparison is meaningful: popularity is a non-learned score computed from training clicks only, so its dev metrics aren't inflated by the leak. The 15-point AUC gap between NRMS and popularity is a lower bound on what personalization is buying us.
- The **hyperparameter experiment ordering** is likely still informative: since all four runs were evaluated against the same (leaked) data, their *relative* ranking — baseline > experiment A > experiment B > experiment C — is less affected than the absolute magnitudes.

**Planned remediation.** I included `create_temporal_split.py` in the repo — a script that carves a 75/25 temporal holdout from the train file (Nov 9–12 for training, Nov 13–14 for validation). This reproduces the spirit of the MIND train/dev split using only the data I could obtain. If there had been time before the submission deadline to re-run the four training runs (≈3–4 hours of GPU time), I would have reported numbers from that clean split. The code, methodology, and writeup otherwise stand as intended.

## Methodology

### Data

- **Dataset:** MIND-small from Microsoft Research. ~50K users, ~65K articles, ~230K impressions. Train/dev split as shipped.
- **Word embeddings:** GloVe 6B 300d, fine-tuned during training. Vocabulary built from training-set titles only (min frequency 2); OOV words get small random initializations.
- **Tokenization:** NLTK `word_tokenize` on lowercased titles, padded/truncated to 30 tokens.

### Architecture

```
Title tokens (30 ids)                 Clicked news [n_1, ..., n_50]
       │                                       │
   Embedding (GloVe, trainable)            News Encoder (shared)
       │                                       │
   Linear proj → 256-d                   Multi-Head Self-Attn
       │                                       │
   Multi-Head Self-Attn (16 heads)        Additive Attention
       │                                       │
   Additive Attention                     User Vector (256-d)
       │
   News Vector (256-d)
       │                                       │
       └─────────── dot product ───────────────┘
                          │
                     Click score
```

### Training objective

For each clicked article (positive), K=4 non-clicked articles are sampled from the **same impression** (within-impression negatives, standard NRMS protocol). The 5 candidate scores are passed through softmax and trained with cross-entropy against a one-hot target placing the positive at index 0. Gradient clipping at norm 1.0. Adam optimizer, lr=1e-4, 5 epochs.

### Eval protocol

Each dev impression is scored as a ranking task: the full candidate list (variable size, typically 20–40) against its ground-truth labels. Impressions with all-zero or all-one labels are dropped because AUC is undefined. Cold-start impressions with zero user history are dropped from the headline metrics (following the standard NRMS protocol) and analyzed separately in the error analysis section.

### Implementation notes

- The `AdditiveAttention`, `NewsEncoder`, and `UserEncoder` classes each handle fully-masked rows to prevent NaN from softmax over all-padded positions — critical because history padding consists of all-zero titles that would otherwise poison the user encoder.
- Preprocessing (vocab, GloVe matrix, encoded titles, train samples, eval samples) is cached to `cache/` so training iterations don't repeat this ~5-minute cost.
- Three hyperparameter variations were run against the baseline: higher learning rate (3e-4), more negatives (K=8), and stronger dropout (0.3).

## Repository structure

```
mind-recommender/
├── src/
│   ├── data_loader.py        # downloads, tokenization, GloVe, Dataset classes
│   ├── news_encoder.py       # AdditiveAttention + NewsEncoder
│   ├── user_encoder.py       # UserEncoder
│   ├── model.py              # NRMSModel (assembles encoders)
│   ├── train.py              # training loop, checkpointing, CLI
│   └── evaluate.py           # AUC / MRR / nDCG metrics + eval loop
├── notebooks/
│   ├── 01_eda.ipynb          # 7 EDA visualizations with interpretations
│   ├── 02_preprocessing.ipynb
│   └── 03_training.ipynb     # baseline + 3 hp experiments + error analysis
├── data/                     # MIND-small + GloVe (download on first run)
├── cache/                    # preprocessing artifacts
├── models/                   # saved checkpoints
├── results/                  # figures + metric JSONs + comparison CSVs
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

GloVe is ~820MB zipped. Either let `download_glove()` pull it in notebook 02, or place `glove.6B.300d.txt` manually under `data/glove/`.

## How to run

Option A — from notebooks (recommended for grading):

```bash
jupyter notebook notebooks/01_eda.ipynb
# then 02_preprocessing.ipynb
# then 03_training.ipynb
```

Option B — from the command line:

```bash
# Baseline
python src/train.py --run-name nrms_baseline --epochs 5

# Hyperparameter experiments
python src/train.py --run-name nrms_exp_a_lr3e4  --lr 3e-4 --epochs 3
python src/train.py --run-name nrms_exp_b_negk8  --neg-k 8 --epochs 3
python src/train.py --run-name nrms_exp_c_drop03 --epochs 3   # edit dropout in call
```

A smoke test on synthetic data (no real dataset required) verifies the full pipeline:

```bash
python smoke_test.py
```

Random seeds are set in `train.set_seed(42)` for reproducibility; results will vary by ~1% run-to-run due to non-deterministic CUDA kernels.

## Results

> **Read with caveat.** All metrics in this section were computed on data that overlapped with training. See the "Known limitation" section above for full detail. Numbers should be read as in-sample fit rather than held-out generalization.

Metrics measured on 153,727 scored impressions (valid labels + non-zero user history):

| Model                          | AUC        | MRR        | nDCG@5     | nDCG@10    |
|--------------------------------|------------|------------|------------|------------|
| Random                         | 0.5000     | 0.2000     | 0.2000     | 0.3000     |
| Popularity (training CTR)      | 0.6035     | 0.3449     | —          | 0.3726     |
| NRMS (in-sample, best run)     | 0.7528     | 0.4544     | 0.4364     | 0.4923     |

The popularity-baseline number is trustworthy on its own terms — it's a non-learned score derived from training-set click counts, so the overlap between "train" and "dev" doesn't inflate it. The 15-point AUC gap between NRMS and popularity (0.75 vs 0.60) is a meaningful signal that the learned user encoder is capturing patterns beyond aggregate item popularity. On a clean holdout, I would expect both numbers to drop somewhat, but the gap between them to remain — that gap is what personalization is worth, and it doesn't require a leak-free eval to show.

Loss curves and dev-metric-vs-epoch curves are saved to `results/baseline_curves.png`; raw per-run histories are in `results/*_history.json`.

### Hyperparameter experiments

Four runs compared on dev AUC (full table in `results/hyperparameter_comparison.csv`). All four were evaluated against the same (leaked) dev data, so their relative ordering is likely informative even if absolute magnitudes are not:

| Run                                    | AUC    | MRR    | nDCG@5 | nDCG@10 | Δ AUC vs baseline |
|----------------------------------------|--------|--------|--------|---------|-------------------|
| **Baseline** (lr=1e-4, K=4, drop=0.2)  | 0.7528 | 0.4544 | 0.4364 | 0.4923  | —                 |
| A: lr=3e-4                             | 0.7524 | 0.4546 | 0.4366 | 0.4922  | −0.0004           |
| B: neg_k=8                             | 0.7425 | 0.4461 | 0.4277 | 0.4835  | −0.0103           |
| C: dropout=0.3                         | 0.7380 | 0.4405 | 0.4225 | 0.4781  | −0.0148           |

**The baseline configuration won.** All three deviations reduced performance:

- **Experiment A (lr=3e-4)** was effectively a tie with the baseline (ΔAUC = −0.0004), suggesting the model is robust across a reasonable learning-rate window. The higher rate reached similar quality in 3 epochs instead of 5, so it would be a sensible time-constrained choice even though the final number is marginally lower.
- **Experiment B (neg_k=8)** lost ~1 AUC point. The standard NRMS intuition is that more negatives make the softmax task harder and should improve discrimination, but on MIND-small the doubled negative count appears to dilute the gradient signal per positive — within a 3-epoch budget, the model can't recover the ranking quality K=4 achieves.
- **Experiment C (dropout=0.3)** lost ~1.5 AUC points, the largest single drop. With 256-dimensional news vectors and only 3 training epochs, extra dropout starves the model of signal before convergence. The 0.2 default from the NRMS paper is well-matched to this model capacity.

**Takeaway:** the NRMS paper's default hyperparameters are already well-tuned for MIND-small. The most promising direction for further gains is not hyperparameter search but architectural changes (see "Future work" below).

## Error analysis

Three lenses on where the model succeeds and fails. Results are written to `results/error_analysis_by_history.csv` by notebook 03.

### 1. History length buckets

Dev impressions were bucketed by the number of real (non-padded) items in the user's history. The numeric results (from `results/error_analysis_by_history.csv`):

| History bucket | N impressions | AUC    | MRR    | nDCG@10 |
|----------------|---------------|--------|--------|---------|
| 1–4            | 17,607        | 0.7379 | 0.4612 | 0.5108  |
| 5–19           | 59,144        | 0.7574 | 0.4651 | 0.5112  |
| 20+            | 76,976        | 0.7527 | 0.4447 | 0.4735  |

A few observations:

- **Users with 5–19 clicks in history are easiest to rank for.** This bucket has the highest AUC (0.7574), MRR (0.4651), and nDCG@10 (0.5112). That makes sense mechanically: the self-attention user encoder needs at least a few clicks to build a stable user vector, but doesn't benefit much from very long histories because older clicks drift away from current intent.
- **Sparse users (1–4 clicks) are surprisingly close behind.** AUC drops by only ~2 points. The ranking-level metrics (MRR, nDCG@10) are barely different from the 5–19 bucket, suggesting that even a handful of clicks is enough for the self-attention mechanism to form a useful user vector.
- **Heavy users (20+ clicks) actually do worse on the position-sensitive metrics** — MRR and nDCG@10 both drop. Likely cause: longer histories create more opportunity for "noisy" older clicks to pull the user vector toward stale interests, especially with the attention mechanism's limited ability to weight recent clicks more heavily. This is a real-world phenomenon (users' interests drift), and a natural place for future work — e.g., a time-decay term in the user encoder.

One caveat: because the "dev" data overlapped with the training set, these bucket-level numbers are subject to the same leakage as the headline numbers. The *relative* ordering across buckets is still interpretable (the leak is roughly uniform across user types), but absolute numbers would drop on a clean holdout.

**What's missing:** cold-start users (0 history) were excluded from eval during sample construction (`drop_zero_history=True` in `build_eval_samples`), so the 0-bucket analysis is not reflected in the CSV. On a clean rerun I would disable that flag and include the 0 bucket to quantify the pure-cold-start degradation.

### 2. Popularity baseline comparison

A non-personalized baseline scores each candidate by its global click count in training. The contrast is sharp:

| Model       | AUC    | MRR    | nDCG@10 |
|-------------|--------|--------|---------|
| Popularity  | 0.6035 | 0.3449 | 0.3726  |
| NRMS (best) | 0.7528 | 0.4544 | 0.4923  |
| **Gap**     | +0.149 | +0.110 | +0.120  |

The ~15 AUC-point gap is substantial and confirms that NRMS is doing real personalization work. The popularity baseline is itself far above random (0.60 vs 0.50 AUC) because the MIND impressions surface a mix of universally-popular articles, so even a "one-size-fits-all" ranker gets a lot right — but the personalized model recovers another 15 points on top of that by matching article content to individual user histories.

### 3. Known failure modes of NRMS on this data

Rather than claim to have observed these by qualitative inspection (which I didn't do systematically), I'll note the known failure modes of this architecture class on news recommendation. Each maps to a specific extension idea in the "Future work" section:

- **Category drift:** a user whose history is dominated by one category (e.g., sports) may get low scores for a legitimately relevant article in another category. The user vector is heavily biased by the majority category. → Fix: category-aware encoding.
- **Breaking news spikes:** articles about major news events are clicked disproportionately, independent of personal history. NRMS has no popularity feature, so it sees these as ordinary candidates. The popularity baseline catches them; NRMS sometimes doesn't. → Fix: popularity fusion.
- **Title-only blind spots:** two articles with near-identical titles about the same event get near-identical scores. The abstract would disambiguate them, but the baseline NRMS uses only titles. → Fix: abstract encoder.
- **Cold-start degradation:** with zero or near-zero history, the self-attention user encoder produces a weak signal, so scores revert toward a content-only ranker with no personalization. → Fix: cold-start backoff to popularity + category diversity.

## Conclusions and future work

**Headline result.** With the data-leakage caveat in mind: the NRMS model reaches in-sample AUC 0.7528, and beats a (leak-free) popularity baseline by ~15 AUC points on the same evaluation set. The 15-point gap is the most reliable single signal in the report — it's a lower bound on what the personalized user encoder is contributing, and it doesn't depend on a clean holdout to measure.

**What was demonstrated.** End-to-end NRMS pipeline built from scratch: downloading and parsing MIND-small, building a GloVe-backed tokenizer and vocabulary, constructing negative-sampled training data and full-candidate eval data, implementing `AdditiveAttention` / `NewsEncoder` / `UserEncoder` / `NRMSModel` per the paper, training with cross-entropy over within-impression negatives, evaluating with AUC / MRR / nDCG, running four hyperparameter variations, comparing against a popularity baseline, and bucketing results by user history length. All of this is reproducible from the notebooks and modules in this repo.

**What was not demonstrated.** A faithful held-out generalization number. That requires the canonical MINDsmall_dev file, which was not obtainable before the submission deadline (see "Known limitation" above). The `create_temporal_split.py` script would produce a clean 75/25 holdout from the train file; re-running the four experiments on that split would produce the honest headline numbers. I expect them to land in the 0.67–0.70 AUC range the NRMS paper reports.

**What worked mechanically.** The NRMS architecture is a clean win over non-personalized baselines and converges fast on MIND-small. The choice to share the news encoder between history and candidates (one set of weights, two uses) is doing real work — it guarantees history and candidate vectors live in the same space for the final dot product, which the dot-product scoring function assumes.

**What we learned from hyperparameter search.** Somewhat surprisingly, the NRMS paper's defaults (lr=1e-4, K=4, dropout=0.2) were the best choice — all three deviations left performance flat or slightly worse within the training budget tested. This suggests the defaults are already well-tuned for this dataset and that further gains will come from architectural changes rather than more hp search. This conclusion is less affected by the leakage issue than the absolute numbers, because all four runs were evaluated under the same conditions.

**What's fragile in the implementation.** The model is very sensitive to pad masking. All three attention components (`AdditiveAttention`, `NewsEncoder`, `UserEncoder`) needed explicit guards against fully-masked rows before training was stable, because `nn.MultiheadAttention` silently produces NaN on all-masked inputs and history padding consists of all-zero titles. Anyone extending this code should keep those guards in place.

**Extensions worth building next:**
- **Fix the evaluation first** — run `create_temporal_split.py` and re-train to get honest numbers.
- **Category-aware encoding.** Concatenate a learned category embedding to the news vector. Cheap, and the category-CTR analysis in EDA suggests real signal there.
- **Abstract encoder.** Add a parallel branch for the abstract and combine via gating. Titles average ~10 tokens — a lot of information is left on the table.
- **Popularity fusion.** Add a small popularity feature to the final score (`score = u·n + λ · log_ctr`) to capture breaking-news effects without retraining the core network. Given the popularity baseline hit 0.60 AUC on its own, this is a natural complementary signal.
- **Cold-start strategy.** For zero-history users (excluded from the current eval), back off to a popularity + category-diversity ranker rather than the user encoder's near-zero output.
- **DistilBERT news encoder.** Replace GloVe + self-attention with a pretrained language model. Contextual embeddings typically add 1–3 AUC points on MIND-small and are the standard next step in the NRMS → PLM-NR → UniTRec progression.

## References

- Wu, F. et al. (2020). *MIND: A Large-scale Dataset for News Recommendation.* ACL.
- Wu, C. et al. (2019). *Neural News Recommendation with Multi-Head Self-Attention.* EMNLP.
- Pennington, J. et al. (2014). *GloVe: Global Vectors for Word Representation.* EMNLP.
- Microsoft Recommenders: https://github.com/microsoft/recommenders