"""
news_encoder.py
---------------
Encodes a news title into a single d-dim vector.

Pipeline (per NRMS paper):
  word_ids -> embedding -> dropout -> linear projection
           -> multi-head self-attention -> dropout -> additive attention -> vec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdditiveAttention(nn.Module):
    """Aggregate (batch, seq, dim) -> (batch, dim) with a learned query."""

    def __init__(self, dim: int, hidden_dim: int = 200):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # x: (B, L, D), mask: (B, L) with 1=keep
        e = torch.tanh(self.proj(x))                   # (B, L, H)
        scores = self.query(e).squeeze(-1)             # (B, L)
        if mask is not None:
            # If a row has no valid positions at all, softmax would yield NaNs.
            # Detect these rows and give them a uniform distribution over pads
            # (the resulting output is just the mean of zero-padded vectors = 0,
            # which is the sensible "no information" default).
            all_masked = (mask.sum(dim=-1) == 0)       # (B,)
            scores = scores.masked_fill(mask == 0, -1e9)
            if all_masked.any():
                # For fully-masked rows, replace scores with zeros so softmax
                # is uniform instead of NaN. The padded x values are zero
                # anyway so the output vector remains zero.
                scores = torch.where(
                    all_masked.unsqueeze(-1).expand_as(scores),
                    torch.zeros_like(scores),
                    scores,
                )
        weights = F.softmax(scores, dim=-1)            # (B, L)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)


class NewsEncoder(nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray,
                 num_heads: int = 16,
                 head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        vocab_size, embed_dim = embedding_matrix.shape
        attn_dim = num_heads * head_dim

        self.word_embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=False,
            padding_idx=0,
        )
        self.proj = nn.Linear(embed_dim, attn_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.additive_attn = AdditiveAttention(attn_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = attn_dim

    def forward(self, title_ids: torch.Tensor) -> torch.Tensor:
        # title_ids: (B, L)
        key_padding_mask = (title_ids == 0)           # True where pad
        # Guard against all-pad rows (used as padding in history). Without this,
        # nn.MultiheadAttention produces NaN for those rows. Un-mask position 0
        # for all-masked rows; the embedding at index 0 is zero, so the resulting
        # vector is effectively zero and doesn't contaminate the user encoder.
        all_masked = key_padding_mask.all(dim=-1)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        x = self.dropout(self.word_embed(title_ids))  # (B, L, E)
        x = self.proj(x)                              # (B, L, attn_dim)
        x, _ = self.multihead_attn(
            x, x, x, key_padding_mask=key_padding_mask
        )
        x = self.dropout(x)
        # Additive attention with the ORIGINAL pad mask (not un-masked version).
        # Its own fallback handles fully-masked rows by returning zeros.
        attn_mask = (title_ids != 0).long()
        return self.additive_attn(x, attn_mask)       # (B, attn_dim)
