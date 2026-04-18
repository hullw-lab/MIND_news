"""
user_encoder.py
---------------
Aggregates a sequence of clicked-news vectors into a single user vector.
"""

import torch
import torch.nn as nn

from news_encoder import AdditiveAttention


class UserEncoder(nn.Module):
    def __init__(self,
                 news_dim: int,
                 num_heads: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        # news_dim must be divisible by num_heads (it is: 16*16=256, heads=16)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=news_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.additive_attn = AdditiveAttention(news_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                clicked_news_vecs: torch.Tensor,
                hist_mask: torch.Tensor = None) -> torch.Tensor:
        # clicked_news_vecs: (B, H, D); hist_mask: (B, H) with 1=real
        key_padding_mask = None
        if hist_mask is not None:
            key_padding_mask = (hist_mask == 0)
            # nn.MultiheadAttention produces NaN when a row is fully masked.
            # For those rows, un-mask position 0 (which is a zero-vector pad
            # anyway, so it contributes nothing meaningful). The AdditiveAttention
            # downstream handles the all-masked case too, so the output is 0.
            all_masked = key_padding_mask.all(dim=-1)
            if all_masked.any():
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False

        x, _ = self.multihead_attn(
            clicked_news_vecs,
            clicked_news_vecs,
            clicked_news_vecs,
            key_padding_mask=key_padding_mask,
        )
        x = self.dropout(x)
        return self.additive_attn(x, hist_mask)  # (B, D)
