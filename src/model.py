"""
model.py
--------
Full NRMS model: news encoder + user encoder + dot-product scoring.
"""

import numpy as np
import torch
import torch.nn as nn

from news_encoder import NewsEncoder
from user_encoder import UserEncoder


class NRMSModel(nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray,
                 num_heads: int = 16,
                 head_dim: int = 16,
                 dropout: float = 0.2):
        super().__init__()
        self.news_encoder = NewsEncoder(embedding_matrix, num_heads, head_dim, dropout)
        news_dim = num_heads * head_dim
        self.user_encoder = UserEncoder(news_dim, num_heads, dropout)

    def encode_candidates(self, candidate_ids: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> (B, C, D)
        B, C, L = candidate_ids.shape
        vecs = self.news_encoder(candidate_ids.view(B * C, L))
        return vecs.view(B, C, -1)

    def encode_user(self,
                    history_ids: torch.Tensor,
                    hist_mask: torch.Tensor = None) -> torch.Tensor:
        # (B, H, L) -> user vec (B, D)
        B, H, L = history_ids.shape
        hist_vecs = self.news_encoder(history_ids.view(B * H, L)).view(B, H, -1)
        return self.user_encoder(hist_vecs, hist_mask)

    def forward(self,
                history_ids: torch.Tensor,
                candidate_ids: torch.Tensor,
                hist_mask: torch.Tensor = None) -> torch.Tensor:
        user_vec = self.encode_user(history_ids, hist_mask)        # (B, D)
        cand_vecs = self.encode_candidates(candidate_ids)          # (B, C, D)
        scores = torch.bmm(cand_vecs, user_vec.unsqueeze(-1)).squeeze(-1)
        return scores  # (B, C)

    def score_variable_candidates(self,
                                  history_ids: torch.Tensor,
                                  candidate_list: list,
                                  hist_mask: torch.Tensor = None) -> list:
        """
        For eval: each sample in the batch may have a different # of candidates.
        candidate_list: list of tensors each (C_i, L).
        Returns list of 1-D score arrays.
        """
        user_vec = self.encode_user(history_ids, hist_mask)  # (B, D)
        out = []
        for b, cands in enumerate(candidate_list):
            # cands: (C_i, L)
            cand_vecs = self.news_encoder(cands)             # (C_i, D)
            s = torch.matmul(cand_vecs, user_vec[b])         # (C_i,)
            out.append(s.detach().cpu().numpy())
        return out
