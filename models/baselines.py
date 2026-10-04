"""
Baseline Recommendation Models:
1. PopularityRanker: frequency-based ranking baseline.
2. MatrixFactorizationRanker: classical collaborative filtering.
3. MLPRanker: classical feature-engineered deep learning ranker.
Used for comparative benchmarking against GenRec.
"""

from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PopularityRanker:
    """Ranks candidates by global historical watch frequency."""

    def __init__(self, num_catalog_items: int):
        self.counts = np.zeros(num_catalog_items + 1)

    def fit(self, interactions_list: List[List[Any]]):
        for inter_seq in interactions_list:
            for inter in inter_seq:
                item_id = getattr(inter, "item_id", inter)
                if item_id < len(self.counts):
                    self.counts[item_id] += 1

    def rank_candidates(self, candidate_ids: List[int]) -> List[Tuple[int, float]]:
        scores = [self.counts[cid] if cid < len(self.counts) else 0.0 for cid in candidate_ids]
        total = sum(scores) + 1e-6
        probs = [s / total for s in scores]
        ranked = sorted(zip(candidate_ids, probs), key=lambda x: x[1], reverse=True)
        return ranked


class MatrixFactorizationRanker(nn.Module):
    """Classical Collaborative Filtering with user and item latent factors."""

    def __init__(self, num_users: int = 500, num_items: int = 30, latent_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users + 1, latent_dim)
        self.item_emb = nn.Embedding(num_items + 1, latent_dim)
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # user_ids: [B], item_ids: [B, K]
        u = self.user_emb(user_ids).unsqueeze(1)  # [B, 1, D]
        i = self.item_emb(item_ids)               # [B, K, D]
        scores = (u * i).sum(dim=-1)              # [B, K]
        scores = scores + self.item_bias(item_ids).squeeze(-1)
        return scores


class MLPRanker(nn.Module):
    """
    Classical Tabular Feature-Engineered Deep Learning Ranker:
    Takes manual features (watch count, avg completion, genre frequencies, device one-hot)
    and passes through a multi-layer perceptron.
    """

    def __init__(self, feature_dim: int = 15, item_dim: int = 32, num_items: int = 30):
        super().__init__()
        self.item_embeddings = nn.Embedding(num_items + 1, item_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim + item_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_features: torch.Tensor, candidate_item_ids: torch.Tensor) -> torch.Tensor:
        # user_features: [B, feature_dim]
        # candidate_item_ids: [B, K]
        batch_size, k = candidate_item_ids.shape
        cand_embeds = self.item_embeddings(candidate_item_ids)  # [B, K, item_dim]
        user_feat_expanded = user_features.unsqueeze(1).expand(-1, k, -1)  # [B, K, feature_dim]
        combined = torch.cat([user_feat_expanded, cand_embeds], dim=-1)  # [B, K, feature_dim + item_dim]
        scores = self.mlp(combined).squeeze(-1)  # [B, K]
        return scores
