"""FullSync psychomotor-retardation branch for AVEC2014.

The module topology mirrors the final supplied AVEC2017 PRE implementation:
71-D tokens -> temporal context encoder -> ordinary evidence and severe MIL
readouts -> base regression + non-negative severe residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.float().unsqueeze(-1)
    return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def masked_std(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mean = masked_mean(x, mask).unsqueeze(1)
    weights = mask.float().unsqueeze(-1)
    var = ((x - mean).square() * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    return torch.sqrt(var + 1e-6)


def masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, ratio: float, min_k: int) -> torch.Tensor:
    outputs = []
    for i in range(values.size(0)):
        valid = values[i][mask[i]]
        k = min(valid.numel(), max(min_k, int(round(valid.numel() * ratio))))
        outputs.append(torch.topk(valid, k=k).values.mean() if k else values.new_tensor(0.0))
    return torch.stack(outputs)


def masked_topk_features(
    x: torch.Tensor, score: torch.Tensor, mask: torch.Tensor, ratio: float, min_k: int
) -> torch.Tensor:
    outputs = []
    for i in range(x.size(0)):
        valid_idx = torch.where(mask[i])[0]
        k = min(valid_idx.numel(), max(min_k, int(round(valid_idx.numel() * ratio))))
        if k == 0:
            outputs.append(x.new_zeros(x.size(-1)))
            continue
        local = torch.topk(score[i, valid_idx], k=k).indices
        outputs.append(x[i, valid_idx[local]].mean(0))
    return torch.stack(outputs)


def masked_topm_severe(logits: torch.Tensor, mask: torch.Tensor, ratio: float, min_k: int):
    mean_logits, mean_probs = [], []
    for i in range(logits.size(0)):
        valid = logits[i][mask[i]]
        k = min(valid.numel(), max(min_k, int(round(valid.numel() * ratio))))
        if k == 0:
            mean_logits.append(logits.new_tensor(0.0))
            mean_probs.append(logits.new_tensor(0.0))
            continue
        top = torch.topk(valid, k=k).values
        mean_logits.append(top.mean())
        mean_probs.append(torch.sigmoid(top).mean())
    return torch.stack(mean_logits), torch.stack(mean_probs)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_large: int, dropout: float):
        super().__init__()
        self.dw_large = nn.Conv1d(dim, dim, kernel_large, padding=kernel_large // 2, groups=dim)
        self.dw_small = nn.Conv1d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        conv_input = x.transpose(1, 2)
        conv = self.dw_large(conv_input) + self.dw_small(conv_input)
        x = self.norm1(x + self.dropout(F.gelu(conv).transpose(1, 2)))
        x = x * mask.float().unsqueeze(-1)
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x * mask.float().unsqueeze(-1)


class PREFullSyncNet(nn.Module):
    """Exact module family used by the final AVEC2017 severe-aware PRE."""

    def __init__(
        self,
        in_dim: int = 71,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        topk_ratio: float = 0.10,
        min_topk: int = 3,
        severe_topk_ratio: float = 0.15,
        severe_min_topk: int = 4,
        severe_residual_max: float = 5.0,
    ):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)
        self.severe_topk_ratio = float(severe_topk_ratio)
        self.severe_min_topk = int(severe_min_topk)
        self.severe_residual_max = float(severe_residual_max)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        kernels = [5, 7, 9, 11, 13, 15]
        self.temporal = nn.ModuleList([
            TemporalConvBlock(hidden_dim, kernels[i % len(kernels)], dropout) for i in range(num_layers)
        ])
        self.token_score = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.token_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.severe_token_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

        # mean/std/attention/top-k embeddings plus nine evidence statistics.
        video_dim = hidden_dim * 4 + 9
        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_dim), nn.Linear(video_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.severe_residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3), nn.Linear(hidden_dim + 3, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 4),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        x = self.input_proj(tokens) * mask.float().unsqueeze(-1)
        for block in self.temporal:
            x = block(x, mask)

        local_score = self.token_score(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn_logits = self.token_attn(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1) * mask.float()
        attn = attn / attn.sum(1, keepdim=True).clamp_min(1e-6)

        severe_token_logit = self.severe_token_head(x).squeeze(-1).masked_fill(~mask, -1e4)
        severe_bag_logit, severe_prob = masked_topm_severe(
            severe_token_logit, mask, self.severe_topk_ratio, self.severe_min_topk
        )
        score_zero = local_score.masked_fill(~mask, 0.0)
        severe_zero = severe_token_logit.masked_fill(~mask, 0.0)
        n = mask.float().sum(1).clamp_min(1.0)

        mean_pool = masked_mean(x, mask)
        std_pool = masked_std(x, mask)
        attn_pool = (x * attn.unsqueeze(-1)).sum(1)
        topk_pool = masked_topk_features(x, local_score, mask, self.topk_ratio, self.min_topk)
        score_mean = (score_zero * mask.float()).sum(1) / n
        score_std = torch.sqrt(
            ((score_zero - score_mean.unsqueeze(1)).square() * mask.float()).sum(1) / n + 1e-6
        )
        score_attn = (score_zero * attn).sum(1)
        score_topk = masked_topk_mean(local_score, mask, self.topk_ratio, self.min_topk)
        attn_peak = attn.max(1).values
        n_tokens_scaled = torch.log1p(n) / 6.0
        severe_logit_mean = (severe_zero * mask.float()).sum(1) / n

        video_features = torch.cat([
            mean_pool, std_pool, attn_pool, topk_pool,
            score_mean[:, None], score_std[:, None], score_attn[:, None], score_topk[:, None],
            attn_peak[:, None], n_tokens_scaled[:, None], severe_logit_mean[:, None],
            severe_bag_logit[:, None], severe_prob[:, None],
        ], dim=1)
        embedding = self.video_proj(video_features)
        base_pred = self.reg_head(embedding).squeeze(-1)
        residual_input = torch.cat([
            embedding, severe_prob[:, None], severe_bag_logit[:, None], score_topk[:, None]
        ], dim=1)
        raw_residual = F.softplus(self.severe_residual_head(residual_input).squeeze(-1))
        severe_residual = (
            self.severe_residual_max
            * torch.sigmoid(raw_residual / self.severe_residual_max)
            * severe_prob
        )
        pred = base_pred + severe_residual
        return {
            "pred": pred, "base_pred": base_pred, "severe_residual": severe_residual,
            "severe_prob": severe_prob, "severe_bag_logit": severe_bag_logit,
            "severe_token_logit": severe_zero, "severity_logits": self.severity_head(embedding),
            "embedding": embedding, "token_score": score_zero, "token_attn": attn,
            "retard_conf": self.conf_head(embedding).squeeze(-1),
            "score_topk": score_topk, "attn_peak": attn_peak,
        }
