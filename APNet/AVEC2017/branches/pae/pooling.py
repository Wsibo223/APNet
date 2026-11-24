
# branches/pae/pooling.py
# ARD-Retardation with top-k pooling head.
#
# This keeps the same lightweight temporal encoder, but adds top-k evidence pooling
# to avoid diluting strong retardation evidence by mean/attention pooling.

import torch
import torch.nn as nn
import torch.nn.functional as F

from branches.pae.temporal import TemporalConvBlock, masked_mean, masked_std


def masked_topk_mean(values: torch.Tensor, mask: torch.Tensor, k_ratio: float = 0.10, min_k: int = 3):
    """
    values: [B,T]
    mask: [B,T]
    Returns [B] top-k mean over valid positions.
    """
    B, T = values.shape
    out = []
    for i in range(B):
        valid = values[i][mask[i]]
        if valid.numel() == 0:
            out.append(values.new_tensor(0.0))
            continue
        k = max(min_k, int(round(valid.numel() * k_ratio)))
        k = min(k, valid.numel())
        out.append(torch.topk(valid, k=k, largest=True).values.mean())
    return torch.stack(out, dim=0)


def masked_topk_feature_mean(x: torch.Tensor, score: torch.Tensor, mask: torch.Tensor, k_ratio: float = 0.10, min_k: int = 3):
    """
    x: [B,T,C], score: [B,T], mask [B,T]
    Select top-k tokens by score and average their features.
    """
    B, T, C = x.shape
    outs = []
    for i in range(B):
        valid_idx = torch.where(mask[i])[0]
        if valid_idx.numel() == 0:
            outs.append(x.new_zeros(C))
            continue
        valid_score = score[i, valid_idx]
        k = max(min_k, int(round(valid_idx.numel() * k_ratio)))
        k = min(k, valid_idx.numel())
        top_local = torch.topk(valid_score, k=k, largest=True).indices
        top_idx = valid_idx[top_local]
        outs.append(x[i, top_idx].mean(dim=0))
    return torch.stack(outs, dim=0)


class ARDRetardationTopKNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        n_severity_bins: int = 4,
        topk_ratio: float = 0.10,
        min_topk: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        kernels = [5, 7, 9, 11, 13, 15]
        self.temporal = nn.ModuleList([
            TemporalConvBlock(
                hidden_dim,
                kernel_large=kernels[i % len(kernels)],
                kernel_small=3,
                dropout=dropout,
                ff_mult=2,
            )
            for i in range(num_layers)
        ])

        self.token_score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.token_attn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # mean, std, attention pool, top-k feature pool + 6 scalar stats
        video_dim = hidden_dim * 4 + 6
        self.video_proj = nn.Sequential(
            nn.LayerNorm(video_dim),
            nn.Linear(video_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.reg_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.severity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_severity_bins),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor):
        x = self.input_proj(tokens)
        x = x * mask.float().unsqueeze(-1)

        for blk in self.temporal:
            x = blk(x, mask)

        local_score = self.token_score(x).squeeze(-1)
        local_score = local_score.masked_fill(~mask, -1e4)

        attn_logits = self.token_attn(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Local score for statistics: invalid positions set to 0 after top-k calculations.
        score_valid_zero = local_score.masked_fill(~mask, 0.0)

        mean_pool = masked_mean(x, mask)
        std_pool = masked_std(x, mask)
        attn_pool = torch.sum(x * attn.unsqueeze(-1), dim=1)
        topk_feat = masked_topk_feature_mean(x, local_score, mask, self.topk_ratio, self.min_topk)

        n = mask.float().sum(dim=1).clamp_min(1.0)
        score_mean = (score_valid_zero * mask.float()).sum(dim=1) / n
        score_std = torch.sqrt((((score_valid_zero - score_mean.unsqueeze(1)) ** 2) * mask.float()).sum(dim=1) / n + 1e-6)
        score_attn = (score_valid_zero * attn).sum(dim=1)
        score_topk = masked_topk_mean(local_score, mask, self.topk_ratio, self.min_topk)
        attn_peak = attn.max(dim=1).values
        n_tokens_scaled = torch.log1p(n) / 6.0

        video_feat = torch.cat([
            mean_pool,
            std_pool,
            attn_pool,
            topk_feat,
            score_mean.unsqueeze(1),
            score_std.unsqueeze(1),
            score_attn.unsqueeze(1),
            score_topk.unsqueeze(1),
            attn_peak.unsqueeze(1),
            n_tokens_scaled.unsqueeze(1),
        ], dim=1)

        z = self.video_proj(video_feat)
        pred = self.reg_head(z).squeeze(-1)

        return {
            "pred": pred,
            "severity_logits": self.severity_head(z),
            "embedding": z,
            "token_score": score_valid_zero,
            "token_attn": attn,
            "retard_conf": self.conf_head(z).squeeze(-1),
            "score_topk": score_topk,
        }
