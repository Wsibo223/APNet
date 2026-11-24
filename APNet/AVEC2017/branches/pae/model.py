# branches/pae/model.py
# Severe-aware ARD-Retardation branch.
#
# Built on the stable v3 Top-k branch.
# New components:
#   1) token-level Severe Evidence Head
#   2) MIL top-m severe evidence pooling
#   3) non-negative severe residual to raise high-score predictions only when severe evidence is strong

import torch
import torch.nn as nn
import torch.nn.functional as F

from branches.pae.temporal import TemporalConvBlock, masked_mean, masked_std
from branches.pae.pooling import masked_topk_mean, masked_topk_feature_mean


def masked_topm_mean(logits: torch.Tensor, mask: torch.Tensor, ratio: float = 0.20, min_k: int = 5):
    """Bag-level MIL pooling over token logits."""
    outs_logit, outs_prob = [], []
    for i in range(logits.size(0)):
        valid = logits[i][mask[i]]
        if valid.numel() == 0:
            outs_logit.append(logits.new_tensor(0.0))
            outs_prob.append(logits.new_tensor(0.0))
            continue
        k = max(min_k, int(round(valid.numel() * ratio)))
        k = min(k, valid.numel())
        top_logits = torch.topk(valid, k=k, largest=True).values
        outs_logit.append(top_logits.mean())
        outs_prob.append(torch.sigmoid(top_logits).mean())
    return torch.stack(outs_logit), torch.stack(outs_prob)


class ARDSevereAwareNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        n_severity_bins: int = 4,
        topk_ratio: float = 0.10,
        min_topk: int = 3,
        severe_topk_ratio: float = 0.20,
        severe_min_topk: int = 5,
        severe_residual_max: float = 10.0,
    ):
        super().__init__()
        self.topk_ratio = float(topk_ratio)
        self.min_topk = int(min_topk)
        self.severe_topk_ratio = float(severe_topk_ratio)
        self.severe_min_topk = int(severe_min_topk)
        self.severe_residual_max = float(severe_residual_max)

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
            TemporalConvBlock(hidden_dim, kernel_large=kernels[i % len(kernels)], kernel_small=3,
                              dropout=dropout, ff_mult=2)
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

        # New: severe evidence is a token-level MIL signal, not a new backbone.
        self.severe_token_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # mean, std, attention, hard top-k feature + 9 scalar statistics
        video_dim = hidden_dim * 4 + 9
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

        # Positive residual branch for severe compensation.
        self.severe_residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim + 3),
            nn.Linear(hidden_dim + 3, hidden_dim // 2),
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

        local_score = self.token_score(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn_logits = self.token_attn(x).squeeze(-1).masked_fill(~mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        severe_token_logit = self.severe_token_head(x).squeeze(-1).masked_fill(~mask, -1e4)
        severe_bag_logit, severe_prob = masked_topm_mean(
            severe_token_logit, mask, self.severe_topk_ratio, self.severe_min_topk
        )

        score_zero = local_score.masked_fill(~mask, 0.0)
        severe_zero = severe_token_logit.masked_fill(~mask, 0.0)
        n = mask.float().sum(dim=1).clamp_min(1.0)

        mean_pool = masked_mean(x, mask)
        std_pool = masked_std(x, mask)
        attn_pool = torch.sum(x * attn.unsqueeze(-1), dim=1)
        topk_feat = masked_topk_feature_mean(x, local_score, mask, self.topk_ratio, self.min_topk)

        score_mean = (score_zero * mask.float()).sum(dim=1) / n
        score_std = torch.sqrt((((score_zero - score_mean.unsqueeze(1)) ** 2) * mask.float()).sum(dim=1) / n + 1e-6)
        score_attn = (score_zero * attn).sum(dim=1)
        score_topk = masked_topk_mean(local_score, mask, self.topk_ratio, self.min_topk)
        attn_peak = attn.max(dim=1).values
        n_tokens_scaled = torch.log1p(n) / 6.0
        severe_logit_mean = (severe_zero * mask.float()).sum(dim=1) / n

        video_feat = torch.cat([
            mean_pool, std_pool, attn_pool, topk_feat,
            score_mean.unsqueeze(1), score_std.unsqueeze(1), score_attn.unsqueeze(1),
            score_topk.unsqueeze(1), attn_peak.unsqueeze(1), n_tokens_scaled.unsqueeze(1),
            severe_logit_mean.unsqueeze(1), severe_bag_logit.unsqueeze(1), severe_prob.unsqueeze(1),
        ], dim=1)

        z = self.video_proj(video_feat)
        base_pred = self.reg_head(z).squeeze(-1)

        residual_in = torch.cat([z, severe_prob.unsqueeze(1), severe_bag_logit.unsqueeze(1), score_topk.unsqueeze(1)], dim=1)
        raw_residual = F.softplus(self.severe_residual_head(residual_in).squeeze(-1))
        severe_residual = self.severe_residual_max * torch.sigmoid(raw_residual / self.severe_residual_max) * severe_prob
        pred = base_pred + severe_residual

        return {
            "pred": pred,
            "base_pred": base_pred,
            "severe_residual": severe_residual,
            "severe_prob": severe_prob,
            "severe_bag_logit": severe_bag_logit,
            "severe_token_logit": severe_token_logit.masked_fill(~mask, 0.0),
            "severity_logits": self.severity_head(z),
            "embedding": z,
            "token_score": score_zero,
            "token_attn": attn,
            "retard_conf": self.conf_head(z).squeeze(-1),
            "score_topk": score_topk,
        }
