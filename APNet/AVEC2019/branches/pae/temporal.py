# temporal.py
# Duration-Aware Psychomotor Retardation Branch

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, keepdim: bool = False):
    m = mask.float().unsqueeze(-1)
    s = (x * m).sum(dim=dim, keepdim=keepdim)
    d = m.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return s / d


def masked_std(x: torch.Tensor, mask: torch.Tensor, dim: int = 1):
    mu = masked_mean(x, mask, dim=dim, keepdim=True)
    m = mask.float().unsqueeze(-1)
    var = (((x - mu) ** 2) * m).sum(dim=dim) / m.sum(dim=dim).clamp_min(1.0)
    return torch.sqrt(var + 1e-6)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, kernel_large: int = 9, kernel_small: int = 3, dropout: float = 0.1, ff_mult: int = 2):
        super().__init__()
        assert kernel_large % 2 == 1 and kernel_small % 2 == 1
        self.dw_large = nn.Conv1d(dim, dim, kernel_large, padding=kernel_large // 2, groups=dim)
        self.dw_small = nn.Conv1d(dim, dim, kernel_small, padding=kernel_small // 2, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        residual = x
        y = x.transpose(1, 2)
        y = self.dw_large(y) + self.dw_small(y)
        y = F.gelu(y).transpose(1, 2)
        x = self.norm1(residual + self.drop(y))
        x = x * mask.float().unsqueeze(-1)

        residual = x
        y = self.ff(x)
        x = self.norm2(residual + self.drop(y))
        x = x * mask.float().unsqueeze(-1)
        return x


class ARDRetardationNetV2(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.15,
        n_severity_bins: int = 4,
        max_score: float = 24.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_severity_bins = n_severity_bins
        self.max_score = max_score

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
            TemporalConvBlock(hidden_dim, kernel_large=kernels[i % len(kernels)], kernel_small=3, dropout=dropout, ff_mult=2)
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

        video_dim = hidden_dim * 3 + 4
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
        local_score = local_score.masked_fill(~mask, 0.0)

        attn_logits = self.token_attn(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e4)
        attn = torch.softmax(attn_logits, dim=1)
        attn = attn * mask.float()
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)

        mean_pool = masked_mean(x, mask)
        std_pool = masked_std(x, mask)
        attn_pool = torch.sum(x * attn.unsqueeze(-1), dim=1)

        n = mask.float().sum(dim=1).clamp_min(1.0)
        score_mean = (local_score * mask.float()).sum(dim=1) / n
        score_std = torch.sqrt((((local_score - score_mean.unsqueeze(1)) ** 2) * mask.float()).sum(dim=1) / n + 1e-6)
        score_attn = (local_score * attn).sum(dim=1)
        attn_peak = attn.max(dim=1).values

        video_feat = torch.cat([
            mean_pool,
            std_pool,
            attn_pool,
            score_mean.unsqueeze(1),
            score_std.unsqueeze(1),
            score_attn.unsqueeze(1),
            attn_peak.unsqueeze(1),
        ], dim=1)

        z = self.video_proj(video_feat)
        pred = self.reg_head(z).squeeze(-1)
        # Important: do NOT sigmoid/clamp during training. Bounded output caused mean-collapse in v1.

        return {
            "pred": pred,
            "severity_logits": self.severity_head(z),
            "embedding": z,
            "token_score": local_score,
            "token_attn": attn,
            "retard_conf": self.conf_head(z).squeeze(-1),
        }
