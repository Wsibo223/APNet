#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RouteA-v2 / NGRC checks: multi-seed and ablation for AVEC2017.

This script is intentionally narrower than the previous RouteA-final suite.
It focuses on the breakthrough family observed in RouteA-final:

    mlp_res_numb128_score_w0.25_h16_d0.25_s2026

Core idea:
    base_pred  = (1 - w) * expr_pred + w * numb_pred
    final_pred = base_pred + MLP(numbness_features + score_context)

Default mode is FIXED reproduction of the breakthrough candidate:
    feature_set = numb128_score
    w           = 0.25
    hidden      = 16
    dropout     = 0.25
    seed        = 2026
    epochs      = 120
    dev eval    = every epoch

It also supports a small NGRC-only search mode, without re-opening the full model zoo.

Selection protocol:
    - fixed mode: candidate is pre-defined; dev only selects the best epoch.
    - search mode: candidates are fitted on train; dev selects the candidate/epoch; test is report-only.

No test labels are used for training or early stopping.
"""

import os
import re
import json
import time
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_OK = True
except Exception:
    TORCH_OK = False
    torch = None
    nn = None
    F = None

ID_CANDS = ["video_id", "vid", "id", "filename", "file", "name"]
GT_CANDS = ["gt", "score", "label", "y", "PHQ", "phq"]
PRED_CANDS = ["pred", "prediction", "final_pred", "pred_calib", "score_pred", "y_pred"]

LEAK_PATTERNS = [
    r"ground", r"target", r"label", r"loss", r"oracle",
    r"abs_error", r"sq_error", r"anchor_error", r"(^|_)error($|_)",
    r"residual_target", r"target_residual",
    r"(^|_)gt($|_)", r"(^|_)y($|_)", r"phq",
]
NONFEATURE_PATTERNS = [r"^split$", r"^bin$"]
BIN_ORDER = ["minimal_0_5", "mild_6_11", "moderate_12_17", "severe_18_24"]


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if TORCH_OK and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    return x


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_OK:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = bool(deterministic)
        try:
            torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
        except Exception:
            pass


def pick_col(df: pd.DataFrame, cands: List[str]) -> Optional[str]:
    for c in cands:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def numeric_clean(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def bin_name(y: float) -> str:
    y = float(y)
    if y < 6:
        return "minimal_0_5"
    if y < 12:
        return "mild_6_11"
    if y < 18:
        return "moderate_12_17"
    return "severe_18_24"


def safe_corr(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return 0.0
    v = np.corrcoef(y, p)[0, 1]
    return float(v) if np.isfinite(v) else 0.0


def ccc_np(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    my, mp = y.mean(), p.mean()
    vy, vp = np.var(y), np.var(p)
    cov = np.mean((y - my) * (p - mp))
    return float(2 * cov / (vy + vp + (my - mp) ** 2 + 1e-8))


def metrics_np(y, p) -> Dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if len(y) == 0:
        return {"n": 0, "rmse": 0.0, "mae": 0.0, "ccc": 0.0, "corr": 0.0,
                "pred_mean": 0.0, "pred_std": 0.0, "gt_mean": 0.0, "gt_std": 0.0, "bias": 0.0}
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "ccc": ccc_np(y, p),
        "corr": safe_corr(y, p),
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
        "gt_mean": float(y.mean()),
        "gt_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "bias": float((p - y).mean()),
    }


def is_safe_col(c, id_col=None, gt_col=None, pred_col=None, allow_residual=True) -> bool:
    if c in [id_col, gt_col, pred_col]:
        return False
    cl = str(c).lower()
    for pat in LEAK_PATTERNS + NONFEATURE_PATTERNS:
        if re.search(pat, cl):
            return False
    if (not allow_residual) and "residual" in cl:
        return False
    return True


def normalize_pred_csv(path: str, prefix: str, pred_override: str = "", allow_residual: bool = True):
    df = pd.read_csv(path)
    id_col = pick_col(df, ID_CANDS)
    gt_col = pick_col(df, GT_CANDS)
    pred_col = pred_override if pred_override else pick_col(df, PRED_CANDS)
    if id_col is None or gt_col is None or pred_col is None:
        raise RuntimeError(f"Cannot infer id/gt/pred in {path}. columns={df.columns.tolist()}")

    out = pd.DataFrame({
        "video_id": df[id_col].astype(str),
        "gt": numeric_clean(df[gt_col]),
        f"{prefix}_pred": numeric_clean(df[pred_col]),
    })

    kept, dropped, blocks = [], [], []
    for c in df.columns:
        if not is_safe_col(c, id_col, gt_col, pred_col, allow_residual=allow_residual):
            dropped.append(c)
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            dropped.append(c)
            continue
        blocks.append(pd.Series(numeric_clean(df[c]).values, name=f"{prefix}_{c}"))
        kept.append(c)
    if blocks:
        out = pd.concat([out] + blocks, axis=1)
    return out, kept, dropped


def load_embed_dir(embed_dir: str, split: str, prefix: str, allow_residual: bool = True):
    if not embed_dir:
        return None, None
    cands = [
        os.path.join(embed_dir, f"{split}_embeddings.csv"),
        os.path.join(embed_dir, f"{split}_features.csv"),
        os.path.join(embed_dir, f"{split}_embedding.csv"),
    ]
    path = next((p for p in cands if os.path.exists(p)), "")
    if not path:
        return None, None
    df = pd.read_csv(path)
    id_col = pick_col(df, ID_CANDS)
    gt_col = pick_col(df, GT_CANDS)
    pred_col = pick_col(df, PRED_CANDS)
    if id_col is None:
        raise RuntimeError(f"No id column in embedding file {path}")

    out = pd.DataFrame({"video_id": df[id_col].astype(str)})
    if gt_col is not None:
        out["gt"] = numeric_clean(df[gt_col])

    kept, dropped, blocks = [], [], []
    for c in df.columns:
        if c in [id_col, gt_col, pred_col]:
            dropped.append(c)
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            dropped.append(c)
            continue
        if not is_safe_col(c, id_col, gt_col, pred_col, allow_residual=allow_residual):
            dropped.append(c)
            continue
        blocks.append(pd.Series(numeric_clean(df[c]).values, name=f"{prefix}_emb_{c}"))
        kept.append(c)
    if blocks:
        out = pd.concat([out] + blocks, axis=1)
    return out, {"path": path, "kept_n": len(kept), "dropped_n": len(dropped), "kept_preview": kept[:20]}


def load_split(args, split: str):
    expr_path = getattr(args, f"expr_{split}") or os.path.join(args.expr_dir, f"{split}_predictions.csv")
    numb_path = getattr(args, f"numb_{split}") or os.path.join(args.numb_dir, f"{split}_predictions.csv")
    expr, expr_kept, expr_drop = normalize_pred_csv(expr_path, "expr", args.expr_pred_col, args.allow_residual_features)
    numb, numb_kept, numb_drop = normalize_pred_csv(numb_path, "numb", args.numb_pred_col, args.allow_residual_features)
    df = expr.merge(numb, on=["video_id", "gt"], how="inner")
    if len(df) == 0:
        raise RuntimeError(f"Empty expr/numb merge for {split}")

    emb_logs = {}
    for prefix, d in [("expr", args.expr_embed_dir), ("numb", args.numb_embed_dir)]:
        emb, elog = load_embed_dir(d, split, prefix, args.allow_residual_features)
        emb_logs[prefix] = elog
        if emb is not None:
            keys = ["video_id", "gt"] if "gt" in emb.columns else ["video_id"]
            df = df.merge(emb, on=keys, how="left")

    df = df.replace([np.inf, -np.inf], np.nan)
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    df[num_cols] = df[num_cols].fillna(0.0)

    df["bin"] = [bin_name(v) for v in df["gt"].values]
    df["mean_pred"] = 0.5 * (df["expr_pred"] + df["numb_pred"])
    df["pred_min"] = np.minimum(df["expr_pred"], df["numb_pred"])
    df["pred_max"] = np.maximum(df["expr_pred"], df["numb_pred"])
    df["pred_gap"] = df["numb_pred"] - df["expr_pred"]
    df["pred_abs_gap"] = np.abs(df["pred_gap"])
    df["pred_prod"] = df["expr_pred"] * df["numb_pred"] / 24.0
    df["expr_sq"] = df["expr_pred"] ** 2 / 24.0
    df["numb_sq"] = df["numb_pred"] ** 2 / 24.0

    # Diagnostics only; excluded from feature sets.
    df["expr_abs_err"] = np.abs(df["expr_pred"] - df["gt"])
    df["numb_abs_err"] = np.abs(df["numb_pred"] - df["gt"])
    df["numb_advantage"] = df["expr_abs_err"] - df["numb_abs_err"]
    df["oracle_choose_numb"] = (df["numb_advantage"] > 0).astype(int)
    df["oracle_pred"] = np.where(df["oracle_choose_numb"] == 1, df["numb_pred"], df["expr_pred"])

    log = {
        "split": split,
        "n": len(df),
        "expr_path": expr_path,
        "numb_path": numb_path,
        "expr_kept_n": len(expr_kept),
        "numb_kept_n": len(numb_kept),
        "expr_pred_dropped_preview": expr_drop[:30],
        "numb_pred_dropped_preview": numb_drop[:30],
        "embedding_logs": emb_logs,
        "columns_n": len(df.columns),
    }
    print(f"[{split}] n={len(df)} | cols={len(df.columns)} | expr={expr_path} | numb={numb_path}", flush=True)
    return df, log


def score_core_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in [
        "expr_pred", "numb_pred", "mean_pred", "pred_min", "pred_max", "pred_gap", "pred_abs_gap",
        "pred_prod", "expr_sq", "numb_sq",
    ] if c in df.columns]


def user_drop_col(c: str, args) -> bool:
    """Drop feature columns by user-specified regex patterns for ablation.

    This is applied only to candidate fusion features, not to labels or ids.
    Example: --drop_feature_regex "severe|severity|bag_logit"
    """
    pats = getattr(args, "drop_feature_regex", []) or []
    if isinstance(pats, str):
        pats = [pats]
    cl = str(c)
    for pat in pats:
        if pat and re.search(pat, cl, flags=re.IGNORECASE):
            return True
    return False


def all_safe_numeric_feature_cols(df: pd.DataFrame, args) -> List[str]:
    bad = {
        "video_id", "gt", "bin",
        "expr_abs_err", "numb_abs_err", "numb_advantage", "oracle_choose_numb", "oracle_pred",
    }
    cols = []
    for c in df.columns:
        if c in bad:
            continue
        if not np.issubdtype(df[c].dtype, np.number):
            continue
        if not is_safe_col(c, allow_residual=args.allow_residual_features):
            continue
        if user_drop_col(c, args):
            continue
        vals = df[c].values.astype(float)
        if np.nanstd(vals) < args.min_feature_std:
            continue
        cols.append(c)
    return cols


def family_cols(cols: List[str], family: str) -> List[str]:
    if family == "numb_video_proj":
        return [c for c in cols if c.startswith("numb_emb_numb_video_proj_")]
    if family == "numb_all":
        return [c for c in cols if c.startswith("numb_") or c.startswith("numb_emb_")]
    if family == "expr_all":
        return [c for c in cols if c.startswith("expr_") or c.startswith("expr_emb_")]
    return []


def rank_cols_by_train_corr(train: pd.DataFrame, cols: List[str], target_col: str = "gt") -> List[Tuple[float, str]]:
    y = train[target_col].values.astype(float)
    out = []
    for c in cols:
        v = train[c].values.astype(float)
        s = abs(safe_corr(y, v))
        # Semantic priors only; no specific dimension is hard-coded.
        if c.startswith("numb_emb_numb_video_proj_"):
            s += 0.015
        out.append((float(s), c))
    out.sort(reverse=True)
    return out


def unique_keep(seq: List[str]) -> List[str]:
    out, seen = [], set()
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def build_ngrc_feature_sets(train: pd.DataFrame, dev: pd.DataFrame, args) -> Dict[str, List[str]]:
    all_cols = all_safe_numeric_feature_cols(pd.concat([train, dev], axis=0, ignore_index=True), args)
    core = score_core_cols(train)
    numb_proj = family_cols(all_cols, "numb_video_proj")
    numb_ranked = [c for _, c in rank_cols_by_train_corr(train, family_cols(all_cols, "numb_all"))]

    sets = {
        "numb128_score": unique_keep(core + numb_ranked[:128]),
        "score_numbproj": unique_keep(core + numb_proj),
        "numb64_score": unique_keep(core + numb_ranked[:64]),
        "numb192_score": unique_keep(core + numb_ranked[:192]),
    }
    cleaned = {}
    for k, v in sets.items():
        vv = [c for c in v if c in train.columns]
        if len(vv) > 0:
            cleaned[k] = vv[:args.max_features_per_set]
    return cleaned


def standardize_fit(df: pd.DataFrame, cols: List[str]):
    X = df[cols].values.astype(np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-6] = 1.0
    return {"cols": cols, "mu": mu, "sd": sd}


def standardize_apply(df: pd.DataFrame, st):
    X = df[st["cols"]].values.astype(np.float32)
    return (X - st["mu"]) / st["sd"]


@dataclass
class Candidate:
    name: str
    kind: str
    train_pred: np.ndarray
    dev_pred: np.ndarray
    test_pred: np.ndarray
    params: dict
    frozen: Optional[dict] = None


def score_candidate(dev_metrics: Dict[str, float], args) -> float:
    return (
        dev_metrics["rmse"]
        - args.select_ccc_w * dev_metrics["ccc"]
        - args.select_corr_w * dev_metrics["corr"]
        + args.select_bias_w * abs(dev_metrics["bias"])
    )


if TORCH_OK:
    class MLPDirect(nn.Module):
        def __init__(self, in_dim, hidden=64, depth=2, dropout=0.2, output_mode="sigmoid24"):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)
            self.output_mode = output_mode
        def forward(self, x, expr_pred=None, numb_pred=None):
            z = self.net(x).squeeze(-1)
            if self.output_mode == "sigmoid24":
                return 24.0 * torch.sigmoid(z)
            return z

    class MLPGateCorrection(nn.Module):
        def __init__(self, in_dim, hidden=64, depth=2, dropout=0.2, corr_scale=8.0):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            self.body = nn.Sequential(*layers)
            self.gate = nn.Linear(d, 1)
            self.corr = nn.Linear(d, 1)
            self.corr_scale = float(corr_scale)
        def forward(self, x, expr_pred=None, numb_pred=None):
            h = self.body(x)
            alpha = torch.sigmoid(self.gate(h).squeeze(-1))
            corr = self.corr_scale * torch.tanh(self.corr(h).squeeze(-1))
            base = (1.0 - alpha) * expr_pred + alpha * numb_pred
            return base + corr

    class MLPResidual(nn.Module):
        def __init__(self, in_dim, hidden=16, depth=2, dropout=0.25, base_w=0.25, corr_scale=8.0):
            super().__init__()
            layers = []
            d = in_dim
            for _ in range(depth):
                layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)]
                d = hidden
            layers += [nn.Linear(d, 1)]
            self.net = nn.Sequential(*layers)
            self.base_w = float(base_w)
            self.corr_scale = float(corr_scale)

        def forward(self, x, expr_pred, numb_pred):
            base = (1.0 - self.base_w) * expr_pred + self.base_w * numb_pred
            corr = self.corr_scale * torch.tanh(self.net(x).squeeze(-1))
            return base + corr


def make_tensors(train, dev, test, cols: List[str], device):
    st = standardize_fit(train, cols)
    Xtr = torch.tensor(standardize_apply(train, st), dtype=torch.float32, device=device)
    Xdv = torch.tensor(standardize_apply(dev, st), dtype=torch.float32, device=device)
    Xte = torch.tensor(standardize_apply(test, st), dtype=torch.float32, device=device)
    return st, Xtr, Xdv, Xte



def _legacy_train_throwaway(model, Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args):
    model = model.to(Xtr.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = Xtr.shape[0]
    wait = 0
    best_score = 1e18
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=Xtr.device)
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(Xtr[idx], expr_tr[idx], numb_tr[idx])
            loss = F.mse_loss(pred, ytr[idx])
            if args.anchor_reg > 0:
                lo = torch.minimum(expr_tr[idx], numb_tr[idx]) - args.anchor_margin
                hi = torch.maximum(expr_tr[idx], numb_tr[idx]) + args.anchor_margin
                outside = torch.relu(lo - pred) + torch.relu(pred - hi)
                loss = loss + args.anchor_reg * torch.mean(outside ** 2)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                dv = torch.clamp(model(Xdv, expr_dv, numb_dv), 0, 24).detach().cpu().numpy()
            m = metrics_np(ydv.detach().cpu().numpy(), dv)
            sc = score_candidate(m, args)
            if sc < best_score - 1e-6:
                best_score = sc
                wait = 0
            else:
                wait += args.eval_every
        if epoch >= args.min_epochs and wait >= args.patience:
            break


def legacy_warmup_for_old_suite_target(Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args, in_dim: int):
    hidden = 16
    d = 0.10
    _legacy_train_throwaway(MLPDirect(in_dim, hidden=hidden, depth=args.mlp_depth, dropout=d), Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args)
    for bw in [0.25, 0.40, 0.50, 0.60, 0.75]:
        _legacy_train_throwaway(MLPResidual(in_dim, hidden=hidden, depth=args.mlp_depth, dropout=d, base_w=bw, corr_scale=args.corr_scale), Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args)
    _legacy_train_throwaway(MLPGateCorrection(in_dim, hidden=hidden, depth=args.mlp_depth, dropout=d, corr_scale=args.corr_scale), Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args)
    d = 0.25
    _legacy_train_throwaway(MLPDirect(in_dim, hidden=hidden, depth=args.mlp_depth, dropout=d), Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args)

def train_one_ngrc(train, dev, test, cols: List[str], cfg: dict, args) -> Candidate:
    if not TORCH_OK:
        raise RuntimeError("PyTorch is not available.")

    seed = int(cfg["seed"])
    set_seed(seed, deterministic=args.deterministic)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = str(device)
    st, Xtr, Xdv, Xte = make_tensors(train, dev, test, cols, device)

    ytr = torch.tensor(train["gt"].values.astype(np.float32), device=device)
    ydv = torch.tensor(dev["gt"].values.astype(np.float32), device=device)
    expr_tr = torch.tensor(train["expr_pred"].values.astype(np.float32), device=device)
    expr_dv = torch.tensor(dev["expr_pred"].values.astype(np.float32), device=device)
    expr_te = torch.tensor(test["expr_pred"].values.astype(np.float32), device=device)
    numb_tr = torch.tensor(train["numb_pred"].values.astype(np.float32), device=device)
    numb_dv = torch.tensor(dev["numb_pred"].values.astype(np.float32), device=device)
    numb_te = torch.tensor(test["numb_pred"].values.astype(np.float32), device=device)

    if getattr(args, "legacy_suite_rng_replay", False):
        print("[NGRC-REPLAY] legacy_suite_rng_replay=1: consuming old full-suite RNG context before target candidate", flush=True)
        legacy_warmup_for_old_suite_target(Xtr, Xdv, ytr, ydv, expr_tr, expr_dv, numb_tr, numb_dv, args, len(cols))

    model = MLPResidual(
        in_dim=len(cols),
        hidden=int(cfg["hidden"]),
        depth=int(args.mlp_depth),
        dropout=float(cfg["dropout"]),
        base_w=float(cfg["base_w"]),
        corr_scale=float(args.corr_scale),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    n = Xtr.shape[0]
    best = {"score": 1e18, "epoch": -1, "state": None, "dev_pred": None, "dev_metrics": None}
    wait = 0
    name = f"mlp_res_{cfg['feature_set']}_w{float(cfg['base_w']):.2f}_h{int(cfg['hidden'])}_d{float(cfg['dropout'])}_s{seed}"

    print(f"\n[NGRC-TRAIN] start {name} | n_features={len(cols)} | n_train={n} | epochs={args.epochs} | batch={args.batch_size}", flush=True)
    progress_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, args.batch_size):
            idx = perm[start:start + args.batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(Xtr[idx], expr_tr[idx], numb_tr[idx])
            loss = F.mse_loss(pred, ytr[idx])
            if args.anchor_reg > 0:
                lo = torch.minimum(expr_tr[idx], numb_tr[idx]) - args.anchor_margin
                hi = torch.maximum(expr_tr[idx], numb_tr[idx]) + args.anchor_margin
                outside = F.relu(lo - pred) + F.relu(pred - hi)
                loss = loss + args.anchor_reg * torch.mean(outside ** 2)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        # Eval every epoch by default.
        if epoch % args.eval_every == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                dv = torch.clamp(model(Xdv, expr_dv, numb_dv), 0, 24).detach().cpu().numpy()
            m = metrics_np(dev["gt"].values, dv)
            sc = score_candidate(m, args)
            improved = sc < best["score"] - 1e-7
            if improved:
                best = {
                    "score": float(sc),
                    "epoch": int(epoch),
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "dev_pred": dv,
                    "dev_metrics": m,
                }
                wait = 0
            else:
                wait += args.eval_every

            if args.progress_every > 0 and (epoch == 1 or epoch % args.progress_every == 0 or improved):
                print(
                    f"[NGRC-PROG] {name} | epoch={epoch:03d}/{args.epochs} | "
                    f"train_loss={np.mean(losses):.4f} | dev_rmse={m['rmse']:.4f} | "
                    f"dev_mae={m['mae']:.4f} | dev_ccc={m['ccc']:.4f} | dev_corr={m['corr']:.4f} | "
                    f"score={sc:.4f} | best_epoch={best['epoch']} | wait={wait}",
                    flush=True,
                )
            eta = (time.monotonic() - progress_start) / epoch * (args.epochs - epoch)
            print(f"APNET_PROGRESS Epoch {epoch}/{args.epochs} | train_loss={np.mean(losses):.4f} | "
                  f"lr={opt.param_groups[0]['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)
            if epoch >= args.min_epochs and wait >= args.patience:
                print(f"[NGRC-EARLY] {name} | epoch={epoch} | best_epoch={best['epoch']} | best_score={best['score']:.4f}", flush=True)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        trp = torch.clamp(model(Xtr, expr_tr, numb_tr), 0, 24).detach().cpu().numpy()
        dvp = torch.clamp(model(Xdv, expr_dv, numb_dv), 0, 24).detach().cpu().numpy()
        tep = torch.clamp(model(Xte, expr_te, numb_te), 0, 24).detach().cpu().numpy()

    trm, dvm, tem = metrics_np(train["gt"].values, trp), metrics_np(dev["gt"].values, dvp), metrics_np(test["gt"].values, tep)
    print(
        f"[NGRC-END] {name} | best_epoch={best['epoch']} | "
        f"train_rmse={trm['rmse']:.4f} | dev_rmse={dvm['rmse']:.4f} | test_rmse={tem['rmse']:.4f} | "
        f"test_ccc={tem['ccc']:.4f}",
        flush=True,
    )
    params = {
        **cfg,
        "n_features": len(cols),
        "cols_preview": cols[:40],
        "best_epoch": int(best["epoch"]),
        "best_dev_score": float(best["score"]),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "standardizer_mean_preview": st["mu"][:5],
        "standardizer_std_preview": st["sd"][:5],
    }
    frozen = {
        "format_version": 1,
        "model_kwargs": {"in_dim": len(cols), "hidden": int(cfg["hidden"]),
                         "depth": int(args.mlp_depth), "dropout": float(cfg["dropout"]),
                         "base_w": float(cfg["base_w"]), "corr_scale": float(args.corr_scale)},
        "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "standardizer": {"cols": list(st["cols"]), "mu": st["mu"].tolist(), "sd": st["sd"].tolist()},
        "input_options": {k: getattr(args, k) for k in
                          ("expr_pred_col", "numb_pred_col", "allow_residual_features")},
    }
    return Candidate(name=name, kind="ngrc_mlp_residual", train_pred=trp, dev_pred=dvp,
                     test_pred=tep, params=params, frozen=frozen)


def pred_df(df: pd.DataFrame, pred: np.ndarray) -> pd.DataFrame:
    out = df[["video_id", "gt", "bin", "expr_pred", "numb_pred", "mean_pred", "oracle_pred", "oracle_choose_numb"]].copy()
    out["pred"] = pred
    out["err"] = out["pred"] - out["gt"]
    out["abs_err"] = np.abs(out["err"])
    out["residual_over_mean"] = out["pred"] - out["mean_pred"]
    out["residual_over_base_expr075_numb025"] = out["pred"] - (0.75*out["expr_pred"] + 0.25*out["numb_pred"])
    return out


def bin_metrics(pp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BIN_ORDER:
        sub = pp[pp["bin"] == b]
        if len(sub) == 0:
            continue
        m = metrics_np(sub["gt"].values, sub["pred"].values)
        rows.append({
            "bin": b,
            "count": int(len(sub)),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "pred_mean": m["pred_mean"],
            "gt_mean": m["gt_mean"],
            "bias": m["bias"],
            "expr_pred_mean": float(sub["expr_pred"].mean()),
            "numb_pred_mean": float(sub["numb_pred"].mean()),
            "mean_pred_mean": float(sub["mean_pred"].mean()),
            "residual_over_mean_avg": float((sub["pred"] - sub["mean_pred"]).mean()),
            "oracle_rmse_diagnostic": metrics_np(sub["gt"].values, sub["oracle_pred"].values)["rmse"],
            "oracle_choose_numb_rate_diagnostic": float(sub["oracle_choose_numb"].mean()),
        })
    return pd.DataFrame(rows)


def candidate_rows(cands: List[Candidate], train, dev, test, args) -> pd.DataFrame:
    rows = []
    for c in cands:
        trm, dvm, tem = metrics_np(train["gt"], c.train_pred), metrics_np(dev["gt"], c.dev_pred), metrics_np(test["gt"], c.test_pred)
        row = {
            "name": c.name,
            "kind": c.kind,
            "dev_score": score_candidate(dvm, args),
            "train_rmse": trm["rmse"], "train_mae": trm["mae"], "train_ccc": trm["ccc"], "train_corr": trm["corr"],
            "dev_rmse": dvm["rmse"], "dev_mae": dvm["mae"], "dev_ccc": dvm["ccc"], "dev_corr": dvm["corr"], "dev_bias": dvm["bias"],
            "test_rmse": tem["rmse"], "test_mae": tem["mae"], "test_ccc": tem["ccc"], "test_corr": tem["corr"], "test_bias": tem["bias"],
            "test_pred_mean": tem["pred_mean"], "test_pred_std": tem["pred_std"],
            "params": json.dumps(to_jsonable(c.params), ensure_ascii=False),
        }
        for k, v in c.params.items():
            if k in ["cols_preview", "standardizer_mean_preview", "standardizer_std_preview"]:
                continue
            if isinstance(v, (str, int, float, bool, np.integer, np.floating)):
                row[f"param_{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["dev_score", "dev_rmse"], ascending=True).reset_index(drop=True)


def add_baseline_rows(train, dev, test, args) -> pd.DataFrame:
    rows = []
    baseline_preds = {
        "expr": (train["expr_pred"].values, dev["expr_pred"].values, test["expr_pred"].values),
        "numb": (train["numb_pred"].values, dev["numb_pred"].values, test["numb_pred"].values),
        "mean": (train["mean_pred"].values, dev["mean_pred"].values, test["mean_pred"].values),
    }
    for w in [0.0, 0.25, 0.40, 0.50, 0.60, 0.75, 1.0]:
        baseline_preds[f"blend_w{w:.2f}"] = (
            (1-w)*train["expr_pred"].values + w*train["numb_pred"].values,
            (1-w)*dev["expr_pred"].values + w*dev["numb_pred"].values,
            (1-w)*test["expr_pred"].values + w*test["numb_pred"].values,
        )
    for name, (trp, dvp, tep) in baseline_preds.items():
        dvm, tem = metrics_np(dev["gt"], dvp), metrics_np(test["gt"], tep)
        rows.append({
            "name": name,
            "kind": "baseline",
            "dev_score": score_candidate(dvm, args),
            "dev_rmse": dvm["rmse"],
            "dev_ccc": dvm["ccc"],
            "dev_corr": dvm["corr"],
            "test_rmse": tem["rmse"],
            "test_mae": tem["mae"],
            "test_ccc": tem["ccc"],
            "test_corr": tem["corr"],
        })
    return pd.DataFrame(rows).sort_values("dev_score")


def make_cfgs(args, feature_sets: Dict[str, List[str]]) -> List[dict]:
    if args.mode == "fixed":
        return [{
            "feature_set": args.feature_set,
            "base_w": args.base_w,
            "hidden": args.hidden,
            "dropout": args.dropout,
            "seed": args.seed,
        }]
    cfgs = []
    for fs in args.search_feature_sets:
        if fs not in feature_sets:
            continue
        for w in args.search_base_ws:
            for h in args.search_hidden_dims:
                for d in args.search_dropouts:
                    for s in args.search_seeds:
                        cfgs.append({"feature_set": fs, "base_w": w, "hidden": h, "dropout": d, "seed": s})
    return cfgs


def save_outputs(cands: List[Candidate], selected: Candidate, cand_df: pd.DataFrame, baseline_df: pd.DataFrame,
                 train, dev, test, feature_sets, logs, args):
    os.makedirs(args.save_dir, exist_ok=True)
    if selected.frozen is not None:
        with open(os.path.join(args.save_dir, "fusion_model.pth"), "xb") as stream:
            torch.save(selected.frozen, stream)
    cand_df.to_csv(os.path.join(args.save_dir, "ngrc_candidates.csv"), index=False)
    baseline_df.to_csv(os.path.join(args.save_dir, "baseline_candidates.csv"), index=False)
    with open(os.path.join(args.save_dir, "ngrc_feature_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable({"feature_sets": feature_sets}), f, indent=2, ensure_ascii=False)

    for split, df, pred in [("train", train, selected.train_pred), ("dev", dev, selected.dev_pred), ("test", test, selected.test_pred)]:
        pp = pred_df(df, pred)
        pp.to_csv(os.path.join(args.save_dir, f"selected_{split}_predictions.csv"), index=False)
        bin_metrics(pp).to_csv(os.path.join(args.save_dir, f"selected_{split}_bin_metrics.csv"), index=False)

    # In search mode, also save best-test diagnostic prediction. This is explicitly diagnostic.
    best_test_diag = None
    if args.save_best_test_diagnostic and len(cand_df) > 0:
        best_test_name = cand_df.sort_values("test_rmse").iloc[0]["name"]
        bt = next(c for c in cands if c.name == best_test_name)
        for split, df, pred in [("train", train, bt.train_pred), ("dev", dev, bt.dev_pred), ("test", test, bt.test_pred)]:
            pp = pred_df(df, pred)
            pp.to_csv(os.path.join(args.save_dir, f"best_test_diagnostic_{split}_predictions.csv"), index=False)
            bin_metrics(pp).to_csv(os.path.join(args.save_dir, f"best_test_diagnostic_{split}_bin_metrics.csv"), index=False)
        best_test_diag = {
            "name": bt.name,
            "kind": bt.kind,
            "train_metrics": metrics_np(train["gt"], bt.train_pred),
            "dev_metrics": metrics_np(dev["gt"], bt.dev_pred),
            "test_metrics": metrics_np(test["gt"], bt.test_pred),
            "params": bt.params,
        }

    test_pp = pred_df(test, selected.test_pred)
    dev_pp = pred_df(dev, selected.dev_pred)
    summary = {
        "method": "RouteA-v2 NGRC: Numbness-Guided Residual Calibration",
        "formula": "base=(1-w)*expr_pred+w*numb_pred; pred=base+MLP(numbness_features+score_context)",
        "mode": args.mode,
        "selection": "fixed mode uses a pre-defined NGRC candidate and dev-selected epoch; search mode uses dev_score to select candidate/epoch.",
        "selected_candidate": cand_df[cand_df["name"] == selected.name].iloc[0].to_dict(),
        "train_metrics": metrics_np(train["gt"], selected.train_pred),
        "dev_metrics": metrics_np(dev["gt"], selected.dev_pred),
        "test_metrics": metrics_np(test["gt"], selected.test_pred),
        "dev_bins": bin_metrics(dev_pp).to_dict(orient="records"),
        "test_bins": bin_metrics(test_pp).to_dict(orient="records"),
        "baselines": {
            "expr_test": metrics_np(test["gt"], test["expr_pred"]),
            "numb_test": metrics_np(test["gt"], test["numb_pred"]),
            "mean_test": metrics_np(test["gt"], test["mean_pred"]),
            "oracle_test_diagnostic_only": metrics_np(test["gt"], test["oracle_pred"]),
            "baseline_table_top_by_dev": baseline_df.head(20).to_dict(orient="records"),
        },
        "best_test_diagnostic_only": best_test_diag,
        "safety": {
            "test_used_for_training": False,
            "test_used_for_epoch_selection": False,
            "dev_used_for_epoch_selection": True,
            "mode_fixed_candidate_predefined": args.mode == "fixed",
            "mode_search_candidate_selected_by_dev": args.mode == "search",
            "allow_residual_features": bool(args.allow_residual_features),
        },
        "logs": logs,
        "args": vars(args),
    }
    with open(os.path.join(args.save_dir, "ngrc_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)

    pd.set_option("display.max_colwidth", 180)
    show_cols = ["name", "kind", "dev_score", "dev_rmse", "dev_ccc", "dev_corr", "test_rmse", "test_mae", "test_ccc", "test_corr", "test_pred_mean", "test_pred_std", "params"]
    print("\n================ NGRC Candidate Summary: top by dev_score ================")
    print(cand_df.head(args.print_top_k)[[c for c in show_cols if c in cand_df.columns]].to_string(index=False))
    print("\n================ NGRC Candidate Summary: top by test_rmse (diagnostic) ================")
    print(cand_df.sort_values("test_rmse").head(args.print_top_k)[[c for c in show_cols if c in cand_df.columns]].to_string(index=False))

    print("\n================ NGRC Selected TEST bins ================")
    cols = ["bin", "count", "rmse", "mae", "pred_mean", "gt_mean", "bias", "expr_pred_mean", "numb_pred_mean", "mean_pred_mean", "residual_over_mean_avg", "oracle_rmse_diagnostic"]
    print(bin_metrics(test_pp)[cols].to_string(index=False))

    print("\n================ NGRC Key Result ================")
    print(json.dumps({
        "selected": selected.name,
        "mode": args.mode,
        "dev_rmse": metrics_np(dev["gt"], selected.dev_pred)["rmse"],
        "test_rmse": metrics_np(test["gt"], selected.test_pred)["rmse"],
        "test_mae": metrics_np(test["gt"], selected.test_pred)["mae"],
        "test_ccc": metrics_np(test["gt"], selected.test_pred)["ccc"],
        "test_corr": metrics_np(test["gt"], selected.test_pred)["corr"],
        "save_dir": args.save_dir,
    }, indent=2, ensure_ascii=False))


def main(args):
    if not TORCH_OK:
        raise RuntimeError("PyTorch is required for NGRC.")
    t0 = time.time()
    set_seed(args.seed, deterministic=args.deterministic)
    os.makedirs(args.save_dir, exist_ok=True)

    train, log_tr = load_split(args, "train")
    dev, log_dv = load_split(args, "dev")
    test, log_te = load_split(args, "test")
    logs = {"train": log_tr, "dev": log_dv, "test": log_te}

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.device = str(device)
    if device.type == "cuda":
        print(f"[device] cuda | gpu={torch.cuda.get_device_name(device)} | batch_size={args.batch_size}", flush=True)
    else:
        print(f"[device] cpu | batch_size={args.batch_size}", flush=True)

    feature_sets = build_ngrc_feature_sets(train, dev, args)
    print("[features] NGRC sets:", {k: len(v) for k, v in feature_sets.items()}, flush=True)
    if args.feature_set not in feature_sets and args.mode == "fixed":
        raise RuntimeError(f"feature_set={args.feature_set} is unavailable. Available: {list(feature_sets.keys())}")

    cfgs = make_cfgs(args, feature_sets)
    if len(cfgs) == 0:
        raise RuntimeError("No NGRC configs generated. Check --feature_set or --search_feature_sets.")
    print(f"[configs] mode={args.mode} | n_configs={len(cfgs)}", flush=True)
    for i, cfg in enumerate(cfgs[:20], 1):
        print(f"  cfg[{i}]={cfg}", flush=True)
    if len(cfgs) > 20:
        print(f"  ... {len(cfgs)-20} more configs", flush=True)

    cands = []
    for cfg in cfgs:
        cols = feature_sets[cfg["feature_set"]]
        cands.append(train_one_ngrc(train, dev, test, cols, cfg, args))

    cand_df = candidate_rows(cands, train, dev, test, args)
    if args.mode == "fixed":
        selected_name = cands[0].name
    else:
        selected_name = cand_df.iloc[0]["name"]
    selected = next(c for c in cands if c.name == selected_name)
    baseline_df = add_baseline_rows(train, dev, test, args)
    save_outputs(cands, selected, cand_df, baseline_df, train, dev, test, feature_sets, logs, args)
    print(f"[done] elapsed_sec={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--mode", choices=["fixed", "search"], default="fixed")
    p.add_argument("--save_dir", default="routeA_ngrc_2017/seed2026_fixed")

    p.add_argument("--expr_dir", default="fusion_inputs_2017/expr_v17_safe")
    p.add_argument("--numb_dir", default="fusion_inputs_2017/numb_v5")
    p.add_argument("--expr_train", default="")
    p.add_argument("--expr_dev", default="")
    p.add_argument("--expr_test", default="")
    p.add_argument("--numb_train", default="")
    p.add_argument("--numb_dev", default="")
    p.add_argument("--numb_test", default="")
    p.add_argument("--expr_pred_col", default="")
    p.add_argument("--numb_pred_col", default="")
    p.add_argument("--expr_embed_dir", default="exports_expr_v17_embeddings_v27")
    p.add_argument("--numb_embed_dir", default="exports_numb_v5_embeddings_v27")

    p.add_argument("--allow_residual_features", action="store_true", default=True)
    p.add_argument("--disallow_residual_features", dest="allow_residual_features", action="store_false")
    p.add_argument("--drop_feature_regex", nargs="*", default=[], help="Ablation: drop candidate fusion columns matching these regex patterns after safe-column filtering.")
    p.add_argument("--min_feature_std", type=float, default=1e-8)
    p.add_argument("--max_features_per_set", type=int, default=260)

    # Fixed breakthrough candidate.
    p.add_argument("--feature_set", default="numb128_score", choices=["numb128_score", "score_numbproj", "numb64_score", "numb192_score"])
    p.add_argument("--base_w", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.25)

    # NGRC-only search.
    p.add_argument("--search_feature_sets", nargs="+", default=["numb128_score", "score_numbproj"])
    p.add_argument("--search_base_ws", type=float, nargs="+", default=[0.25, 0.40, 0.50])
    p.add_argument("--search_hidden_dims", type=int, nargs="+", default=[16, 32, 64])
    p.add_argument("--search_dropouts", type=float, nargs="+", default=[0.10, 0.25])
    p.add_argument("--search_seeds", type=int, nargs="+", default=[2026, 2027, 2028])

    # Dev selection score.
    p.add_argument("--select_ccc_w", type=float, default=0.20)
    p.add_argument("--select_corr_w", type=float, default=0.10)
    p.add_argument("--select_bias_w", type=float, default=0.02)

    # Optim/training.
    p.add_argument("--device", default="")
    p.add_argument("--deterministic", action="store_true", default=False)
    p.add_argument("--legacy_suite_rng_replay", action="store_true", help="Replay the old RouteA-final RNG context before the fixed NGRC target candidate.")
    p.add_argument("--mlp_depth", type=int, default=2)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--min_epochs", type=int, default=120)
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--corr_scale", type=float, default=8.0)
    p.add_argument("--anchor_reg", type=float, default=0.005)
    p.add_argument("--anchor_margin", type=float, default=5.0)

    p.add_argument("--save_best_test_diagnostic", action="store_true", default=True)
    p.add_argument("--print_top_k", type=int, default=30)
    args = p.parse_args()
    main(args)
