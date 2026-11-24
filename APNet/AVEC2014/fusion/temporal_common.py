#!/usr/bin/env python3
"""Shared utilities for P16 shallow-capacity experiments."""

from __future__ import annotations

import json
import math
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ("train_fit", "dev", "test")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_torch(path: Path, device="cpu"):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def metrics(y_true, y_pred):
    y = np.asarray(y_true, np.float64).reshape(-1)
    p = np.asarray(y_pred, np.float64).reshape(-1)
    if y.shape != p.shape or len(y) == 0:
        raise ValueError(f"invalid metric arrays: {y.shape}, {p.shape}")
    error = p - y
    ym, pm = y.mean(), p.mean()
    yv, pv = y.var(), p.var()
    cov = np.mean((y - ym) * (p - pm))
    den = yv + pv + (ym - pm) ** 2
    corr_den = math.sqrt(max(yv * pv, 0.0))
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "ccc": float(2 * cov / den) if den > 1e-12 else 0.0,
        "corr": float(cov / corr_den) if corr_den > 1e-12 else 0.0,
        "pred_mean": float(pm), "pred_std": float(p.std(ddof=1)),
        "gt_mean": float(ym), "gt_std": float(y.std(ddof=1)),
        "bias": float(error.mean()),
    }


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(convert(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_archive(path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    signature = f"{path.resolve()}::{path.stat().st_size}::{path.stat().st_mtime_ns}"
    marker = destination / ".complete"
    if marker.is_file() and marker.read_text(encoding="utf-8") == signature:
        return destination
    with zipfile.ZipFile(path) as archive:
        root = destination.resolve()
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"unsafe zip member: {member.filename}")
        archive.extractall(destination)
    marker.write_text(signature, encoding="utf-8")
    return destination


def locate(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    matches = [p for p in root.rglob(name) if p.name == name]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {name!r} below {root}, found {len(matches)}")
    return matches[0]


def validate_pair_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    required = {"pair_id", "gt"}
    if required - set(frame.columns):
        raise KeyError(f"{split}: missing {sorted(required - set(frame.columns))}")
    out = frame.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    if len(out) != 50 or out.pair_id.duplicated().any():
        raise AssertionError(f"{split}: expected 50 unique pairs, got {len(out)}")
    return out.sort_values("pair_id").reset_index(drop=True)


def validate_feature_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """P16 uses exactly one row per pair for every split."""
    required = {"pair_id", "gt", "aee_pred", "pre_pred"}
    if required - set(frame.columns):
        raise KeyError(f"{split}: missing {sorted(required - set(frame.columns))}")
    out = frame.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    if len(out) != 50 or out["pair_id"].duplicated().any():
        raise AssertionError(f"{split}: expected 50 unique pair rows, got {len(out)}")
    numeric = out.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError(f"{split}: non-finite feature values")
    return out.sort_values("pair_id").reset_index(drop=True)


def validate_fold_table(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Validate one row per pair/fold (50 OOF rows or 250 inference rows)."""
    required = {"pair_id", "gt", "inference_fold"}
    if required - set(frame.columns):
        raise KeyError(f"{split}: missing {sorted(required - set(frame.columns))}")
    out = frame.copy()
    out["pair_id"] = out["pair_id"].astype(str)
    out["inference_fold"] = out["inference_fold"].astype(int)
    duplicate = out.duplicated(["pair_id", "inference_fold"]).any()
    if duplicate:
        raise AssertionError(f"{split}: duplicate pair/fold rows")
    per_pair = out.groupby("pair_id")["inference_fold"].nunique()
    expected = 1 if split == "train_oof" else 5
    if len(per_pair) != 50 or not (per_pair == expected).all():
        raise AssertionError(
            f"{split}: expected 50 pairs x {expected} folds, got {len(out)} rows")
    if split != "train_oof":
        folds = sorted(out["inference_fold"].unique().tolist())
        if folds != [0, 1, 2, 3, 4]:
            raise AssertionError(f"{split}: expected inference folds 0..4, got {folds}")
    return out.sort_values(["pair_id", "inference_fold"]).reset_index(drop=True)


def assert_aligned(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if not np.array_equal(left.pair_id.astype(str), right.pair_id.astype(str)):
        raise AssertionError(f"{label}: pair IDs differ")
    if not np.allclose(left["gt"], right["gt"], atol=1e-6):
        raise AssertionError(f"{label}: labels differ")
