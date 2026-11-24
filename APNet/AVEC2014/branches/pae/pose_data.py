"""AVEC2014 pose-only psychomotor-retardation data pipeline.

This is a direct AVEC2014 adaptation of the final AVEC2017 PRE pipeline:
confidence filtering -> temporal sanitization -> 10 Hz resampling ->
overlapping 4 s / 2 s windows -> 71-D retardation descriptors.

The official AVEC2014 labels are pair-level.  Freeform and Northwind clips are
kept as separate samples, while ``pair_id`` is carried through every batch so
cross-validation and paired evaluation can never split the two views.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Data audit can run on a CPU utility host without PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
VIEWS = ("Freeform", "Northwind")


@dataclass(frozen=True)
class SanitizerConfig:
    target_hz: float = 10.0
    conf_th: float = 0.75
    median_kernel: int = 5
    smooth_kernel: int = 5
    ema_alpha: float = 0.65
    deadband_pose: float = 1e-3
    robust_eps: float = 1e-6


@dataclass(frozen=True)
class PatchConfig:
    window_sec: float = 4.0
    stride_sec: float = 2.0
    max_tokens: int = 512


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def read_labels(labels_csv: str | os.PathLike) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df = _clean_columns(df)
    required = {"split", "pair_id", "score"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise KeyError(f"Label file is missing columns: {missing}")
    df = df.loc[:, ["split", "pair_id", "score"]].copy()
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["pair_id"] = df["pair_id"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="raise").astype(float)
    if df.duplicated(["split", "pair_id"]).any():
        dup = df.loc[df.duplicated(["split", "pair_id"], keep=False)]
        raise ValueError(f"Duplicate split/pair_id labels:\n{dup.to_string(index=False)}")
    return df.sort_values(["split", "pair_id"]).reset_index(drop=True)


def discover_records(
    data_root: str | os.PathLike,
    labels_csv: str | os.PathLike,
    strict: bool = True,
) -> pd.DataFrame:
    """Resolve both scenario CSVs for every labeled pair."""
    root = Path(data_root).expanduser().resolve()
    labels = read_labels(labels_csv)
    rows: List[Dict] = []
    missing: List[str] = []
    for rec in labels.itertuples(index=False):
        split_dir = root / rec.split
        for view in VIEWS:
            expected = split_dir / view / f"{rec.pair_id}_{view}_video.csv"
            if not expected.exists():
                # Case-insensitive fallback for copied filesystems.
                candidates = list(split_dir.glob(f"*/{rec.pair_id}_*_video.csv"))
                matched = [p for p in candidates if p.parent.name.lower() == view.lower()]
                if len(matched) == 1:
                    expected = matched[0]
                else:
                    missing.append(str(expected))
                    continue
            rows.append({
                "split": rec.split,
                "pair_id": rec.pair_id,
                "score": float(rec.score),
                "view": view,
                "path": str(expected),
                "clip_id": f"{rec.pair_id}_{view}",
            })
    if strict and missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} expected AVEC2014 clip CSV files. First entries:\n{preview}"
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No AVEC2014 clips were discovered.")
    return out.sort_values(["split", "pair_id", "view"]).reset_index(drop=True)


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    return pd.DataFrame(x).rolling(k, center=True, min_periods=1).median().to_numpy(np.float32)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    if k % 2 == 0:
        k += 1
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(xp[:, c], kernel, mode="valid")[: len(x)]
    return out


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0 or len(x) == 0:
        return x.astype(np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def _deadband(x: np.ndarray, threshold: float) -> np.ndarray:
    if threshold <= 0 or len(x) <= 1:
        return x.astype(np.float32)
    y = x.copy().astype(np.float32)
    for i in range(1, len(y)):
        small = np.abs(y[i] - y[i - 1]) < threshold
        y[i, small] = y[i - 1, small]
    return y


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x).replace([np.inf, -np.inf], np.nan)
    return df.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy(np.float32)


def _resample(values: np.ndarray, timestamps: np.ndarray, target_hz: float) -> np.ndarray:
    if len(values) <= 1:
        return values.astype(np.float32)
    t = np.asarray(timestamps, dtype=np.float64)
    valid = np.isfinite(t)
    if valid.sum() < 2:
        return values.astype(np.float32)
    t, values = t[valid], values[valid]
    t, unique_idx = np.unique(t, return_index=True)
    values = values[unique_idx]
    if len(t) <= 1 or t[-1] <= t[0]:
        return values.astype(np.float32)
    grid = np.arange(float(t[0]), float(t[-1]) + 1e-8, 1.0 / target_hz)
    if len(grid) < 2:
        return values.astype(np.float32)
    out = np.zeros((len(grid), values.shape[1]), dtype=np.float32)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(grid, t, values[:, c]).astype(np.float32)
    return out


def _robust_z(x: np.ndarray, eps: float) -> np.ndarray:
    med = np.nanmedian(x, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    iqr_scale = (q75 - q25) / 1.349
    std = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(iqr_scale > eps, iqr_scale, std)
    scale = np.where(scale > eps, scale, 1.0)
    return ((x - med) / scale).astype(np.float32)


def sanitize_pose(values: np.ndarray, timestamps: np.ndarray, cfg: SanitizerConfig) -> np.ndarray:
    x = _interpolate_nan(values)
    x = _resample(x, timestamps, cfg.target_hz)
    x = _interpolate_nan(x)
    x = _rolling_median(x, cfg.median_kernel)
    x = _moving_average(x, cfg.smooth_kernel)
    x = _ema(x, cfg.ema_alpha)
    x = _deadband(x, cfg.deadband_pose)
    return _robust_z(x, cfg.robust_eps)


def _diff(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x, dtype=np.float32)
    return np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)]).astype(np.float32)


def _safe_stats(x: np.ndarray) -> List[float]:
    if x.size == 0:
        return [0.0] * 7
    return [
        float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x)),
        float(np.percentile(x, 10)), float(np.percentile(x, 50)), float(np.percentile(x, 90)),
    ]


def _run_stats(mask: np.ndarray) -> Tuple[float, float, float]:
    runs: List[int] = []
    current = 0
    for value in np.asarray(mask, dtype=bool):
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    if not runs:
        return 0.0, 0.0, 0.0
    length = max(1, len(mask))
    return max(runs) / length, float(np.mean(runs)) / length, len(runs) / length


def _posture_states(rotation: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    rx, ry, rz = rotation[:, 0], rotation[:, 1], rotation[:, 2]
    states = np.zeros(len(rotation), dtype=np.int64)
    states[rx > threshold] = 1
    states[rx < -threshold] = 2
    states[ry > threshold] = 3
    states[ry < -threshold] = 4
    states[np.abs(rz) > threshold] = 5
    return states


def _state_statistics(states: np.ndarray, n_states: int = 6) -> Tuple[float, float, float]:
    if len(states) <= 1:
        return 0.0, 1.0, 0.0
    transition = float(np.mean(states[1:] != states[:-1]))
    counts = np.bincount(states, minlength=n_states).astype(np.float32)
    prob = counts / (counts.sum() + 1e-8)
    dominant = float(prob.max())
    nz = prob[prob > 0]
    entropy = float(-(nz * np.log(nz + 1e-8)).sum() / math.log(n_states + 1e-8))
    return transition, dominant, entropy


def window_features_71(pose: np.ndarray, target_hz: float) -> np.ndarray:
    """Exact 71-D fast descriptor family used by the final 2017 PRE package."""
    if len(pose) < 2:
        return np.zeros(71, dtype=np.float32)
    translation, rotation = pose[:, :3], pose[:, 3:]
    d_translation = _diff(translation) * target_hz
    d_rotation = _diff(rotation) * target_hz
    dd_translation = _diff(d_translation) * target_hz
    dd_rotation = _diff(d_rotation) * target_hz
    trans_speed = np.linalg.norm(d_translation, axis=1)
    rot_speed = np.linalg.norm(d_rotation, axis=1)
    trans_acc = np.linalg.norm(dd_translation, axis=1)
    rot_acc = np.linalg.norm(dd_rotation, axis=1)

    kernel = max(3, int(round(target_hz * 1.5)))
    if kernel % 2 == 0:
        kernel += 1
    slow_rot = _moving_average(rotation, kernel)
    slow_trans = _moving_average(translation, kernel)
    fast_rot = rotation - slow_rot
    fast_trans = translation - slow_trans
    energies = [
        float(np.mean(slow_rot ** 2)), float(np.mean(fast_rot ** 2)),
        float(np.mean(slow_trans ** 2)), float(np.mean(fast_trans ** 2)),
    ]

    rot_threshold = max(0.05, float(np.percentile(rot_speed, 25)))
    trans_threshold = max(0.05, float(np.percentile(trans_speed, 25)))
    still = (rot_speed <= rot_threshold) & (trans_speed <= trans_threshold)
    longest_still, mean_still, count_still = _run_stats(still)
    transition, dominant, entropy = _state_statistics(_posture_states(rotation))

    features: List[float] = []
    for arr in (rotation, translation):
        features.extend(np.mean(arr, axis=0).tolist())
        features.extend(np.std(arr, axis=0).tolist())
        features.extend((np.max(arr, axis=0) - np.min(arr, axis=0)).tolist())
        features.extend(np.percentile(arr, 90, axis=0).tolist())
        features.extend(np.percentile(arr, 10, axis=0).tolist())
    for signal in (rot_speed, trans_speed, rot_acc, trans_acc):
        features.extend(_safe_stats(signal))
    slow_rot_e, fast_rot_e, slow_trans_e, fast_trans_e = energies
    features.extend([
        float(np.mean(still)), longest_still, mean_still, count_still,
        transition, dominant, entropy,
        slow_rot_e, fast_rot_e, slow_rot_e / (fast_rot_e + 1e-6),
        slow_trans_e, fast_trans_e, slow_trans_e / (fast_trans_e + 1e-6),
    ])
    out = np.nan_to_num(np.asarray(features, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if out.shape != (71,):
        raise AssertionError(f"Expected 71-D PRE descriptor, got {out.shape}")
    return out


def build_tokens(path: str, sanitizer: SanitizerConfig, patch: PatchConfig) -> Tuple[np.ndarray, Dict]:
    df = _clean_columns(pd.read_csv(path))
    missing = [c for c in POSE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"{path}: missing OpenFace pose columns {missing}")
    values = df[POSE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    timestamps = (
        pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(np.float32)
        if "timestamp" in df.columns else np.arange(len(df), dtype=np.float32) / 30.0
    )
    valid = np.ones(len(df), dtype=bool)
    if "success" in df.columns:
        valid &= pd.to_numeric(df["success"], errors="coerce").fillna(0).to_numpy() > 0.5
    if "confidence" in df.columns:
        valid &= pd.to_numeric(df["confidence"], errors="coerce").fillna(0).to_numpy() >= sanitizer.conf_th
    values[~valid] = np.nan
    pose = sanitize_pose(values, timestamps, sanitizer)
    if len(pose) == 0:
        raise ValueError(f"{path}: no frames remain after preprocessing")

    win = max(4, int(round(patch.window_sec * sanitizer.target_hz)))
    stride = max(1, int(round(patch.stride_sec * sanitizer.target_hz)))
    windows: List[np.ndarray] = []
    if len(pose) < win:
        windows.append(window_features_71(pose, sanitizer.target_hz))
    else:
        starts = list(range(0, len(pose) - win + 1, stride))
        if not starts or starts[-1] != len(pose) - win:
            starts.append(len(pose) - win)
        for start in starts:
            windows.append(window_features_71(pose[start:start + win], sanitizer.target_hz))
    tokens = np.stack(windows).astype(np.float32)
    tokens_before_cap = len(tokens)
    if patch.max_tokens > 0 and len(tokens) > patch.max_tokens:
        idx = np.linspace(0, len(tokens) - 1, patch.max_tokens).round().astype(np.int64)
        tokens = tokens[idx]
    duration = float(np.nanmax(timestamps) - np.nanmin(timestamps)) if len(timestamps) else 0.0
    quality = {
        "n_raw_frames": int(len(df)),
        "valid_frame_ratio": float(valid.mean()) if len(valid) else 0.0,
        "duration_sec": duration,
        "n_resampled_frames": int(len(pose)),
        "tokens_before_cap": int(tokens_before_cap),
        "n_tokens": int(len(tokens)),
    }
    return tokens, quality


class PRE2014Corpus:
    """In-memory metadata plus disk-cached, unnormalized token arrays."""

    def __init__(
        self,
        records: pd.DataFrame,
        cache_root: str | os.PathLike,
        sanitizer: SanitizerConfig = SanitizerConfig(),
        patch: PatchConfig = PatchConfig(),
        use_cache: bool = True,
        verbose: bool = True,
    ):
        self.records = records.reset_index(drop=True).copy()
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.sanitizer = sanitizer
        self.patch = patch
        self.use_cache = use_cache
        self.verbose = verbose
        self.samples: List[Dict] = []
        iterator: Iterable = self.records.itertuples(index=False)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="PRE token cache", ncols=100)
            except Exception:
                pass
        failures: List[str] = []
        for row in iterator:
            try:
                sample = self._load_or_build(row)
                self.samples.append(sample)
            except Exception as exc:
                failures.append(f"{row.path}: {exc!r}")
        if failures:
            raise RuntimeError("Failed to build PRE tokens:\n" + "\n".join(failures[:20]))
        if not self.samples:
            raise RuntimeError("PRE2014Corpus contains no samples")
        self.feature_dim = int(self.samples[0]["tokens"].shape[1])

    def _cache_path(self, row) -> Path:
        stat = os.stat(row.path)
        key = json.dumps({
            "path": os.path.abspath(row.path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sanitizer": asdict(self.sanitizer), "patch": asdict(self.patch),
            "version": "avec2014_pre_pose71_v1",
        }, sort_keys=True)
        return self.cache_root / f"{hashlib.sha256(key.encode()).hexdigest()}.npz"

    def _load_or_build(self, row) -> Dict:
        cache_path = self._cache_path(row)
        if self.use_cache and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            tokens = cached["tokens"].astype(np.float32)
            quality = json.loads(str(cached["quality_json"].item()))
        else:
            tokens, quality = build_tokens(row.path, self.sanitizer, self.patch)
            if self.use_cache:
                np.savez_compressed(cache_path, tokens=tokens, quality_json=np.asarray(json.dumps(quality)))
        return {
            "tokens": tokens, "score": float(row.score), "pair_id": str(row.pair_id),
            "view": str(row.view), "split": str(row.split), "clip_id": str(row.clip_id),
            "path": str(row.path), "quality": quality,
        }

    def indices_for_split(self, split: str) -> List[int]:
        split = split.lower()
        return [i for i, sample in enumerate(self.samples) if sample["split"].lower() == split]

    def quality_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {k: s[k] for k in ("split", "pair_id", "view", "clip_id", "path")} | s["quality"]
            for s in self.samples
        ])


class NormalizedSubset(Dataset):
    def __init__(self, corpus: PRE2014Corpus, indices: Sequence[int], mean: np.ndarray, std: np.ndarray):
        self.corpus = corpus
        self.indices = list(map(int, indices))
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict:
        if torch is None:
            raise RuntimeError("PyTorch is required for training datasets")
        sample = self.corpus.samples[self.indices[index]]
        tokens = (sample["tokens"] - self.mean) / self.std
        return {
            "tokens": torch.tensor(tokens, dtype=torch.float32),
            "score": torch.tensor(sample["score"], dtype=torch.float32),
            "pair_id": sample["pair_id"], "view": sample["view"],
            "split": sample["split"], "clip_id": sample["clip_id"],
        }


def compute_norm(corpus: PRE2014Corpus, indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    tokens = np.concatenate([corpus.samples[int(i)]["tokens"] for i in indices], axis=0)
    mean = tokens.mean(axis=0).astype(np.float32)
    std = tokens.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    return mean, std


def collate_pre(batch: List[Dict]) -> Dict:
    if torch is None:
        raise RuntimeError("PyTorch is required for batching")
    max_len = max(item["tokens"].shape[0] for item in batch)
    feat_dim = batch[0]["tokens"].shape[1]
    tokens = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    scores = torch.zeros(len(batch), dtype=torch.float32)
    metadata = {key: [] for key in ("pair_id", "view", "split", "clip_id")}
    for i, item in enumerate(batch):
        length = item["tokens"].shape[0]
        tokens[i, :length] = item["tokens"]
        mask[i, :length] = True
        scores[i] = item["score"]
        for key in metadata:
            metadata[key].append(item[key])
    return {"tokens": tokens, "mask": mask, "score": scores, **metadata}
