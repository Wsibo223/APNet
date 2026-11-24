
# dataset_fast.py
# Fast dataset for ARD-Retardation 4s experiments.
#
# Why this file exists:
# The original ARD dataset recomputed relatively expensive window features
# for every overlapped 4s patch and printed no progress inside the loading loop,
# so it looked "stuck". This version:
#   1) prints per-video progress with tqdm,
#   2) uses a lighter fast window descriptor,
#   3) optionally caps tokens per video by uniform sampling,
#   4) uses a separate cache version to avoid stale cache confusion.

import os
import hashlib
from dataclasses import asdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

from .dataset import (
    SanitizerConfig,
    PatchConfig,
    POSE_COLS,
    _read_txt,
    _candidate_paths_from_au,
    _find_existing,
    _video_id_from_path,
    _sanitize_array,
    _longest_run_ratio,
    _transition_rate,
    _dominant_ratio,
    _entropy_from_states,
    _posture_states,
    _diff,
    _safe_stats,
    _moving_average,
    collate_ard_retardation,
)


def _fast_window_features(pose: np.ndarray, target_hz: float) -> np.ndarray:
    """
    Faster pose-only retardation descriptor.
    pose: [T, 6], already sanitized and robust-normalized.
    """
    if len(pose) < 2:
        return np.zeros(64, dtype=np.float32)

    trans = pose[:, 0:3]
    rot = pose[:, 3:6]

    d_trans = _diff(trans) * target_hz
    d_rot = _diff(rot) * target_hz
    dd_trans = _diff(d_trans) * target_hz
    dd_rot = _diff(d_rot) * target_hz

    trans_speed = np.linalg.norm(d_trans, axis=1)
    rot_speed = np.linalg.norm(d_rot, axis=1)
    trans_acc = np.linalg.norm(dd_trans, axis=1)
    rot_acc = np.linalg.norm(dd_rot, axis=1)

    # Slow-fast decomposition, but no FFT in fast dataset.
    k = max(3, int(round(target_hz * 1.5)))
    if k % 2 == 0:
        k += 1
    slow_rot = _moving_average(rot, k)
    fast_rot = rot - slow_rot
    slow_trans = _moving_average(trans, k)
    fast_trans = trans - slow_trans

    slow_rot_energy = float(np.mean(slow_rot ** 2))
    fast_rot_energy = float(np.mean(fast_rot ** 2))
    slow_trans_energy = float(np.mean(slow_trans ** 2))
    fast_trans_energy = float(np.mean(fast_trans ** 2))

    rot_thr = max(0.05, float(np.percentile(rot_speed, 25)))
    trans_thr = max(0.05, float(np.percentile(trans_speed, 25)))
    still_mask = (rot_speed <= rot_thr) & (trans_speed <= trans_thr)
    long_still, mean_still, count_still = _longest_run_ratio(still_mask)

    states = _posture_states(rot)
    transition = _transition_rate(states)
    dom_ratio = _dominant_ratio(states, 6)
    state_entropy = _entropy_from_states(states, 6)

    feats = []

    # Pose distribution.
    for arr in [rot, trans]:
        feats.extend(np.mean(arr, axis=0).tolist())
        feats.extend(np.std(arr, axis=0).tolist())
        feats.extend((np.max(arr, axis=0) - np.min(arr, axis=0)).tolist())
        feats.extend(np.percentile(arr, 90, axis=0).tolist())
        feats.extend(np.percentile(arr, 10, axis=0).tolist())

    # Dynamics.
    for sig in [rot_speed, trans_speed, rot_acc, trans_acc]:
        feats.extend(_safe_stats(sig))

    # Retardation state.
    feats.extend([
        float(np.mean(still_mask)),
        long_still,
        mean_still,
        count_still,
        transition,
        dom_ratio,
        state_entropy,
        slow_rot_energy,
        fast_rot_energy,
        slow_rot_energy / (fast_rot_energy + 1e-6),
        slow_trans_energy,
        fast_trans_energy,
        slow_trans_energy / (fast_trans_energy + 1e-6),
    ])

    return np.nan_to_num(np.asarray(feats, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


class ARDRetardationDataset2017Fast(Dataset):
    def __init__(
        self,
        txt_path: str,
        sanitizer_cfg: SanitizerConfig = SanitizerConfig(),
        patch_cfg: PatchConfig = PatchConfig(),
        cache_root: str = "cache_ard_retardation_2017_v2_fast",
        use_cache: bool = True,
        max_tokens: int = 256,
        verbose: bool = True,
    ):
        self.txt_path = txt_path
        self.sanitizer_cfg = sanitizer_cfg
        self.patch_cfg = patch_cfg
        self.cache_root = cache_root
        self.use_cache = use_cache
        self.max_tokens = int(max_tokens) if max_tokens is not None else 0
        self.verbose = verbose
        self.feature_mean = None
        self.feature_std = None
        self.samples: List[Dict] = []
        self.feature_dim = None

        os.makedirs(cache_root, exist_ok=True)
        rows = _read_txt(txt_path)

        if verbose:
            print(
                f"[ARD-Retardation-FAST] loading | win={patch_cfg.window_sec}s "
                f"| stride={patch_cfg.stride_sec}s | hz={sanitizer_cfg.target_hz} "
                f"| median={sanitizer_cfg.median_kernel} | ema={sanitizer_cfg.ema_alpha} "
                f"| max_tokens={self.max_tokens}",
                flush=True,
            )

        it = rows
        if verbose and tqdm is not None:
            it = tqdm(rows, desc="building/cache videos", ncols=100)

        skipped = 0
        for au_path, score in it:
            pose_path = _find_existing(_candidate_paths_from_au(au_path, "pose"))
            if pose_path is None:
                skipped += 1
                if verbose:
                    print(f"[WARN] pose not found for {au_path}", flush=True)
                continue

            sample = self._load_or_build(au_path, pose_path, score)
            if sample is None:
                skipped += 1
                continue
            self.samples.append(sample)
            self.feature_dim = sample["tokens"].shape[1]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples loaded from {txt_path}")

        if verbose:
            lens = np.asarray([s["tokens"].shape[0] for s in self.samples])
            print(
                f"Loaded {len(self.samples)} videos. skipped={skipped}. feature_dim={self.feature_dim}. "
                f"tokens: min={lens.min()}, p50={np.percentile(lens,50):.1f}, "
                f"p90={np.percentile(lens,90):.1f}, max={lens.max()}, mean={lens.mean():.1f}",
                flush=True,
            )

    def _cache_path(self, au_path: str, pose_path: str, score: float) -> str:
        key = {
            "au_path": os.path.abspath(au_path),
            "pose_path": os.path.abspath(pose_path),
            "score": score,
            "sanitizer": asdict(self.sanitizer_cfg),
            "patch": asdict(self.patch_cfg),
            "max_tokens": self.max_tokens,
            "version": "ard_retardation_v2_fast_pose_only_202604",
        }
        h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_root, f"{h}.npz")

    def _load_or_build(self, au_path: str, pose_path: str, score: float):
        cp = self._cache_path(au_path, pose_path, score)
        if self.use_cache and os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            return {
                "tokens": z["tokens"].astype(np.float32),
                "score": float(z["score"]),
                "video_id": str(z["video_id"]),
            }

        try:
            import pandas as pd
            df = pd.read_csv(pose_path)
            missing = [c for c in POSE_COLS if c not in df.columns]
            if missing:
                raise KeyError(f"Missing pose columns {missing} in {pose_path}")

            timestamps = df["timestamp"].values.astype(np.float32) if "timestamp" in df.columns else np.arange(len(df), dtype=np.float32) / 30.0
            vals = df[POSE_COLS].values.astype(np.float32)

            if "success" in df.columns:
                ok = df["success"].values.astype(np.float32) > 0.5
                vals[~ok] = np.nan
            if "confidence" in df.columns:
                ok = df["confidence"].values.astype(np.float32) >= self.sanitizer_cfg.conf_th
                vals[~ok] = np.nan

            pose = _sanitize_array(vals, timestamps, self.sanitizer_cfg, self.sanitizer_cfg.deadband_pose)

            L = len(pose)
            win = max(4, int(round(self.patch_cfg.window_sec * self.sanitizer_cfg.target_hz)))
            stride = max(1, int(round(self.patch_cfg.stride_sec * self.sanitizer_cfg.target_hz)))

            feats = []
            if L < win:
                feats.append(_fast_window_features(pose, self.sanitizer_cfg.target_hz))
            else:
                for st in range(0, L - win + 1, stride):
                    feats.append(_fast_window_features(pose[st:st + win], self.sanitizer_cfg.target_hz))
                if len(feats) == 0 or (L - win) % stride != 0:
                    feats.append(_fast_window_features(pose[-win:], self.sanitizer_cfg.target_hz))

            tokens = np.stack(feats, axis=0).astype(np.float32)

            # Token cap for very long videos. This prevents slow training and huge padded batches.
            if self.max_tokens and tokens.shape[0] > self.max_tokens:
                idx = np.linspace(0, tokens.shape[0] - 1, self.max_tokens).round().astype(np.int64)
                tokens = tokens[idx]

            sample = {
                "tokens": tokens,
                "score": float(score),
                "video_id": _video_id_from_path(au_path),
            }

            if self.use_cache:
                np.savez_compressed(
                    cp,
                    tokens=tokens,
                    score=np.asarray(score, dtype=np.float32),
                    video_id=np.asarray(sample["video_id"]),
                )

            return sample

        except Exception as e:
            if self.verbose:
                print(f"[WARN] failed {au_path}: {repr(e)}", flush=True)
            return None

    def compute_feature_norm(self):
        all_tokens = np.concatenate([s["tokens"] for s in self.samples], axis=0)
        mean = all_tokens.mean(axis=0).astype(np.float32)
        std = all_tokens.std(axis=0).astype(np.float32) + 1e-6
        return mean, std

    def set_feature_norm(self, mean, std):
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x = s["tokens"].copy()
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / self.feature_std
        return {
            "tokens": torch.tensor(x, dtype=torch.float32),
            "score": torch.tensor(float(s["score"]), dtype=torch.float32),
            "video_id": s["video_id"],
        }
