# dataset.py
# ARDNet Retardation State Branch dataset
# Input: OpenFace pose/gaze CSVs; AVEC2019 uses the pose-only fast dataset.
# Output: one video-level sample = overlapped behavioral tokens.

import os
import math
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


POSE_COLS = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]
GAZE_WORLD0 = ["x_0", "y_0", "z_0"]
GAZE_WORLD1 = ["x_1", "y_1", "z_1"]
GAZE_HEAD0 = ["x_h0", "y_h0", "z_h0"]
GAZE_HEAD1 = ["x_h1", "y_h1", "z_h1"]


@dataclass
class SanitizerConfig:
    target_hz: float = 10.0
    conf_th: float = 0.75
    median_kernel: int = 5
    smooth_kernel: int = 5
    ema_alpha: float = 0.65
    deadband_pose: float = 1e-3
    deadband_gaze: float = 1e-3
    robust_eps: float = 1e-6
    use_gaze: bool = True


@dataclass
class PatchConfig:
    # 300 raw frames at 30 fps ≈ 10 sec; stride 200 raw frames ≈ 6.67 sec.
    window_sec: float = 10.0
    stride_sec: float = 6.67
    min_tokens: int = 1


def _read_txt(txt_path: str) -> List[Tuple[str, float]]:
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 2:
                raise ValueError(f"Bad line in {txt_path}: {line}")
            rows.append((parts[0], float(parts[1])))
    return rows


def _candidate_paths_from_au(au_path: str, kind: str) -> List[str]:
    """
    kind in {"pose", "gaze"}.
    Robustly derive OpenFace pose/gaze csv path from an AU path.
    """
    d = os.path.dirname(au_path)
    b = os.path.basename(au_path)
    stem, ext = os.path.splitext(b)

    if kind == "pose":
        repls = [
            b.replace("_AUs.csv", "_pose.csv"),
            b.replace("_aus.csv", "_pose.csv"),
            b.replace("AUs.csv", "pose.csv"),
            b.replace("aus.csv", "pose.csv"),
            b.replace("_AU.csv", "_pose.csv"),
            stem + "_pose.csv",
            stem.replace("_AUs", "_pose") + ".csv",
            stem.replace("AUs", "pose") + ".csv",
        ]
    else:
        repls = [
            b.replace("_AUs.csv", "_gaze.csv"),
            b.replace("_aus.csv", "_gaze.csv"),
            b.replace("AUs.csv", "gaze.csv"),
            b.replace("aus.csv", "gaze.csv"),
            b.replace("_AU.csv", "_gaze.csv"),
            stem + "_gaze.csv",
            stem.replace("_AUs", "_gaze") + ".csv",
            stem.replace("AUs", "gaze") + ".csv",
        ]

    cands = [os.path.join(d, x) for x in repls]
    if kind == "pose":
        cands.extend([
            au_path.replace("/AUs/", "/pose/").replace("_AUs.csv", "_pose.csv"),
            au_path.replace("/au/", "/pose/").replace("_AUs.csv", "_pose.csv"),
            au_path.replace("/AU/", "/pose/").replace("_AUs.csv", "_pose.csv"),
        ])
    else:
        cands.extend([
            au_path.replace("/AUs/", "/gaze/").replace("_AUs.csv", "_gaze.csv"),
            au_path.replace("/au/", "/gaze/").replace("_AUs.csv", "_gaze.csv"),
            au_path.replace("/AU/", "/gaze/").replace("_AUs.csv", "_gaze.csv"),
        ])

    out = []
    for p in cands:
        if p not in out:
            out.append(p)
    return out


def _find_existing(cands: List[str]) -> Optional[str]:
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def _video_id_from_path(path: str) -> str:
    b = os.path.basename(path)
    for suf in ["_AUs.csv", "_aus.csv", "_AU.csv", "_pose.csv", "_gaze.csv", ".csv"]:
        if b.endswith(suf):
            return b[: -len(suf)]
    return os.path.splitext(b)[0]


def _rolling_median(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    df = pd.DataFrame(x)
    return df.rolling(k, center=True, min_periods=1).median().values.astype(np.float32)


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(k, dtype=np.float32) / float(k)
    out = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[1]):
        out[:, c] = np.convolve(xp[:, c], kernel, mode="valid")[: x.shape[0]]
    return out.astype(np.float32)


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return x.astype(np.float32)
    y = np.zeros_like(x, dtype=np.float32)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * y[i - 1] + (1.0 - alpha) * x[i]
    return y


def _deadband(x: np.ndarray, thr: float) -> np.ndarray:
    if thr <= 0 or len(x) <= 1:
        return x.astype(np.float32)
    y = x.copy().astype(np.float32)
    for i in range(1, len(y)):
        diff = y[i] - y[i - 1]
        small = np.abs(diff) < thr
        y[i, small] = y[i - 1, small]
    return y


def _robust_z(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    med = np.nanmedian(x, axis=0, keepdims=True)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    std = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(iqr > eps, iqr / 1.349, std)
    scale = np.where(scale > eps, scale, 1.0)
    return ((x - med) / scale).astype(np.float32)


def _interpolate_nan(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if len(x) == 0:
        return x
    df = pd.DataFrame(x)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.interpolate(method="linear", limit_direction="both", axis=0)
    df = df.ffill().bfill().fillna(0.0)
    return df.values.astype(np.float32)


def _resample_by_timestamp(values: np.ndarray, timestamps: np.ndarray, target_hz: float) -> np.ndarray:
    if len(values) <= 1:
        return values.astype(np.float32)
    t = timestamps.astype(np.float64)
    ok = np.isfinite(t)
    if ok.sum() < 2 or t.max() <= t.min():
        return values.astype(np.float32)

    values = values[ok]
    t = t[ok]
    uniq, idx = np.unique(t, return_index=True)
    t = uniq
    values = values[idx]
    if len(t) <= 1:
        return values.astype(np.float32)

    start, end = float(t[0]), float(t[-1])
    step = 1.0 / float(target_hz)
    grid = np.arange(start, end + 1e-8, step)
    if len(grid) < 2:
        return values.astype(np.float32)

    out = np.zeros((len(grid), values.shape[1]), dtype=np.float32)
    for c in range(values.shape[1]):
        out[:, c] = np.interp(grid, t, values[:, c]).astype(np.float32)
    return out


def _sanitize_array(arr: np.ndarray, timestamps: np.ndarray, cfg: SanitizerConfig, deadband_thr: float) -> np.ndarray:
    arr = _interpolate_nan(arr)
    arr = _resample_by_timestamp(arr, timestamps, cfg.target_hz)
    arr = _interpolate_nan(arr)
    arr = _rolling_median(arr, cfg.median_kernel)
    arr = _moving_average(arr, cfg.smooth_kernel)
    arr = _ema(arr, cfg.ema_alpha)
    arr = _deadband(arr, deadband_thr)
    arr = _robust_z(arr, cfg.robust_eps)
    return arr.astype(np.float32)


def _vector_to_angles(v: np.ndarray) -> np.ndarray:
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    norm = np.sqrt(x * x + y * y + z * z) + 1e-8
    yaw = np.arctan2(x, -z)
    pitch = np.arctan2(y, np.sqrt(x * x + z * z) + 1e-8)
    bias = norm
    return np.stack([yaw, pitch, bias], axis=1).astype(np.float32)


def _safe_stats(x: np.ndarray) -> List[float]:
    if x.size == 0:
        return [0.0] * 7
    return [
        float(np.mean(x)),
        float(np.std(x)),
        float(np.min(x)),
        float(np.max(x)),
        float(np.percentile(x, 10)),
        float(np.percentile(x, 50)),
        float(np.percentile(x, 90)),
    ]


def _longest_run_ratio(mask: np.ndarray) -> Tuple[float, float, float]:
    mask = np.asarray(mask).astype(bool)
    if len(mask) == 0:
        return 0.0, 0.0, 0.0
    runs = []
    cur = 0
    for v in mask:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        return 0.0, 0.0, 0.0
    return float(max(runs) / len(mask)), float(np.mean(runs) / len(mask)), float(len(runs) / len(mask))


def _entropy_from_states(states: np.ndarray, n_states: int) -> float:
    if len(states) == 0:
        return 0.0
    cnt = np.bincount(states.astype(np.int64), minlength=n_states).astype(np.float32)
    p = cnt / (cnt.sum() + 1e-8)
    p = p[p > 0]
    return float(-(p * np.log(p + 1e-8)).sum() / math.log(n_states + 1e-8))


def _dominant_ratio(states: np.ndarray, n_states: int) -> float:
    if len(states) == 0:
        return 0.0
    cnt = np.bincount(states.astype(np.int64), minlength=n_states).astype(np.float32)
    return float(cnt.max() / (cnt.sum() + 1e-8))


def _transition_rate(states: np.ndarray) -> float:
    if len(states) <= 1:
        return 0.0
    return float(np.mean(states[1:] != states[:-1]))


def _posture_states(rot: np.ndarray, thr: float = 0.55) -> np.ndarray:
    rx, ry, rz = rot[:, 0], rot[:, 1], rot[:, 2]
    states = np.zeros(len(rot), dtype=np.int64)
    states[rx > thr] = 1
    states[rx < -thr] = 2
    states[ry > thr] = 3
    states[ry < -thr] = 4
    states[np.abs(rz) > thr] = 5
    return states


def _spectral_features(sig: np.ndarray, target_hz: float) -> List[float]:
    sig = np.asarray(sig).reshape(-1).astype(np.float32)
    if len(sig) < 4 or np.allclose(sig.std(), 0):
        return [0.0, 0.0, 0.0, 0.0]
    sig = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / target_hz)
    if len(spec) <= 1:
        return [0.0, 0.0, 0.0, 0.0]
    spec[0] = 0.0
    total = spec.sum() + 1e-8
    low = spec[freqs <= 0.5].sum() / total
    mid = spec[(freqs > 0.5) & (freqs <= 2.0)].sum() / total
    high = spec[freqs > 2.0].sum() / total
    centroid = float((freqs * spec).sum() / total)
    return [float(low), float(mid), float(high), centroid]


def _diff(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros_like(x)
    return np.vstack([np.zeros((1, x.shape[1]), dtype=np.float32), np.diff(x, axis=0)]).astype(np.float32)


def _window_features(pose: np.ndarray, gaze_angles: Optional[np.ndarray], target_hz: float, use_gaze: bool) -> np.ndarray:
    T = len(pose)
    if T < 2:
        return np.zeros(1, dtype=np.float32)

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

    slow_rot = _moving_average(rot, max(3, int(round(target_hz * 2.0))))
    fast_rot = rot - slow_rot
    slow_trans = _moving_average(trans, max(3, int(round(target_hz * 2.0))))
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

    feats: List[float] = []

    for arr in [rot, trans]:
        feats.extend(np.mean(arr, axis=0).tolist())
        feats.extend(np.std(arr, axis=0).tolist())
        feats.extend((np.max(arr, axis=0) - np.min(arr, axis=0)).tolist())
        feats.extend(np.percentile(arr, 90, axis=0).tolist())
        feats.extend(np.percentile(arr, 10, axis=0).tolist())

    for sig in [rot_speed, trans_speed, rot_acc, trans_acc]:
        feats.extend(_safe_stats(sig))

    feats.extend([
        float(np.mean(still_mask)), long_still, mean_still, count_still,
        transition, dom_ratio, state_entropy,
        slow_rot_energy, fast_rot_energy, slow_rot_energy / (fast_rot_energy + 1e-6),
        slow_trans_energy, fast_trans_energy, slow_trans_energy / (fast_trans_energy + 1e-6),
    ])

    feats.extend(_spectral_features(rot_speed, target_hz))
    feats.extend(_spectral_features(trans_speed, target_hz))

    if use_gaze and gaze_angles is not None and len(gaze_angles) >= 2:
        G = gaze_angles
        dg = _diff(G) * target_hz
        world = G[:, 0:3]
        head = G[:, 3:6] if G.shape[1] >= 6 else G[:, 0:3]
        d_world = dg[:, 0:3]
        d_head = dg[:, 3:6] if dg.shape[1] >= 6 else dg[:, 0:3]
        world_speed = np.linalg.norm(d_world[:, 0:2], axis=1)
        head_speed = np.linalg.norm(d_head[:, 0:2], axis=1)
        gaze_thr = max(0.05, float(np.percentile(world_speed, 25)))
        fixation = world_speed <= gaze_thr
        fix_long, fix_mean, fix_count = _longest_run_ratio(fixation)
        scanpath = float(np.sum(world_speed) / max(1, len(world_speed)))
        decouple = np.linalg.norm(world[:, 0:2] - head[:, 0:2], axis=1)

        for arr in [world, head]:
            feats.extend(np.mean(arr, axis=0).tolist())
            feats.extend(np.std(arr, axis=0).tolist())
            feats.extend((np.max(arr, axis=0) - np.min(arr, axis=0)).tolist())

        for sig in [world_speed, head_speed, decouple]:
            feats.extend(_safe_stats(sig))

        feats.extend([
            float(np.mean(fixation)), fix_long, fix_mean, fix_count,
            scanpath,
            float(np.mean(np.abs(world[:, 1]))),
            float(np.mean(decouple)),
            float(np.std(decouple)),
        ])
        feats.extend(_spectral_features(world_speed, target_hz))

    return np.asarray(feats, dtype=np.float32)


class ARDRetardationDataset2017(Dataset):
    def __init__(self, txt_path: str, sanitizer_cfg: SanitizerConfig = SanitizerConfig(), patch_cfg: PatchConfig = PatchConfig(), cache_root: str = "cache_ard_retardation_2017", use_cache: bool = True, verbose: bool = True):
        self.txt_path = txt_path
        self.sanitizer_cfg = sanitizer_cfg
        self.patch_cfg = patch_cfg
        self.cache_root = cache_root
        self.use_cache = use_cache
        self.verbose = verbose
        os.makedirs(cache_root, exist_ok=True)

        self.rows = _read_txt(txt_path)
        self.samples: List[Dict] = []
        self.feature_dim: Optional[int] = None
        self.feature_mean: Optional[np.ndarray] = None
        self.feature_std: Optional[np.ndarray] = None

        if verbose:
            print(f"[ARD-Retardation] loading | use_gaze={sanitizer_cfg.use_gaze} | win={patch_cfg.window_sec}s | stride={patch_cfg.stride_sec}s | hz={sanitizer_cfg.target_hz} | median={sanitizer_cfg.median_kernel} | ema={sanitizer_cfg.ema_alpha}")

        for au_path, score in self.rows:
            pose_path = _find_existing(_candidate_paths_from_au(au_path, "pose"))
            gaze_path = _find_existing(_candidate_paths_from_au(au_path, "gaze"))
            if pose_path is None:
                try:
                    if os.path.exists(au_path) and all(x in pd.read_csv(au_path, nrows=0).columns for x in POSE_COLS):
                        pose_path = au_path
                except Exception:
                    pass
            if pose_path is None:
                if verbose:
                    print(f"[WARN] pose not found for {au_path}, skip.")
                continue
            if sanitizer_cfg.use_gaze and gaze_path is None and verbose:
                print(f"[WARN] gaze not found for {au_path}; continue with pose-only for this video.")

            sample = self._load_or_build(au_path, pose_path, gaze_path, score)
            if sample is None:
                continue
            self.samples.append(sample)
            self.feature_dim = sample["tokens"].shape[1]

        if len(self.samples) == 0:
            raise RuntimeError(f"No valid samples loaded from {txt_path}")

        if verbose:
            print(f"Loaded {len(self.samples)} videos. feature_dim={self.feature_dim}")

    def _cache_path(self, au_path: str, pose_path: str, gaze_path: Optional[str], score: float) -> str:
        key = {
            "au_path": os.path.abspath(au_path),
            "pose_path": os.path.abspath(pose_path),
            "gaze_path": os.path.abspath(gaze_path) if gaze_path else None,
            "score": score,
            "sanitizer": asdict(self.sanitizer_cfg),
            "patch": asdict(self.patch_cfg),
            "version": "ard_retardation_v1",
        }
        h = hashlib.md5(str(key).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_root, f"{h}.npz")

    def _load_or_build(self, au_path: str, pose_path: str, gaze_path: Optional[str], score: float) -> Optional[Dict]:
        cp = self._cache_path(au_path, pose_path, gaze_path, score)
        if self.use_cache and os.path.exists(cp):
            z = np.load(cp, allow_pickle=True)
            return {"tokens": z["tokens"].astype(np.float32), "score": float(z["score"]), "video_id": str(z["video_id"]), "pose_path": pose_path, "gaze_path": gaze_path or ""}
        sample = self._build_sample(au_path, pose_path, gaze_path, score)
        if sample is not None and self.use_cache:
            np.savez_compressed(cp, tokens=sample["tokens"], score=np.asarray(sample["score"], dtype=np.float32), video_id=np.asarray(sample["video_id"]))
        return sample

    def _read_pose(self, pose_path: str) -> Tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(pose_path)
        missing = [c for c in POSE_COLS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing pose columns {missing} in {pose_path}")
        timestamps = df["timestamp"].values.astype(np.float32) if "timestamp" in df.columns else np.arange(len(df), dtype=np.float32) / 30.0
        vals = df[POSE_COLS].values.astype(np.float32)
        if "success" in df.columns:
            vals[df["success"].values.astype(np.float32) <= 0.5] = np.nan
        if "confidence" in df.columns:
            vals[df["confidence"].values.astype(np.float32) < self.sanitizer_cfg.conf_th] = np.nan
        vals = _sanitize_array(vals, timestamps, self.sanitizer_cfg, self.sanitizer_cfg.deadband_pose)
        return vals, timestamps

    def _read_gaze_angles(self, gaze_path: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if gaze_path is None or not os.path.exists(gaze_path):
            return None
        df = pd.read_csv(gaze_path)
        required = GAZE_WORLD0 + GAZE_WORLD1
        if not all(c in df.columns for c in required):
            return None
        timestamps = df["timestamp"].values.astype(np.float32) if "timestamp" in df.columns else np.arange(len(df), dtype=np.float32) / 30.0
        w = 0.5 * (df[GAZE_WORLD0].values.astype(np.float32) + df[GAZE_WORLD1].values.astype(np.float32))
        world_ang = _vector_to_angles(w)
        if all(c in df.columns for c in GAZE_HEAD0 + GAZE_HEAD1):
            h = 0.5 * (df[GAZE_HEAD0].values.astype(np.float32) + df[GAZE_HEAD1].values.astype(np.float32))
            head_ang = _vector_to_angles(h)
        else:
            head_ang = world_ang.copy()
        vals = np.concatenate([world_ang, head_ang], axis=1).astype(np.float32)
        if "success" in df.columns:
            vals[df["success"].values.astype(np.float32) <= 0.5] = np.nan
        if "confidence" in df.columns:
            vals[df["confidence"].values.astype(np.float32) < self.sanitizer_cfg.conf_th] = np.nan
        vals = _sanitize_array(vals, timestamps, self.sanitizer_cfg, self.sanitizer_cfg.deadband_gaze)
        return vals, timestamps

    def _build_sample(self, au_path: str, pose_path: str, gaze_path: Optional[str], score: float) -> Optional[Dict]:
        try:
            pose, _ = self._read_pose(pose_path)
            gaze_tuple = self._read_gaze_angles(gaze_path) if self.sanitizer_cfg.use_gaze and gaze_path else None
            gaze = gaze_tuple[0] if gaze_tuple is not None else None
            if gaze is not None:
                L = min(len(pose), len(gaze))
                pose = pose[:L]
                gaze = gaze[:L]
            else:
                L = len(pose)
            win = max(4, int(round(self.patch_cfg.window_sec * self.sanitizer_cfg.target_hz)))
            stride = max(1, int(round(self.patch_cfg.stride_sec * self.sanitizer_cfg.target_hz)))
            tokens = []
            if L <= 0:
                return None
            if L < win:
                tokens.append(_window_features(pose, gaze, self.sanitizer_cfg.target_hz, self.sanitizer_cfg.use_gaze))
            else:
                for st in range(0, L - win + 1, stride):
                    ed = st + win
                    tokens.append(_window_features(pose[st:ed], gaze[st:ed] if gaze is not None else None, self.sanitizer_cfg.target_hz, self.sanitizer_cfg.use_gaze))
                st = max(0, L - win)
                if len(tokens) == 0 or st != (L - win) - ((L - win) % stride):
                    tokens.append(_window_features(pose[st:L], gaze[st:L] if gaze is not None else None, self.sanitizer_cfg.target_hz, self.sanitizer_cfg.use_gaze))
            tokens = np.stack(tokens, axis=0).astype(np.float32)
            tokens = np.nan_to_num(tokens, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            return {"tokens": tokens, "score": float(score), "video_id": _video_id_from_path(au_path), "pose_path": pose_path, "gaze_path": gaze_path or ""}
        except Exception as e:
            if self.verbose:
                print(f"[WARN] failed {au_path}: {repr(e)}")
            return None

    def set_feature_norm(self, mean: np.ndarray, std: np.ndarray):
        self.feature_mean = mean.astype(np.float32)
        self.feature_std = std.astype(np.float32)

    def compute_feature_norm(self) -> Tuple[np.ndarray, np.ndarray]:
        all_tokens = np.concatenate([s["tokens"] for s in self.samples], axis=0)
        mean = all_tokens.mean(axis=0).astype(np.float32)
        std = all_tokens.std(axis=0).astype(np.float32) + 1e-6
        return mean, std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        tokens = s["tokens"].copy()
        if self.feature_mean is not None and self.feature_std is not None:
            tokens = (tokens - self.feature_mean) / self.feature_std
        return {"tokens": torch.tensor(tokens, dtype=torch.float32), "score": torch.tensor(float(s["score"]), dtype=torch.float32), "video_id": s["video_id"]}


def collate_ard_retardation(batch: List[Dict]) -> Dict:
    max_len = max(item["tokens"].shape[0] for item in batch)
    feat_dim = batch[0]["tokens"].shape[1]
    B = len(batch)
    tokens = torch.zeros(B, max_len, feat_dim, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    scores = torch.zeros(B, dtype=torch.float32)
    video_ids = []
    for i, item in enumerate(batch):
        n = item["tokens"].shape[0]
        tokens[i, :n] = item["tokens"]
        mask[i, :n] = True
        scores[i] = item["score"]
        video_ids.append(item["video_id"])
    return {"tokens": tokens, "mask": mask, "score": scores, "video_ids": video_ids}
