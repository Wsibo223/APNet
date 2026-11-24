#!/usr/bin/env python3
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

HEAD_NAMES = ("global", "stillness", "burst", "retardation", "topk", "severe")
HEAD_COLUMNS = tuple("head_%s_pred" % x for x in HEAD_NAMES)
SOURCE_FILES = {
    "train": "full_train/final_model/train_fit_predictions.csv",
    "dev": "full_train/final_model/dev_predictions.csv",
    "test": "full_train/final_model/test_predictions.csv",
}


def _default(x):
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=_default))


def atomic_csv(frame, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False); tmp.replace(path)


def _read_csv(source, member):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        p = source / member
        if not p.is_file():
            matches = [x for x in source.rglob(Path(member).name)
                       if x.as_posix().endswith(member)]
            if len(matches) != 1: raise FileNotFoundError(p)
            p = matches[0]
        return pd.read_csv(p)
    with zipfile.ZipFile(source) as z:
        matches = [x for x in z.namelist() if x.replace("\\", "/").endswith(member)]
        if len(matches) != 1: raise FileNotFoundError("expected one %s" % member)
        with z.open(matches[0]) as f: return pd.read_csv(f)


def load_tables(source, include_test=True):
    required = {"pair_id", "gt", "aee_pred", "pre_pred", "base_pred", "pred", *HEAD_COLUMNS}
    result = {}
    for split, member in SOURCE_FILES.items():
        if split == "test" and not include_test: continue
        f = _read_csv(source, member)
        missing = sorted(required - set(f.columns))
        if missing: raise AssertionError("%s missing %s" % (split, missing))
        if len(f) != 50 or f["pair_id"].astype(str).nunique() != 50:
            raise AssertionError("%s expected 50 unique pairs" % split)
        for c in required - {"pair_id"}: f[c] = pd.to_numeric(f[c], errors="raise")
        result[split] = f.sort_values("pair_id").reset_index(drop=True)
    return result


def heads(frame): return frame[list(HEAD_COLUMNS)].to_numpy(np.float64)


def refs(frame):
    h = heads(frame); ordered = np.sort(h, axis=1)
    out = {n: h[:, i] for i, n in enumerate(HEAD_NAMES)}
    out.update({"sorted_%d" % i: ordered[:, i] for i in range(6)})
    out.update({
        "aee": frame["aee_pred"].to_numpy(float),
        "pre": frame["pre_pred"].to_numpy(float),
        "p16": frame["pred"].to_numpy(float),
        "head_mean": h.mean(1), "head_median": np.median(h, axis=1),
        "head_min": h.min(1), "head_max": h.max(1),
    })
    return out


def routes(frame):
    h = heads(frame); ordered = np.sort(h, axis=1); gaps = np.diff(ordered, axis=1)
    a = frame["aee_pred"].to_numpy(float); p = frame["pre_pred"].to_numpy(float)
    b = frame["base_pred"].to_numpy(float)
    return {
        "head_min": ordered[:, 0], "head_q25": np.quantile(h, .25, axis=1),
        "head_mean": h.mean(1), "head_median": np.median(h, axis=1),
        "head_q75": np.quantile(h, .75, axis=1), "head_max": ordered[:, -1],
        "head_range": ordered[:, -1] - ordered[:, 0], "head_std": h.std(1),
        "largest_gap": gaps.max(1), "lower_gap": ordered[:, 2] - ordered[:, 0],
        "upper_gap": ordered[:, 5] - ordered[:, 3],
        "p16": frame["pred"].to_numpy(float), "aee": a, "pre": p, "base": b,
        "aee_pre_gap": a - p, "abs_aee_pre_gap": np.abs(a - p),
        "base_head_mean_gap": b - h.mean(1),
        "severe_prob": frame["severe_prob"].to_numpy(float),
    }


def predict_prior(frame, config):
    route = routes(frame)[config["route_feature"]]
    bins = np.digitize(route, np.asarray(config["edges"], float))
    r = refs(frame); base = frame["base_pred"].to_numpy(float)
    pred = np.zeros(len(frame), float); used = np.empty(len(frame), object)
    for i, rule in enumerate(config["rules"]):
        mask = bins == i
        if not np.any(mask): continue
        name = rule["reference"]
        pred[mask] = (rule["alpha_base"] * base[mask] +
                      rule["alpha_reference"] * r[name][mask] + rule["bias"])
        used[mask] = name
    return np.clip(pred, 0., 63.), route, bins, used


def anchor(frame, name):
    if name == "p16": return frame["pred"].to_numpy(float)
    if name == "aee": return frame["aee_pred"].to_numpy(float)
    if name == "base": return frame["base_pred"].to_numpy(float)
    if name == "pre": return frame["pre_pred"].to_numpy(float)
    if name == "head_mean": return heads(frame).mean(1)
    raise KeyError(name)


def final_prediction(frame, config):
    prior, route, bins, used = predict_prior(frame, config["prior_config"])
    a = anchor(frame, config["anchor"]); pre = frame["pre_pred"].to_numpy(float)
    gamma = float(config["gamma"]); blend = float(config["pre_blend"])
    mid = a + gamma * (prior - a)
    pred = np.clip((1. - blend) * mid + blend * pre, 0., 63.)
    return pred, prior, route, bins, used


def metrics(y, p):
    y = np.asarray(y, float).reshape(-1); p = np.asarray(p, float).reshape(-1)
    e = p - y; yc = y - y.mean(); pc = p - p.mean()
    den = np.var(y) + np.var(p) + (y.mean() - p.mean()) ** 2
    cden = np.sqrt(np.sum(yc ** 2) * np.sum(pc ** 2))
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(e ** 2))),
            "mae": float(np.mean(np.abs(e))),
            "ccc": float(2 * np.mean(yc * pc) / den) if den > 1e-12 else 0.,
            "corr": float(np.sum(yc * pc) / cden) if cden > 1e-12 else 0.,
            "pred_mean": float(p.mean()), "pred_std": float(p.std(ddof=1)),
            "gt_mean": float(y.mean()), "gt_std": float(y.std(ddof=1)),
            "bias": float(e.mean())}


def prediction_frame(frame, config):
    pred, prior, route, bins, used = final_prediction(frame, config)
    out = frame[["pair_id", "gt", "aee_pred", "pre_pred", "pred"]].copy()
    out = out.rename(columns={"pred": "p16_pred"})
    out["historical_prior_pred"] = prior; out["route_value"] = route
    out["route_bin"] = bins; out["route_reference"] = used
    out["p19_pred"] = pred; out["error"] = pred - out["gt"]
    out["abs_error"] = np.abs(out["error"])
    return out


def format_table(frame, columns=None, digits=4):
    f = frame.copy()
    if columns is not None: f = f[[c for c in columns if c in f]]
    n = f.select_dtypes(include=[np.number]).columns; f[n] = f[n].round(digits)
    def cell(x): return "" if pd.isna(x) else str(x).replace("|", "\\|")
    header = [cell(x) for x in f.columns]
    rows = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    rows += ["| " + " | ".join(cell(x) for x in row) + " |" for row in f.itertuples(index=False, name=None)]
    return "\n".join(rows)
