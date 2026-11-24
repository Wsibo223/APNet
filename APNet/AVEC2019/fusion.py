#!/usr/bin/env python3
"""Fit the historical AVEC2019 Ridge architecture on fresh Train exports; score separately."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RECIPE = json.loads((Path(__file__).resolve().parent / "configs" / "fusion.json").read_text())
FEATURES = tuple(RECIPE["feature_columns"])
BRANCH_PREFIX = {"aee": "expr", "pae": "numb"}


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)
        file.write("\n")


def participant_id(value):
    match = re.search(r"\d+", Path(str(value)).name)
    if match is None:
        raise ValueError("An exported participant ID contains no digits")
    return match.group()


def read_split(dirs, split):
    frames = []
    for branch, prefix in BRANCH_PREFIX.items():
        cols = [c.split("__", 1)[1] for c in FEATURES if c.startswith(prefix + "__")]
        table = pd.read_csv(dirs[branch] / f"{split}_predictions.csv",
                            usecols=["video_id", "gt", *cols], dtype={"video_id": str})
        if len(table) != RECIPE["n"][split] or table["video_id"].isna().any():
            raise ValueError(f"{branch} {split}: expected {RECIPE['n'][split]} participants, read {len(table)}")
        table["video_id"] = table["video_id"].map(participant_id)
        if table["video_id"].duplicated().any():
            raise ValueError(f"{branch} {split}: duplicate participant IDs")
        if not table["gt"].between(0, 24).all():
            raise ValueError(f"{branch} {split}: invalid PHQ-8 scores")
        frames.append(table.rename(columns={c: prefix + "__" + c for c in cols}))
    if set(frames[0]["video_id"]) != set(frames[1]["video_id"]):
        raise ValueError(f"{split}: AEE/PAE participant sets are different")
    data = frames[0].merge(frames[1], on="video_id", sort=False, validate="one_to_one", suffixes=("", "__pae"))
    if not np.allclose(data["gt"], data["gt__pae"], atol=1e-6, rtol=0):
        raise ValueError(f"{split}: AEE/PAE labels differ")
    return data.drop(columns="gt__pae")


def matrix(data):
    return data[list(FEATURES)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def metrics(data, pred):
    y = data["gt"].to_numpy(dtype=float)
    residual = pred - y
    vy, vp = np.var(y), np.var(pred)
    covariance = ((y-y.mean()) * (pred-pred.mean())).mean()
    return {"n": len(y), "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual**2))),
            "ccc": float(2*covariance / (vy + vp + residual.mean()**2 + 1e-8)),
            "corr": float(np.corrcoef(y, pred)[0, 1]) if min(y.std(),pred.std()) >= 1e-12 else 0.0,
            "bias": float(residual.mean())}


def save_table(path, data, pred):
    data[["video_id", "gt"]].assign(pred=pred).to_csv(path, index=False, mode="x")


def fit(train_dev_dirs, out):
    train, dev = (read_split(train_dev_dirs, s) for s in ("train", "dev"))
    seen = set(train["video_id"]) | set(dev["video_id"])
    if len(seen) != len(train) + len(dev):
        raise ValueError("Train and Dev share participant IDs")
    xtr, xdev = matrix(train), matrix(dev)
    if np.isnan(xtr).all(axis=0).any():
        raise ValueError("A required fusion feature is entirely missing on Train")
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=RECIPE["alpha"]))
    model.fit(xtr, train["gt"].to_numpy())
    raw_dev = model.predict(xdev)
    shift = float(RECIPE["dev_bias_target"] - (raw_dev - dev["gt"].to_numpy()).mean())
    imp, scaler, ridge = [step for _, step in model.steps]
    frozen = {"protocol": "fit_train_then_dev_calibrate", "alpha": RECIPE["alpha"],
              "feature_columns": list(FEATURES), "train_impute_mean": imp.statistics_.tolist(),
              "train_scale_mean": scaler.mean_.tolist(), "train_scale_std": scaler.scale_.tolist(),
              "ridge_weights": ridge.coef_.tolist(), "ridge_intercept": float(ridge.intercept_),
              "dev_calibration_target": RECIPE["dev_bias_target"], "dev_calibration_shift": shift}
    out.mkdir(exist_ok=False)
    model_path = out / "fusion_model.json"
    write_json(model_path, frozen)
    pred_train, pred_dev = model.predict(xtr) + shift, raw_dev + shift
    save_table(out / "train_predictions.csv", train, pred_train)
    save_table(out / "dev_predictions.csv", dev, pred_dev)
    return model_path, seen, {"train": metrics(train, pred_train), "dev": metrics(dev, pred_dev)}


def score_test(test_dirs, model_path, seen):
    frozen = json.loads(model_path.read_text())
    test = read_split(test_dirs, "test")
    if seen & set(test["video_id"]):
        raise ValueError("Test shares participant IDs with Train or Dev")
    if tuple(frozen["feature_columns"]) != FEATURES:
        raise ValueError("Frozen model has unexpected feature ordering")
    x = matrix(test)
    x = np.where(np.isnan(x), frozen["train_impute_mean"], x)
    x = (x - frozen["train_scale_mean"]) / frozen["train_scale_std"]
    pred = x @ np.asarray(frozen["ridge_weights"]) + frozen["ridge_intercept"] + frozen["dev_calibration_shift"]
    save_table(model_path.parent / "test_predictions.csv", test, pred)
    return metrics(test, pred)
