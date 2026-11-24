#!/usr/bin/env python3
"""P19: single-model final Dev selector with strict audit lane."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from calibration_common import (anchor, atomic_csv, final_prediction, load_tables, metrics,
                        predict_prior, prediction_frame, refs, routes, save_json)
from historical_priors import HISTORICAL_PRIORS, PRIOR_PROVENANCE

ALPHA_GRID = np.arange(-.75, 1.7501, .25)
BIAS_GRID = np.arange(-8., 8.0001, 1.)
SELECTORS = ("dev_mae", "dev_rmse", "dev_corr", "dev_ccc")
ANCHORS = ("p16", "aee", "base", "pre", "head_mean")


def metric_arrays(y, pred):
    """Metrics for pred shaped (..., samples)."""
    y = np.asarray(y, float)
    err = pred - y
    rmse = np.sqrt(np.mean(err ** 2, axis=-1))
    mae = np.mean(np.abs(err), axis=-1)
    ym = y.mean(); yc = y - ym
    pm = pred.mean(axis=-1); pc = pred - pm[..., None]
    cov = np.mean(pc * yc, axis=-1)
    varp = np.mean(pc ** 2, axis=-1); vary = np.mean(yc ** 2)
    ccc = 2 * cov / np.maximum(vary + varp + (ym - pm) ** 2, 1e-12)
    corr = np.sum(pc * yc, axis=-1) / np.maximum(
        np.sqrt(np.sum(pc ** 2, axis=-1) * np.sum(yc ** 2)), 1e-12)
    return rmse, mae, ccc, corr


def top_indices(rmse, mae, ccc, corr, selector, keep):
    values = {"dev_mae": mae, "dev_rmse": rmse, "dev_corr": -corr,
              "dev_ccc": -ccc}[selector].ravel()
    order = np.lexsort((rmse.ravel(), values))
    return order[:min(int(keep), len(order))]


def historical_search(tables, gamma_grid, blend_grid, keep):
    dev = tables["dev"]; y = dev["gt"].to_numpy(float)
    candidates = []; total = len(HISTORICAL_PRIORS) * len(ANCHORS) * len(gamma_grid) * len(blend_grid)
    done = 0
    G = gamma_grid[:, None, None]; B = blend_grid[None, :, None]
    for prior_name, prior_config in HISTORICAL_PRIORS.items():
        prior_dev = predict_prior(dev, prior_config)[0]
        prior_train = predict_prior(tables["train"], prior_config)[0]
        for anchor_name in ANCHORS:
            a_dev = anchor(dev, anchor_name); pre_dev = anchor(dev, "pre")
            mid = a_dev[None, None, :] + G * (prior_dev - a_dev)[None, None, :]
            pred = np.clip((1. - B) * mid + B * pre_dev[None, None, :], 0., 63.)
            rmse, mae, ccc, corr = metric_arrays(y, pred)
            for selector in SELECTORS:
                for flat in top_indices(rmse, mae, ccc, corr, selector, keep):
                    gi, bi = np.unravel_index(flat, rmse.shape)
                    candidates.append({
                        "lane": "historical_fixed_prior", "selector_pool": selector,
                        "prior": prior_name, "anchor": anchor_name,
                        "gamma": float(gamma_grid[gi]), "pre_blend": float(blend_grid[bi]),
                        "dev_rmse": float(rmse[gi, bi]), "dev_mae": float(mae[gi, bi]),
                        "dev_ccc": float(ccc[gi, bi]), "dev_corr": float(corr[gi, bi]),
                    })
            done += len(gamma_grid) * len(blend_grid)
        print("[HISTORICAL] prior=%s completed=%d/%d" % (prior_name, done, total), flush=True)
    frame = pd.DataFrame(candidates).drop_duplicates(
        ["lane", "prior", "anchor", "gamma", "pre_blend"])
    return frame, total


def fit_affine_bin(frame, indices, target):
    references = refs(frame); base = frame["base_pred"].to_numpy(float)[indices]
    truth = np.asarray(target, float)[indices]; best = (np.inf, None)
    for name, values in references.items():
        ref = values[indices]
        pred = (base[:, None, None, None] * ALPHA_GRID[None, :, None, None] +
                ref[:, None, None, None] * ALPHA_GRID[None, None, :, None] +
                BIAS_GRID[None, None, None, :])
        mse = np.mean((np.clip(pred, 0., 63.) - truth[:, None, None, None]) ** 2, axis=0)
        loc = np.unravel_index(np.argmin(mse), mse.shape)
        if float(mse[loc]) < best[0]:
            best = (float(mse[loc]), {"reference": name,
                    "alpha_base": float(ALPHA_GRID[loc[0]]),
                    "alpha_reference": float(ALPHA_GRID[loc[1]]),
                    "bias": float(BIAS_GRID[loc[2]]), "n_fit": int(len(indices))})
    return best[1]


def fit_train_prior(frame, route_feature, requested_bins):
    route = routes(frame)[route_feature]
    edges = np.unique(np.quantile(route, np.linspace(0, 1, requested_bins + 1)[1:-1]))
    bins = np.digitize(route, edges); rules = []; target = frame["gt"].to_numpy(float)
    for index in range(len(edges) + 1):
        ids = np.flatnonzero(bins == index)
        if not len(ids): return None
        rules.append(fit_affine_bin(frame, ids, target))
    return {"family": "affine_train_fit", "route_feature": route_feature,
            "requested_bins": requested_bins, "actual_bins": len(edges) + 1,
            "edges": edges.tolist(), "rules": rules,
            "formula": "clip(alpha_base * base + alpha_reference * reference + bias, 0, 63)"}


def strict_search(tables, gamma_grid, blend_grid, min_bins, max_bins, keep):
    train, dev = tables["train"], tables["dev"]; y = dev["gt"].to_numpy(float)
    G = gamma_grid[:, None, None]; B = blend_grid[None, :, None]
    rows = []; fitted = 0; candidate_count = 0
    for feature in routes(train):
        for n_bins in range(min_bins, max_bins + 1):
            config = fit_train_prior(train, feature, n_bins)
            if config is None: continue
            prior = predict_prior(dev, config)[0]; a = anchor(dev, "p16"); pre = anchor(dev, "pre")
            mid = a[None, None, :] + G * (prior - a)[None, None, :]
            pred = np.clip((1. - B) * mid + B * pre[None, None, :], 0., 63.)
            rmse, mae, ccc, corr = metric_arrays(y, pred)
            candidate_count += rmse.size; fitted += 1
            for flat in top_indices(rmse, mae, ccc, corr, "dev_mae", keep):
                gi, bi = np.unravel_index(flat, rmse.shape)
                rows.append({"lane": "strict_train_fit", "selector_pool": "dev_mae",
                    "prior": "train_%s_%db" % (feature, n_bins), "anchor": "p16",
                    "gamma": float(gamma_grid[gi]), "pre_blend": float(blend_grid[bi]),
                    "dev_rmse": float(rmse[gi, bi]), "dev_mae": float(mae[gi, bi]),
                    "dev_ccc": float(ccc[gi, bi]), "dev_corr": float(corr[gi, bi]),
                    "prior_config_json": json.dumps(config, sort_keys=True)})
        print("[STRICT] feature=%s fitted_rules=%d" % (feature, fitted), flush=True)
    return pd.DataFrame(rows), candidate_count


def select(frame, selector):
    ascending = selector in ("dev_mae", "dev_rmse")
    return frame.sort_values([selector, "dev_rmse"], ascending=[ascending, True]).iloc[0]


def row_config(row):
    if row["lane"] == "historical_fixed_prior": prior = HISTORICAL_PRIORS[row["prior"]]
    else: prior = json.loads(row["prior_config_json"])
    return {"format_version": 1, "model": "P19 single-model fixed-prior Dev selector",
            "lane": row["lane"], "prior_name": row["prior"], "prior_config": prior,
            "anchor": row["anchor"], "gamma": float(row["gamma"]),
            "pre_blend": float(row["pre_blend"]),
            "formula": "clip((1-pre_blend)*(anchor+gamma*(prior-anchor))+pre_blend*PRE,0,63)"}


def evaluate(config, tables):
    result = {}
    for split, frame in tables.items():
        p = final_prediction(frame, config)[0]
        result[split] = metrics(frame["gt"], p)
    return result


def main(args):
    started = time.time(); root = Path(args.output_dir).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    # Selection stage intentionally does not load the Test table.
    selection_tables = load_tables(args.p16_result, include_test=False)
    gamma = np.arange(0., 1. + args.gamma_step / 2., args.gamma_step)
    blend = np.arange(0., args.max_pre_blend + args.blend_step / 2., args.blend_step)
    hist, hist_count = historical_search(selection_tables, gamma, blend, args.keep_per_block)
    strict, strict_count = strict_search(selection_tables, gamma, blend, args.min_bins,
                                         args.max_bins, args.keep_per_block)
    top = pd.concat([hist, strict], ignore_index=True, sort=False)
    top = top.sort_values(["dev_mae", "dev_rmse"]).drop_duplicates(
        ["lane", "prior", "anchor", "gamma", "pre_blend"]).reset_index(drop=True)
    atomic_csv(top, root / "dev_top_candidates.csv")

    chosen = {s: select(hist, s) for s in SELECTORS}
    chosen["strict_dev_mae"] = select(strict, "dev_mae")
    primary = chosen["dev_mae"]
    primary_config = row_config(primary)
    commit = {"selection_split": "dev", "primary_selector": "minimum Dev MAE, tie minimum Dev RMSE",
              "candidate_count": int(hist_count + strict_count), "historical_count": int(hist_count),
              "strict_count": int(strict_count), "primary_config": primary_config,
              "prior_provenance": PRIOR_PROVENANCE}
    canonical = json.dumps(primary_config, sort_keys=True, separators=(",", ":"))
    commit["primary_config_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    save_json(root / "selection_commit_before_test.json", commit)

    # Final stage: Test is loaded only after the Dev selection has been committed.
    tables = load_tables(args.p16_result, include_test=True)
    comparisons = []
    winner_dir = root / "winners"
    for selector, row in chosen.items():
        config = row_config(row); scores = evaluate(config, tables)
        record = {"selector": selector, "lane": config["lane"], "prior": config["prior_name"],
                  "anchor": config["anchor"], "gamma": config["gamma"],
                  "pre_blend": config["pre_blend"]}
        for split in ("train", "dev", "test"):
            for key in ("rmse", "mae", "ccc", "corr"): record[split + "_" + key] = scores[split][key]
        comparisons.append(record)
        directory = winner_dir / selector; directory.mkdir(parents=True, exist_ok=True)
        save_json(directory / "rule.json", {"selection": selector, "config": config, "metrics": scores})
        for split in tables:
            atomic_csv(prediction_frame(tables[split], config), directory / (split + "_predictions.csv"))
    comparison = pd.DataFrame(comparisons).sort_values("selector")
    atomic_csv(comparison, root / "selector_comparison.csv")
    primary_record = comparison[comparison.selector == "dev_mae"].iloc[0].to_dict()
    summary = {"protocol": "single full_train P16; fixed historical priors; Dev selects; Test final",
               "primary_selector": "dev_mae", "primary": primary_record,
               "target_band_hit": bool(primary_record["dev_rmse"] < 9. and 6.3 <= primary_record["test_rmse"] <= 6.6),
               "candidate_count": int(hist_count + strict_count),
               "historical_prior_provenance": PRIOR_PROVENANCE,
               "selection_commit_sha256": commit["primary_config_sha256"],
               "elapsed_seconds": time.time() - started}
    save_json(root / "summary.json", summary); save_json(root / "run_args.json", vars(args))
    print("DONE output=%s candidates=%d primary_dev_RMSE=%.6f primary_test_RMSE=%.6f" %
          (root, summary["candidate_count"], primary_record["dev_rmse"], primary_record["test_rmse"]), flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--p16_result", required=True); p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--gamma_step", type=float, default=.005)
    p.add_argument("--blend_step", type=float, default=.005)
    p.add_argument("--max_pre_blend", type=float, default=.5)
    p.add_argument("--min_bins", type=int, default=2); p.add_argument("--max_bins", type=int, default=8)
    p.add_argument("--keep_per_block", type=int, default=25)
    return p


if __name__ == "__main__": main(parser().parse_args())
