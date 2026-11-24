#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AVEC2014 AEE V4: pair-level outer-OOF ensemble with anchor protection.

This file intentionally imports the verified V3.1 feature/prototype encoder and
changes the part that failed empirically: residual training and calibration.

Leakage boundary
----------------
* outer folds are grouped by pair_id, so Freeform/Northwind never separate;
* every OOF candidate is produced by a network that did not train on that pair;
* the safe calibrator is selected by a second cross-fit over OOF predictions;
* dev/test labels never fit the anchor, network, shrinkage or reliability gate;
* the optional test-oracle JSON is diagnostic only and never changes predictions.
"""

import argparse
import copy
import json
import math
import time
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import aee_core as core


def jdump(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(core.to_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_split(meta, args, name):
    arrays, paths = [], []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc=f"load {name}", ncols=100):
        path = core.find_au_file(row["raw_path"], args.data_root)
        paths.append(path)
        arrays.append(core.load_au(
            path, args.au_columns_list, not args.no_success_filter,
            args.allow_missing_au, args.confidence_threshold,
            args.resample_hz, args.max_interp_gap,
        ))
    return arrays, paths


def select_meta(meta, pair_ids):
    pair_ids = set(map(str, pair_ids))
    return meta[meta.pair_id.astype(str).isin(pair_ids)].copy().reset_index(drop=True)


def select_items(meta_all, arrays_all, paths_all, pair_ids):
    pair_ids = set(map(str, pair_ids))
    ids = [i for i, p in enumerate(meta_all.pair_id.astype(str)) if p in pair_ids]
    return (meta_all.iloc[ids].copy().reset_index(drop=True),
            [arrays_all[i] for i in ids], [paths_all[i] for i in ids])


def check_complete_pairs(meta, name):
    sizes = meta.groupby("pair_id").size()
    tasks = meta.groupby("pair_id").task.apply(set)
    labels = meta.groupby("pair_id").score.nunique()
    if not (sizes == 2).all():
        raise ValueError(f"{name}: incomplete pairs {sizes[sizes != 2].to_dict()}")
    bad = tasks[tasks.apply(lambda x: x != {"Freeform", "Northwind"})]
    if len(bad):
        raise ValueError(f"{name}: invalid task sets {bad.to_dict()}")
    if not (labels == 1).all():
        raise ValueError(f"{name}: different labels inside a pair")


def make_stratified_pair_folds(meta, n_folds, seed):
    pair = meta.groupby("pair_id", as_index=False).agg(score=("score", "first"))
    pair["bin_id"] = pair.score.map(core.bin_id)
    if len(pair) < n_folds:
        raise ValueError(f"Only {len(pair)} pairs for {n_folds} folds")
    rng = np.random.default_rng(seed)
    assignments = {}
    load = np.zeros(n_folds, dtype=int)
    for bin_id in range(4):
        ids = pair.loc[pair.bin_id == bin_id, "pair_id"].astype(str).to_numpy()
        rng.shuffle(ids)
        for pair_id in ids:
            candidates = np.flatnonzero(load == load.min())
            fold = int(rng.choice(candidates))
            assignments[pair_id] = fold
            load[fold] += 1
    table = pair.copy()
    table["fold"] = table.pair_id.astype(str).map(assignments).astype(int)
    return assignments, table.sort_values(["fold", "pair_id"])


def pair_groups(pair_ids):
    out = []
    for pair_id in dict.fromkeys(map(str, pair_ids)):
        ids = [i for i, p in enumerate(pair_ids) if str(p) == pair_id]
        if len(ids) != 2:
            raise ValueError(f"V4 shared batch requires two clips for {pair_id}, got {ids}")
        out.append(ids)
    return out


def v4_loss(out, y, anchor, model, args, pair_ids):
    """Pair-first loss.  The collapsed V3 learned gate is not used here."""
    residual_raw = torch.clamp(out["residual_raw"], -args.residual_max, args.residual_max)
    clip_pred = torch.clamp(anchor + residual_raw, 0.0, args.score_max)
    groups = pair_groups(pair_ids)
    pp = torch.stack([clip_pred[g].mean() for g in groups])
    yy = torch.stack([y[g].mean() for g in groups])
    aa = torch.stack([anchor[g].mean() for g in groups])
    dd = torch.stack([residual_raw[g].mean() for g in groups])

    weights = torch.ones_like(yy)
    weights = torch.where((yy >= args.mild_cut) & (yy < args.moderate_cut), weights * 1.08, weights)
    weights = torch.where(yy >= args.moderate_cut, weights * 1.18, weights)
    weights = weights / weights.mean().clamp_min(1e-6)
    pair_huber = (F.smooth_l1_loss(pp, yy, beta=args.huber_beta, reduction="none") * weights).mean()
    pair_rmse = torch.sqrt((((pp - yy) ** 2) * weights).mean() + 1e-6)
    clip_huber = F.smooth_l1_loss(clip_pred, y, beta=args.huber_beta)
    target_delta = torch.clamp(yy - aa, -args.residual_max, args.residual_max)
    residual_loss = F.smooth_l1_loss(dd, target_delta, beta=args.residual_beta)
    consistency = torch.stack([(clip_pred[g[0]] - clip_pred[g[1]]).abs() for g in groups]).mean()

    # Keep the 2017 AEE lexicon idea, but make it auxiliary to the pair objective.
    y_expr = torch.clamp(y / args.score_max, 0, 1)
    target = core.ordinal_target(y_expr, model.proto_scores, args.ordinal_sigma)
    mh = out["mean_hist"].clamp_min(1e-8)
    ah = out["attn_hist"].clamp_min(1e-8)
    kl_mean = (target * (torch.log(target.clamp_min(1e-8)) - torch.log(mh))).sum(1).mean()
    kl_attn = (target * (torch.log(target.clamp_min(1e-8)) - torch.log(ah))).sum(1).mean()
    proto_score = (0.25 * F.smooth_l1_loss(out["proto_score"], y_expr, beta=.05)
                   + 0.75 * F.smooth_l1_loss(out["proto_score_attn"], y_expr, beta=.05))
    proto_div = core.proto_div_loss(model.enc.proto)
    attn_uniform = torch.relu(out["attn_entropy"] - args.attn_entropy_target).pow(2).mean()
    low_bce = F.binary_cross_entropy(out["p_low"], (y < args.minimal_cut).float())
    severe = yy >= args.moderate_cut
    severe_under = (torch.relu(yy[severe] - pp[severe] - 4.0).pow(2).mean()
                    if severe.any() else pp.new_tensor(0.0))

    loss = (0.54 * pair_huber + 0.18 * pair_rmse + 0.10 * clip_huber
            + 0.16 * residual_loss + 0.025 * consistency
            + 0.045 * core.ccc_loss(pp, yy) + 0.015 * core.corr_loss(pp, yy)
            + 0.035 * kl_mean + 0.075 * kl_attn + 0.045 * proto_score
            + 0.012 * proto_div + 0.035 * attn_uniform + 0.025 * low_bce
            + 0.010 * severe_under + 0.004 * (dd ** 2).mean())
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pair_huber": float(pair_huber.detach().cpu()),
        "pair_rmse_batch": float(pair_rmse.detach().cpu()),
        "residual_loss": float(residual_loss.detach().cpu()),
        "consistency": float(consistency.detach().cpu()),
        "attn_entropy": float(out["attn_entropy"].mean().detach().cpu()),
        "abs_match": float(out["abs_match"].mean().detach().cpu()),
        "raw_residual_std": float(residual_raw.std(unbiased=False).detach().cpu()),
    }


def model_from_videos(videos, args):
    model = core.AEE2014(
        videos[0].segments.shape[-1], videos[0].desc.shape[0],
        args.hidden_dim, args.n_proto, 1.0, args.residual_max,
        args.dropout, args.proto_tau, args.attn_temp, args.score_max,
        args.minimal_cut, args.match_center, args.match_scale,
        args.frame_match_weight, args.segment_match_weight,
    ).to(core.DEVICE)
    model.force_uniform_attention = args.attention_mode == "uniform"
    # V3's learned gate is intentionally excluded.  Freeze it so gradients and
    # optimizer state cannot silently turn it into part of the V4 candidate.
    for p in model.res_gate.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def evaluate_candidate(model, videos, args, split, fold):
    model.eval()
    rows = []
    for v in tqdm(videos, desc=f"eval {split} fold{fold}", ncols=100, leave=False):
        item = core.ExprDataset([v], train=False, args=args)[0]
        b = core.collate([item])
        anchor = b["anchor"].to(core.DEVICE)
        out = model(b["x"].to(core.DEVICE), b["mask"].to(core.DEVICE),
                    b["bag_mask"].to(core.DEVICE), b["desc"].to(core.DEVICE), anchor)
        delta = torch.clamp(out["residual_raw"], -args.residual_max, args.residual_max)
        candidate = torch.clamp(anchor + delta, 0.0, args.score_max)
        row = {
            "fold": fold, "split": split, "video_id": v.video_id,
            "pair_id": v.pair_id, "task": v.task, "gt": float(v.score),
            "bin": v.bin, "anchor_pred": float(anchor.cpu()[0]),
            "candidate_pred": float(candidate.cpu()[0]),
            "raw_delta": float(delta.cpu()[0]),
            "v3_gate_unused": float(out["residual_gate"].cpu()[0]),
            "proto_score": float(out["proto_score"].cpu()[0]),
            "proto_score_attn": float(out["proto_score_attn"].cpu()[0]),
            "seg_mean": float(out["seg_mean"].cpu()[0]),
            "seg_attn": float(out["seg_attn"].cpu()[0]),
            "top_score": float(out["top_score"].cpu()[0]),
            "low_mass": float(out["low_mass"].cpu()[0]),
            "mid_mass": float(out["mid_mass"].cpu()[0]),
            "high_mass": float(out["high_mass"].cpu()[0]),
            "attn_peak": float(out["attn_peak"].cpu()[0]),
            "attn_entropy": float(out["attn_entropy"].cpu()[0]),
            "proto_entropy": float(out["proto_entropy"].cpu()[0]),
            "proto_peak": float(out["proto_peak"].cpu()[0]),
            "abs_match": float(out["abs_match"].cpu()[0]),
            "null_reject_rate": float(out["null_reject_rate"].cpu()[0]),
            "raw_sim_max": float(out["raw_sim_max"].cpu()[0]),
            "n_segments": len(v.segments),
        }
        for k, value in enumerate(out["temporal_stats"].cpu().numpy()[0]):
            row[f"temporal{k:02d}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def ema_update(ema, model, decay):
    state = model.state_dict()
    if ema is None:
        return {k: v.detach().cpu().clone() for k, v in state.items()}
    for k, value in state.items():
        value = value.detach().cpu()
        if torch.is_floating_point(value):
            ema[k].mul_(decay).add_(value, alpha=1.0 - decay)
        else:
            ema[k].copy_(value)
    return ema


def train_fold(fold, train_videos, val_videos, dev_videos, test_videos, args, fold_dir):
    Path(fold_dir).mkdir(parents=True, exist_ok=True)
    dataset = core.ExprDataset(train_videos, train=True, args=args)
    sampler = core.PairBalancedSampler(train_videos, args.batch_size, args.steps_per_epoch,
                                       args.seed + 1009 * fold)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=core.collate,
                        num_workers=args.num_workers, pin_memory=True)
    model = model_from_videos(train_videos, args)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.min_lr)
    ema, history = None, []

    progress_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_rows = []
        bar = tqdm(loader, desc=f"V4 fold{fold} ep{epoch}", ncols=145)
        for batch in bar:
            x = batch["x"].to(core.DEVICE)
            mask = batch["mask"].to(core.DEVICE)
            bag = batch["bag_mask"].to(core.DEVICE)
            desc = batch["desc"].to(core.DEVICE)
            anchor = batch["anchor"].to(core.DEVICE)
            y = batch["y"].to(core.DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(x, mask, bag, desc, anchor)
            loss, stats = v4_loss(out, y, anchor, model, args, batch["pair_id"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            batch_rows.append(stats)
            bar.set_postfix(loss=f"{stats['loss']:.3f}", prmse=f"{stats['pair_rmse_batch']:.2f}")
        scheduler.step()
        if epoch >= args.ema_start:
            ema = ema_update(ema, model, args.ema_decay)
        rec = pd.DataFrame(batch_rows).mean(numeric_only=True).to_dict()
        rec.update({"fold": fold, "epoch": epoch, "lr": opt.param_groups[0]["lr"]})
        if epoch == 1 or epoch == args.epochs or epoch % args.eval_every == 0:
            current = copy.deepcopy(model.state_dict())
            if ema is not None:
                model.load_state_dict(ema, strict=True)
            val_df = evaluate_candidate(model, val_videos, args, "outer_oof", fold)
            val_pair = pair_frame(val_df)
            met = core.metrics_np(val_pair["gt"], val_pair["candidate_pred"])
            rec.update({f"val_{k}": v for k, v in met.items() if k != "n"})
            model.load_state_dict(current, strict=True)
            print(f"fold={fold} epoch={epoch} OOF-monitor RMSE={met['rmse']:.4f} "
                  f"CCC={met['ccc']:.4f}; monitoring only, no epoch selection")
        history.append(rec)
        pd.DataFrame(history).to_csv(os.path.join(fold_dir, "history.csv"), index=False)
        eta = (time.monotonic() - progress_start) / epoch * (args.epochs - epoch)
        print(f"APNET_PROGRESS Fold {fold + 1}/{args.folds} | Epoch {epoch}/{args.epochs} | "
              f"train_loss={rec['loss']:.4f} | lr={rec['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)

    if ema is not None:
        model.load_state_dict(ema, strict=True)
    checkpoint = {
        "model_state": model.state_dict(), "fold": fold, "args": vars(args),
        "model_name": "AEE2014V4PairOOFSafeResidual",
        "note": "fixed epoch + EMA; outer validation did not select checkpoint",
    }
    torch.save(checkpoint, os.path.join(fold_dir, "model_ema.pth"))
    return (evaluate_candidate(model, val_videos, args, "train_oof", fold),
            evaluate_candidate(model, dev_videos, args, "dev", fold),
            evaluate_candidate(model, test_videos, args, "test", fold))


def pair_frame(clips):
    rows = []
    for pair_id, group in clips.groupby("pair_id", sort=True):
        group = group.sort_values("task")
        if set(group.task) != {"Freeform", "Northwind"} or len(group) != 2:
            raise ValueError(f"Incomplete pair in predictions: {pair_id}")
        ff = group[group.task == "Freeform"].iloc[0]
        nw = group[group.task == "Northwind"].iloc[0]
        row = {
            "pair_id": str(pair_id), "gt": float(group["gt"].iloc[0]),
            "bin": str(group.bin.iloc[0]), "fold": int(group.fold.iloc[0]),
            "anchor_pred": float(group.anchor_pred.mean()),
            "candidate_pred": float(group.candidate_pred.mean()),
            "ff_anchor": float(ff.anchor_pred), "nw_anchor": float(nw.anchor_pred),
            "ff_candidate": float(ff.candidate_pred), "nw_candidate": float(nw.candidate_pred),
            "ff_delta": float(ff.candidate_pred - ff.anchor_pred),
            "nw_delta": float(nw.candidate_pred - nw.anchor_pred),
        }
        for column in [
            "proto_score", "proto_score_attn", "seg_mean", "seg_attn", "top_score",
            "low_mass", "mid_mass", "high_mass", "attn_peak", "attn_entropy",
            "proto_entropy", "proto_peak", "abs_match", "null_reject_rate",
            "raw_sim_max", *[f"temporal{k:02d}" for k in range(10)],
        ]:
            if column in group:
                row[column] = float(group[column].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    result["raw_delta_pair"] = result.candidate_pred - result.anchor_pred
    result["delta_disagreement"] = (result.ff_delta - result.nw_delta).abs()
    result["candidate_task_gap"] = (result.ff_candidate - result.nw_candidate).abs()
    return result


def ensemble_clips(frames):
    data = pd.concat(frames, ignore_index=True)
    id_cols = ["split", "video_id", "pair_id", "task", "gt", "bin"]
    numeric = [c for c in data.select_dtypes(include=[np.number]).columns
               if c not in {"gt", "fold"}]
    rows = []
    for video_id, group in data.groupby("video_id", sort=True):
        row = {c: group[c].iloc[0] for c in id_cols}
        for c in numeric:
            row[c] = float(group[c].mean())
        row["candidate_ensemble_std"] = float(group.candidate_pred.std(ddof=0))
        row["delta_ensemble_std"] = float((group.candidate_pred - group.anchor_pred).std(ddof=0))
        row["fold"] = -1
        rows.append(row)
    return pd.DataFrame(rows)


def sigmoid(x):
    x = np.clip(np.asarray(x, float), -35, 35)
    return 1.0 / (1.0 + np.exp(-x))


def gate_features(pair):
    columns = ["anchor_pred", "raw_delta_pair", "delta_disagreement",
               "candidate_task_gap", "abs_match", "null_reject_rate",
               "attn_entropy", "proto_score_attn", "low_mass", "high_mass"]
    x = pair.reindex(columns=columns, fill_value=0.0).to_numpy(float)
    x[:, 0] /= 63.0
    x[:, 1:4] /= 12.0
    return x, columns


def logistic_fit(x, y, l2=8.0, iterations=60):
    x = np.asarray(x, float); y = np.asarray(y, float)
    mu = x.mean(0); sd = np.maximum(x.std(0), 1e-5)
    z = np.c_[np.ones(len(x)), (x - mu) / sd]
    if len(np.unique(y)) < 2:
        prior = float((y.sum() + 1.0) / (len(y) + 2.0))
        return {"mu": mu, "sd": sd, "w": np.r_[math.log(prior / (1-prior)), np.zeros(x.shape[1])]}
    w = np.zeros(z.shape[1], dtype=float)
    reg = np.eye(len(w)) * l2; reg[0, 0] = 0.0
    for _ in range(iterations):
        p = sigmoid(z @ w)
        ww = np.maximum(p * (1 - p), 1e-4)
        grad = z.T @ (p - y) + reg @ w
        hess = z.T @ (z * ww[:, None]) + reg
        step = np.linalg.solve(hess + np.eye(len(w)) * 1e-8, grad)
        w -= step
        if np.linalg.norm(step) < 1e-7:
            break
    return {"mu": mu, "sd": sd, "w": w}


def logistic_predict(model, x):
    z = np.c_[np.ones(len(x)), (np.asarray(x, float) - model["mu"]) / model["sd"]]
    return sigmoid(z @ model["w"])


def clipped_pair_delta(pair, cap):
    return (np.clip(pair.ff_delta.to_numpy(float), -cap, cap)
            + np.clip(pair.nw_delta.to_numpy(float), -cap, cap)) / 2.0


def fit_shrink(pair, caps, alpha_max=1.25):
    y = pair["gt"].to_numpy(float); anchor = pair["anchor_pred"].to_numpy(float)
    best = None
    for cap in caps:
        delta = clipped_pair_delta(pair, cap)
        denom = np.mean(delta ** 2)
        alpha = 0.0 if denom < 1e-12 else float(np.clip(
            -np.mean((anchor - y) * delta) / denom, 0.0, alpha_max))
        pred = np.clip(anchor + alpha * delta, 0, 63)
        rmse = core.metrics_np(y, pred)["rmse"]
        item = {"method": "shrink", "cap": float(cap), "alpha": alpha, "rmse_fit": rmse}
        if best is None or rmse < best["rmse_fit"]:
            best = item
    return best


def fit_gate(pair, caps, feature_names):
    x, _ = gate_features(pair)
    y = pair["gt"].to_numpy(float); anchor = pair["anchor_pred"].to_numpy(float)
    best = None
    for cap in caps:
        delta = clipped_pair_delta(pair, cap)
        helpful = ((anchor + delta - y) ** 2 < (anchor - y) ** 2).astype(float)
        gate_model = logistic_fit(x, helpful)
        probability = logistic_predict(gate_model, x)
        gated_delta = probability * delta
        denom = np.mean(gated_delta ** 2)
        alpha = 0.0 if denom < 1e-12 else float(np.clip(
            -np.mean((anchor - y) * gated_delta) / denom, 0.0, 2.0))
        pred = np.clip(anchor + alpha * gated_delta, 0, 63)
        rmse = core.metrics_np(y, pred)["rmse"]
        item = {"method": "gate", "cap": float(cap), "alpha": alpha,
                "rmse_fit": rmse, "gate_model": gate_model,
                "helpful_rate": float(helpful.mean()), "feature_names": feature_names}
        if best is None or rmse < best["rmse_fit"]:
            best = item
    return best


def predict_pairs(pair, calibrator):
    anchor = pair.anchor_pred.to_numpy(float)
    if calibrator["method"] == "anchor":
        return anchor.copy(), np.zeros(len(pair)), np.zeros(len(pair))
    delta = clipped_pair_delta(pair, calibrator["cap"])
    if calibrator["method"] == "shrink":
        gate = np.ones(len(pair))
    else:
        x, _ = gate_features(pair)
        gate = logistic_predict(calibrator["gate_model"], x)
    correction = calibrator["alpha"] * gate * delta
    return np.clip(anchor + correction, 0, 63), gate, correction


def crossfit_calibrator(oof_pair, args):
    folds = sorted(oof_pair.fold.unique())
    methods = {"anchor": np.zeros(len(oof_pair)), "shrink": np.zeros(len(oof_pair)),
               "gate": np.zeros(len(oof_pair))}
    records = []
    _, feature_names = gate_features(oof_pair)
    caps = [float(x) for x in args.calibration_caps.split(",") if x.strip()]
    for fold in folds:
        train = oof_pair[oof_pair.fold != fold].copy()
        valid = oof_pair[oof_pair.fold == fold].copy()
        idx = np.flatnonzero(oof_pair.fold.to_numpy() == fold)
        shrink = fit_shrink(train, caps)
        gate = fit_gate(train, caps, feature_names)
        anchor_model = {"method": "anchor"}
        for name, model in [("anchor", anchor_model), ("shrink", shrink), ("gate", gate)]:
            pred, _, _ = predict_pairs(valid, model)
            methods[name][idx] = pred
            records.append({"fold": int(fold), "method": name,
                            "parameters": core.to_jsonable(model),
                            **core.metrics_np(valid["gt"], pred)})
    summary = {}
    for name, pred in methods.items():
        summary[name] = core.metrics_np(oof_pair["gt"], pred)
    anchor_rmse = summary["anchor"]["rmse"]
    candidates = sorted(["shrink", "gate"], key=lambda n: summary[n]["rmse"])
    chosen = candidates[0] if summary[candidates[0]]["rmse"] <= anchor_rmse - args.safety_margin else "anchor"
    if chosen == "anchor":
        final = {"method": "anchor"}
    elif chosen == "shrink":
        final = fit_shrink(oof_pair, caps)
    else:
        final = fit_gate(oof_pair, caps, feature_names)
    return final, summary, pd.DataFrame(records), methods


def apply_calibrator_to_clips(clips, calibrator):
    result = clips.copy()
    pair = pair_frame(result)
    pair_pred, gate, correction = predict_pairs(pair, calibrator)
    pair = pair.assign(pred=pair_pred, safe_gate=gate, safe_correction=correction)
    controls = pair.set_index("pair_id")[["safe_gate"]].to_dict(orient="index")
    if calibrator["method"] == "anchor":
        result["safe_gate"] = 0.0
        result["safe_correction"] = 0.0
        result["pred"] = result.anchor_pred
    else:
        cap = calibrator["cap"]; alpha = calibrator["alpha"]
        result["safe_gate"] = result.pair_id.astype(str).map(lambda p: controls[str(p)]["safe_gate"])
        result["safe_correction"] = alpha * result.safe_gate * np.clip(
            result.candidate_pred - result.anchor_pred, -cap, cap)
        result["pred"] = np.clip(result.anchor_pred + result.safe_correction, 0, 63)
    return result, pair


def save_views(clips, pair, out_dir, split):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    clips.to_csv(os.path.join(out_dir, f"{split}_clip_predictions.csv"), index=False)
    pair.to_csv(os.path.join(out_dir, f"{split}_paired_predictions.csv"), index=False)
    metrics = [
        {"view": "paired_mean", **core.metrics_np(pair["gt"], pair["pred"])},
        {"view": "clips_pooled", **core.metrics_np(clips["gt"], clips["pred"])},
    ]
    for task in ["Freeform", "Northwind"]:
        g = clips[clips.task == task]
        metrics.append({"view": task, **core.metrics_np(g["gt"], g["pred"])})
    pd.DataFrame(metrics).to_csv(os.path.join(out_dir, f"{split}_metrics_by_view.csv"), index=False)
    core.bin_metrics(pair).to_csv(os.path.join(out_dir, f"{split}_paired_bin_metrics.csv"), index=False)
    return metrics


def ridge_fit_predict(train_x, train_y, test_x, alpha):
    mu = train_x.mean(0); sd = np.maximum(train_x.std(0), 1e-5)
    x = (train_x - mu) / sd; z = (test_x - mu) / sd
    xb = np.c_[np.ones(len(x)), x]; zb = np.c_[np.ones(len(z)), z]
    reg = np.eye(xb.shape[1]) * alpha; reg[0, 0] = 0
    beta = np.linalg.solve(xb.T @ xb + reg, xb.T @ train_y)
    return zb @ beta


def component_probes(oof_pair, safe_pair):
    uniform_cols = ["proto_score", "seg_mean", "low_mass", "mid_mass", "high_mass",
                    "abs_match", "null_reject_rate", "raw_sim_max",
                    *[f"temporal{k:02d}" for k in range(10)]]
    attention_cols = uniform_cols + ["proto_score_attn", "seg_attn", "top_score",
                                     "attn_peak", "attn_entropy"]
    predictions = {
        "M0_anchor": oof_pair.anchor_pred.to_numpy(float),
        "M4_deep_candidate_raw": oof_pair.candidate_pred.to_numpy(float),
        "M5_safe_calibrated": safe_pair.pred.to_numpy(float),
    }
    for name, columns in [("M2_lexicon_uniform_probe", uniform_cols),
                          ("M3_plus_attention_probe", attention_cols)]:
        x = oof_pair.reindex(columns=columns, fill_value=0.0).to_numpy(float)
        pred = np.zeros(len(oof_pair))
        for fold in sorted(oof_pair.fold.unique()):
            tr = oof_pair.fold.to_numpy() != fold; va = ~tr
            target = oof_pair["gt"].to_numpy(float)[tr] - oof_pair["anchor_pred"].to_numpy(float)[tr]
            delta = ridge_fit_predict(x[tr], target, x[va], alpha=10.0)
            pred[va] = np.clip(oof_pair.anchor_pred.to_numpy(float)[va] + delta, 0, 63)
        predictions[name] = pred
    rows = []
    for name, pred in predictions.items():
        rows.append({"component": name, **core.metrics_np(oof_pair["gt"], pred)})
    table = pd.DataFrame(rows).sort_values("rmse")
    m = table.set_index("component").rmse.to_dict()
    verdict = {
        "lexicon": ("helpful" if m["M2_lexicon_uniform_probe"] < m["M0_anchor"] - .15
                    else "weak_or_not_generalizing"),
        "attention": ("helpful" if m["M3_plus_attention_probe"] < m["M2_lexicon_uniform_probe"] - .15
                      else "weak_or_not_generalizing"),
        "deep_residual": ("helpful" if m["M4_deep_candidate_raw"] < m["M0_anchor"] - .15
                          else "overfit_or_harmful"),
        "safe_calibration": ("helpful" if m["M5_safe_calibrated"] < m["M0_anchor"] - .15
                             else "fell_back_to_anchor_or_no_gain"),
    }
    return table, verdict


def diagnostic_test_oracle(test_clips, args):
    pair = pair_frame(test_clips)
    caps = [float(x) for x in args.calibration_caps.split(",") if x.strip()]
    shrink = fit_shrink(pair, caps)
    pred, _, _ = predict_pairs(pair, shrink)
    return {
        "diagnostic_only": True,
        "warning": "fits cap/alpha on test labels; never report as a model result",
        "anchor": core.metrics_np(pair["gt"], pair["anchor_pred"]),
        "raw_candidate": core.metrics_np(pair["gt"], pair["candidate_pred"]),
        "test_fitted_shrink": {**core.to_jsonable(shrink), "metrics": core.metrics_np(pair["gt"], pred)},
    }


def main(args):
    core.set_seed(args.seed)
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    args.au_columns_list = [c.strip() for c in args.au_columns.split(",") if c.strip()]
    train_meta = core.read_txt(args.train_txt)
    dev_meta = core.read_txt(args.dev_txt)
    test_meta = core.read_txt(args.test_txt)
    for name, meta in [("train", train_meta), ("dev", dev_meta), ("test", test_meta)]:
        check_complete_pairs(meta, name)
    if ((set(train_meta.pair_id) & set(dev_meta.pair_id)) or
            (set(train_meta.pair_id) & set(test_meta.pair_id)) or
            (set(dev_meta.pair_id) & set(test_meta.pair_id))):
        raise ValueError("pair_id overlap across train/dev/test")

    train_arrays, train_paths = load_split(train_meta, args, "train")
    dev_arrays, dev_paths = load_split(dev_meta, args, "dev")
    test_arrays, test_paths = load_split(test_meta, args, "test")
    assignments, fold_table = make_stratified_pair_folds(train_meta, args.folds, args.seed)
    fold_table.to_csv(os.path.join(args.save_dir, "pair_folds.csv"), index=False)

    oof_frames, dev_frames, test_frames, anchor_summaries = [], [], [], []
    all_pairs = sorted(set(train_meta.pair_id.astype(str)))
    for fold in range(args.folds):
        val_pairs = [p for p in all_pairs if assignments[p] == fold]
        fit_pairs = [p for p in all_pairs if assignments[p] != fold]
        fit_meta, fit_arrays, fit_paths = select_items(
            train_meta, train_arrays, train_paths, fit_pairs)
        val_meta, val_arrays, val_paths = select_items(
            train_meta, train_arrays, train_paths, val_pairs)
        mean, std = core.fit_global_stats(fit_arrays)
        fit_desc = np.stack([core.make_descriptor(a, mean, std, t)
                             for a, t in zip(fit_arrays, fit_meta.task)])
        val_desc = np.stack([core.make_descriptor(a, mean, std, t)
                             for a, t in zip(val_arrays, val_meta.task)])
        dev_desc = np.stack([core.make_descriptor(a, mean, std, t)
                             for a, t in zip(dev_arrays, dev_meta.task)])
        test_desc = np.stack([core.make_descriptor(a, mean, std, t)
                              for a, t in zip(test_arrays, test_meta.task)])
        standardized, dmu, dsig = core.standardize(fit_desc, val_desc, dev_desc, test_desc)
        xfit, xval, xdev, xtest = standardized
        yfit = fit_meta.score.to_numpy(np.float32)
        yval = val_meta.score.to_numpy(np.float32)
        ydev = dev_meta.score.to_numpy(np.float32)
        ytest = test_meta.score.to_numpy(np.float32)
        # fit_anchor_mode accepts one dev and one test.  Concatenate dev+test as
        # its test matrix, then split predictions; no evaluation labels enter fitting.
        eval_meta = pd.concat([dev_meta, test_meta], ignore_index=True)
        xeval = np.concatenate([xdev, xtest], axis=0)
        yeval = np.concatenate([ydev, ytest], axis=0)
        anchor = core.fit_anchor_mode(xfit, yfit, xval, yval, xeval, yeval,
                                      fit_meta, val_meta, eval_meta, args)
        dev_anchor = anchor["test"][:len(dev_meta)]
        test_anchor = anchor["test"][len(dev_meta):]
        anchor_summaries.append({"fold": fold, **anchor["summary"]})
        fit_videos = core.build_videos(fit_meta, fit_arrays, fit_paths,
                                       anchor["train_oof"], mean, std, dmu, dsig,
                                       args, f"fold{fold}_fit")
        val_videos = core.build_videos(val_meta, val_arrays, val_paths,
                                       anchor["dev"], mean, std, dmu, dsig,
                                       args, f"fold{fold}_oof")
        dev_videos = core.build_videos(dev_meta, dev_arrays, dev_paths,
                                       dev_anchor, mean, std, dmu, dsig,
                                       args, f"fold{fold}_dev")
        test_videos = core.build_videos(test_meta, test_arrays, test_paths,
                                        test_anchor, mean, std, dmu, dsig,
                                        args, f"fold{fold}_test")
        fold_dir = os.path.join(args.save_dir, f"fold_{fold}")
        Path(fold_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(os.path.join(fold_dir, "preprocessing.npz"),
                            raw_mean=mean, raw_std=std, desc_mu=dmu, desc_sig=dsig,
                            au_columns=np.asarray(args.au_columns_list, dtype=str))
        jdump({"fold": fold, "fit_pairs": fit_pairs, "held_out_pairs": val_pairs,
               "anchor_models": anchor["models"], "anchor_summary": anchor["summary"]},
              os.path.join(fold_dir, "fold_training_context.json"))
        oof, dev, test = train_fold(fold, fit_videos, val_videos,
                                    dev_videos, test_videos, args, fold_dir)
        oof_frames.append(oof); dev_frames.append(dev); test_frames.append(test)
        oof.to_csv(os.path.join(fold_dir, "oof_clip_predictions.csv"), index=False)
        dev.to_csv(os.path.join(fold_dir, "dev_clip_predictions.csv"), index=False)
        test.to_csv(os.path.join(fold_dir, "test_clip_predictions.csv"), index=False)

    oof_clips = pd.concat(oof_frames, ignore_index=True).sort_values(["pair_id", "task"])
    dev_clips = ensemble_clips(dev_frames)
    test_clips = ensemble_clips(test_frames)
    oof_pair = pair_frame(oof_clips)
    calibrator, calibration_cv, calibration_folds, _ = crossfit_calibrator(oof_pair, args)
    oof_safe_clips, oof_safe_pair = apply_calibrator_to_clips(oof_clips, calibrator)
    dev_safe_clips, dev_safe_pair = apply_calibrator_to_clips(dev_clips, calibrator)
    test_safe_clips, test_safe_pair = apply_calibrator_to_clips(test_clips, calibrator)

    # Preserve original outer fold ids in pair tables for component cross-fitting.
    oof_safe_pair["fold"] = oof_pair.fold.to_numpy()
    out = args.save_dir
    oof_metrics = save_views(oof_safe_clips, oof_safe_pair, out, "train_oof")
    dev_metrics = save_views(dev_safe_clips, dev_safe_pair, out, "dev")
    test_metrics = save_views(test_safe_clips, test_safe_pair, out, "test")
    calibration_folds.to_csv(os.path.join(out, "calibration_crossfit_folds.csv"), index=False)
    jdump({"chosen": calibrator, "crossfit_metrics": calibration_cv,
           "safety_margin": args.safety_margin}, os.path.join(out, "safe_calibrator.json"))
    jdump(anchor_summaries, os.path.join(out, "anchor_by_fold.json"))

    component_table, component_verdict = component_probes(oof_pair, oof_safe_pair)
    component_table.to_csv(os.path.join(out, "component_ablation_oof.csv"), index=False)
    jdump(diagnostic_test_oracle(test_clips, args),
          os.path.join(out, "diagnostic_test_oracle_ceiling.json"))
    summary = {
        "protocol": "train_pair_oof_ensemble_safe_calibration_dev_and_test_report",
        "test_used_for_training_or_selection": False,
        "seed": args.seed, "folds": args.folds,
        "attention_mode": args.attention_mode,
        "calibrator": calibrator, "calibration_crossfit": calibration_cv,
        "component_verdict": component_verdict,
        "train_oof_metrics": oof_metrics, "dev_metrics": dev_metrics,
        "test_metrics": test_metrics,
        "notes": {
            "primary_view": "paired_mean",
            "v3_learned_gate": "disabled",
            "model_selection": "fixed epoch plus EMA; outer OOF metrics monitored but did not select epoch",
            "score_range": "BDI-II 0..63",
        },
    }
    jdump(summary, os.path.join(out, "v4_summary.json"))
    with open(os.path.join(out, "forensic_component_report.md"), "w", encoding="utf-8") as f:
        f.write("# V4 strict-OOF component diagnosis\n\n")
        f.write("Test labels did not fit or select any reported prediction.\n\n")
        f.write("## OOF ablation\n\n```text\n" + component_table.to_string(index=False) + "\n```\n\n")
        f.write("## Verdict\n\n")
        for key, value in component_verdict.items():
            f.write(f"- **{key}**: `{value}`\n")
        f.write("\n## Calibration cross-fit\n\n```text\n")
        f.write(pd.DataFrame(calibration_cv).T.reset_index(names="method").to_string(index=False))
        f.write("\n```\n\n")
        f.write("The separate diagnostic_test_oracle_ceiling.json is label-fitted and must never be reported.\n")
    print(json.dumps(core.to_jsonable(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", default="manifests/train_2014.txt")
    p.add_argument("--dev_txt", default="manifests/dev_2014.txt")
    p.add_argument("--test_txt", default="manifests/test_2014.txt")
    p.add_argument("--data_root", default="")
    p.add_argument("--save_dir", required=True)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--score_max", type=float, default=63.0)
    p.add_argument("--minimal_cut", type=float, default=14.0)
    p.add_argument("--mild_cut", type=float, default=20.0)
    p.add_argument("--moderate_cut", type=float, default=29.0)
    p.add_argument("--au_columns", default=",".join(core.DEFAULT_AU_COLUMNS))
    p.add_argument("--no_success_filter", action="store_true")
    p.add_argument("--allow_missing_au", action="store_true")
    p.add_argument("--confidence_threshold", type=float, default=.55)
    p.add_argument("--resample_hz", type=float, default=30.0)
    p.add_argument("--max_interp_gap", type=float, default=.25)
    p.add_argument("--input_mode", default="raw_rel_delta",
                   choices=["raw", "raw_rel", "raw_rel_delta"])
    p.add_argument("--au_clip", type=float, default=5.0)
    p.add_argument("--robust_eps", type=float, default=.15)
    p.add_argument("--rel_clip", type=float, default=5.0)
    p.add_argument("--delta_clip", type=float, default=5.0)
    p.add_argument("--anchor_mode", default="task_specific", choices=["shared", "task_specific"])
    p.add_argument("--task_mode", default="shared")
    p.add_argument("--anchor_folds", type=int, default=4)
    p.add_argument("--anchor_alphas", default="3,10,30,100,300,1000,3000")
    p.add_argument("--seq_len", type=int, default=300)
    p.add_argument("--stride", type=int, default=150)
    p.add_argument("--max_segments", type=int, default=128)
    p.add_argument("--max_train_segments_per_video", type=int, default=40)
    p.add_argument("--hidden_dim", type=int, default=40)
    p.add_argument("--n_proto", type=int, default=10)
    p.add_argument("--residual_max", type=float, default=12.0)
    p.add_argument("--dropout", type=float, default=.18)
    p.add_argument("--proto_tau", type=float, default=.16)
    p.add_argument("--attn_temp", type=float, default=.55)
    p.add_argument("--match_center", type=float, default=.15)
    p.add_argument("--match_scale", type=float, default=.10)
    p.add_argument("--frame_match_weight", type=float, default=.35)
    p.add_argument("--segment_match_weight", type=float, default=.25)
    p.add_argument("--attention_mode", choices=["learned", "uniform"], default="learned")
    p.add_argument("--epochs", type=int, default=48)
    p.add_argument("--steps_per_epoch", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=4e-6)
    p.add_argument("--weight_decay", type=float, default=3e-3)
    p.add_argument("--grad_clip", type=float, default=3.0)
    p.add_argument("--ema_start", type=int, default=20)
    p.add_argument("--ema_decay", type=float, default=.88)
    p.add_argument("--eval_every", type=int, default=8)
    p.add_argument("--segment_keep_ratio", type=float, default=.88)
    p.add_argument("--noise_std", type=float, default=.008)
    p.add_argument("--feature_dropout", type=float, default=.015)
    p.add_argument("--time_mask_prob", type=float, default=.08)
    p.add_argument("--time_mask_ratio", type=float, default=.05)
    p.add_argument("--huber_beta", type=float, default=2.5)
    p.add_argument("--residual_beta", type=float, default=2.0)
    p.add_argument("--ordinal_sigma", type=float, default=.08)
    p.add_argument("--attn_entropy_target", type=float, default=.92)
    p.add_argument("--calibration_caps", default="2,3,4,5,6,8,10,12")
    p.add_argument("--safety_margin", type=float, default=.15)
    args = p.parse_args()
    if args.score_max != 63.0:
        raise ValueError("AVEC2014 BDI-II implementation requires score_max=63")
    if args.batch_size % 2:
        raise ValueError("pair-level batches require even batch_size")
    main(args)
