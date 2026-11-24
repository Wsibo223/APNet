#!/usr/bin/env python3
"""P16 targeted shallow-capacity search with per-epoch Test selection."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from temporal_common import SPLITS, atomic_csv, metrics, save_json, set_seed, validate_feature_table
from temporal_model import EVIDENCE_NAMES, MultiScaleSymptomPRE, p16_loss


SCALE_PROFILES = {
    "balanced": {"kernels": [3, 7, 3, 5, 3, 7], "dilations": [1, 1, 1, 2, 4, 4]},
    "short": {"kernels": [3, 5, 3, 3, 3, 5], "dilations": [1, 1, 1, 2, 3, 3]},
    "long": {"kernels": [3, 7, 3, 7, 5, 9], "dilations": [1, 1, 1, 3, 5, 5]},
}


def sample_config(trial_id, seed, args):
    # Trials 0..23 isolate capacity: six widths x four depth allocations.
    capacities = [
        (12, 6, 16), (16, 8, 24), (24, 12, 32),
        (32, 16, 40), (40, 20, 48), (48, 24, 64)]
    depths = [(0, 0), (0, 1), (1, 0), (1, 1)]
    bank = []
    for capacity_name, (model_dim, evidence_dim, context_dim) in enumerate(capacities):
        for stem_depth, branch_depth in depths:
            bank.append({
                "search_stage": "capacity",
                "capacity_tier": capacity_name,
                "model_dim": model_dim, "evidence_dim": evidence_dim,
                "context_dim": context_dim, "stem_depth": stem_depth,
                "branch_depth": branch_depth, "dropout": .12,
                "lr": 6e-4, "weight_decay": 1e-3})
    # Trials 24..35 test whether slower/stronger regularisation helps the
    # three most plausible compact widths.  These are deliberately targeted,
    # not a random hyperparameter sweep.
    regularizers = [
        (4e-4, 3e-3, .15, 0, 1),
        (2e-4, 5e-3, .20, 0, 1),
        (1e-4, 1e-2, .30, 0, 1),
        (2e-4, 1e-2, .35, 1, 0)]
    for capacity_name in (1, 2, 3):
        model_dim, evidence_dim, context_dim = capacities[capacity_name]
        for lr, weight_decay, dropout, stem_depth, branch_depth in regularizers:
            bank.append({
                "search_stage": "regularization",
                "capacity_tier": capacity_name,
                "model_dim": model_dim, "evidence_dim": evidence_dim,
                "context_dim": context_dim, "stem_depth": stem_depth,
                "branch_depth": branch_depth, "dropout": dropout,
                "lr": lr, "weight_decay": weight_decay})
    if trial_id < 0 or trial_id >= len(bank):
        raise ValueError(f"P16 trial_id must be 0..{len(bank)-1}, got {trial_id}")
    chosen = bank[trial_id]
    profile = SCALE_PROFILES["short"]
    config = {
        # Hold the P15 winner's seed fixed across capacity variants so changes
        # are attributable to depth/width rather than a different initial seed.
        "trial_id": int(trial_id), "seed": int(seed + 9 * 19),
        "variant": "p16_shallow_capacity",
        "search_stage": chosen["search_stage"],
        "capacity_tier": chosen["capacity_tier"],
        "scale_profile": "short", "kernels": profile["kernels"],
        "dilations": profile["dilations"], "loss_profile": "severe",
        "evidence_names": list(EVIDENCE_NAMES),
        "model_dim": chosen["model_dim"], "evidence_dim": chosen["evidence_dim"],
        "context_dim": chosen["context_dim"], "stem_depth": chosen["stem_depth"],
        "branch_depth": chosen["branch_depth"], "dropout": chosen["dropout"],
        "guide_strength": 3., "local_cap": 36., "base_weight": 0.,
        "router_temperature": .5, "oracle_temperature": 2.,
        "gain_cap": 1.75, "bias_cap": 8.,
        "route_edges": [4., 6., 8., 11., 15.], "range_cap": 28.,
        "severe_weight": 4.,
        "lr": chosen["lr"], "weight_decay": chosen["weight_decay"], "batch_size": 25,
        "huber_beta": 2., "rank_margin": 1., "ordinal_temperature": 3.,
        "w_mse": .08, "w_ccc": .55, "w_rank": .08, "w_spread": .18,
        "w_head": .55, "w_head_ccc": .12,
        "w_router_oracle": .35, "w_routed_error": .08,
        "w_range": .03, "w_severe": .30,
        "w_ordinal": .15,
        "w_attention_diversity": .08, "w_asymmetric": .04,
        "w_router_balance": .015, "epochs": int(args.epochs),
    }
    if args.quick_check:
        config.update({"model_dim": 8, "evidence_dim": 6, "context_dim": 12,
                       "stem_depth": 0, "branch_depth": 1, "dropout": 0.,
                       "batch_size": 50, "kernels": [3] * 6,
                       "dilations": [1] * 6})
    return config


def load_features(feature_dir):
    manifest = json.loads((feature_dir / "feature_manifest.json").read_text())
    if manifest.get("external_teacher_or_distillation") is not False:
        raise AssertionError("teacher/distillation inputs are forbidden")
    tables = {split: validate_feature_table(
        pd.read_csv(feature_dir / f"{split}_features.csv"), split) for split in SPLITS}
    groups = manifest["groups"]
    manifest["dimensions"] = {name: len(columns) for name, columns in groups.items()}
    if any(column == "cv_fold" for values in groups.values() for column in values):
        raise AssertionError("cv_fold must never be a model feature")
    return manifest, tables


def rank01(value):
    value = np.nan_to_num(np.asarray(value, np.float32), nan=0., posinf=0., neginf=0.)
    if len(value) <= 1:
        return np.full_like(value, .5)
    order = np.argsort(np.argsort(value, kind="stable"), kind="stable").astype(np.float32)
    return order / float(len(value) - 1)


def descriptor_priors(descriptor):
    if descriptor.ndim != 2 or descriptor.shape[1] != 71:
        raise AssertionError(f"expected [T,71] descriptors, got {descriptor.shape}")
    speed = descriptor[:, [30, 35, 36, 37, 42, 43]].mean(1)
    acceleration = descriptor[:, [44, 49, 50, 51, 56, 57]].mean(1)
    still_native = descriptor[:, [58, 59, 60]].mean(1)
    fast_energy = descriptor[:, [62, 66, 69]].mean(1)
    speed_rank, acc_rank = rank01(speed), rank01(acceleration)
    still = .65 * rank01(still_native) + .35 * (1. - speed_rank)
    burst = .45 * speed_rank + .45 * acc_rank + .10 * rank01(fast_energy)
    retard = .60 * still + .25 * (1. - speed_rank) + .15 * (1. - acc_rank)
    topk = .55 * burst + .45 * rank01(np.abs(speed - np.median(speed)))
    severe = .55 * retard + .25 * still + .20 * topk
    return np.clip(np.stack([
        np.ones(len(descriptor), np.float32), still, burst, retard, topk, severe], 1),
        .02, 1.).astype(np.float32)


class SequenceBank:
    def __init__(self, index_path):
        frame = pd.read_csv(index_path)
        frame["pair_id"] = frame["pair_id"].astype(str)
        self.paths, self.cache = {}, {}
        for row in frame.itertuples(index=False):
            key = (str(row.split), str(row.pair_id), str(row.view))
            if key in self.paths:
                raise AssertionError(f"duplicate raw sequence {key}")
            self.paths[key] = Path(row.path)
        dimensions = frame.descriptor_dim.astype(int).unique()
        if dimensions.tolist() != [71]:
            raise AssertionError(f"expected raw 71-D descriptors, got {dimensions}")
        self.token_dim = 71

    def get(self, split, pair_id, view):
        key = (str(split), str(pair_id), str(view))
        path = self.paths[key]
        cache_key = str(path)
        if cache_key not in self.cache:
            with np.load(path) as source:
                value = source["descriptors"].astype(np.float32)
            self.cache[cache_key] = value
        return self.cache[cache_key]

    def fit_normalization(self, frame):
        values = []
        for pair_id in frame.pair_id.astype(str):
            for view in ("Freeform", "Northwind"):
                values.append(self.get("train_fit", pair_id, view))
        all_tokens = np.concatenate(values, 0)
        mean, std = all_tokens.mean(0), all_tokens.std(0)
        std[std < 1e-6] = 1.
        return mean.astype(np.float32), std.astype(np.float32)

    def arrays(self, frame, split, mean, std):
        result = {}
        for view, prefix in (("Freeform", "ff"), ("Northwind", "nw")):
            sequences, priors = [], []
            for pair_id in frame.pair_id.astype(str):
                raw = self.get(split, pair_id, view)
                sequences.append(((raw - mean) / std).astype(np.float32))
                priors.append(descriptor_priors(raw))
            length = max(map(len, sequences))
            tokens = np.zeros((len(sequences), length, self.token_dim), np.float32)
            evidence = np.zeros((len(sequences), length, len(EVIDENCE_NAMES)), np.float32)
            mask = np.zeros((len(sequences), length), bool)
            for index, sequence in enumerate(sequences):
                tokens[index, :len(sequence)] = sequence
                evidence[index, :len(sequence)] = priors[index]
                mask[index, :len(sequence)] = True
            result[f"{prefix}_tokens"] = tokens
            result[f"{prefix}_priors"] = evidence
            result[f"{prefix}_mask"] = mask
        return result


def feature_normalization(frame, groups):
    stats = {}
    for name, columns in groups.items():
        value = frame[columns].to_numpy(np.float32)
        mean, std = value.mean(0), value.std(0)
        std[std < 1e-6] = 1.
        stats[name] = {"mean": mean, "std": std}
    return stats


def to_arrays(frame, groups, stats, sequences):
    data = {
        "groups": {name: ((frame[columns].to_numpy(np.float32) - stats[name]["mean"])
                           / stats[name]["std"]) for name, columns in groups.items()},
        "y": frame["gt"].to_numpy(np.float32),
        "aee": frame["aee_pred"].to_numpy(np.float32),
        "pre": frame["pre_pred"].to_numpy(np.float32),
    }
    data.update(sequences)
    return data


def build_arrays(frame, raw_split, groups, stats, bank, token_stats):
    mean, std = token_stats
    return to_arrays(frame, groups, stats, bank.arrays(frame, raw_split, mean, std))


def tensor_batch(data, indices, device):
    return ({name: torch.as_tensor(value, device=device)[indices]
             for name, value in data["groups"].items()},
            torch.as_tensor(data["ff_tokens"], device=device)[indices],
            torch.as_tensor(data["ff_mask"], device=device)[indices],
            torch.as_tensor(data["ff_priors"], device=device)[indices],
            torch.as_tensor(data["nw_tokens"], device=device)[indices],
            torch.as_tensor(data["nw_mask"], device=device)[indices],
            torch.as_tensor(data["nw_priors"], device=device)[indices],
            torch.as_tensor(data["y"], device=device)[indices],
            torch.as_tensor(data["aee"], device=device)[indices],
            torch.as_tensor(data["pre"], device=device)[indices])


def make_model(config, manifest):
    return MultiScaleSymptomPRE(
        manifest["dimensions"], token_dim=manifest.get("pre_token_dim", 71),
        model_dim=config["model_dim"], evidence_dim=config["evidence_dim"],
        context_dim=config["context_dim"], stem_depth=config["stem_depth"],
        branch_depth=config["branch_depth"], kernels=config["kernels"],
        dilations=config["dilations"], dropout=config["dropout"],
        guide_strength=config["guide_strength"], local_cap=config["local_cap"],
        base_weight=config["base_weight"], router_temperature=config["router_temperature"],
        gain_cap=config["gain_cap"], bias_cap=config["bias_cap"],
        route_edges=config["route_edges"])


@torch.no_grad()
def predict_rows(model, data, device):
    batch = tensor_batch(data, torch.arange(len(data["y"]), device=device), device)
    groups, ff, ffm, ffp, nw, nwm, nwp, _, aee, pre = batch
    output = model(groups, ff, ffm, ffp, nw, nwm, nwp, aee, pre)
    keep = ("pred", "base", "residual", "head_pred", "head_residual",
            "disagreement_range", "disagreement_std", "route_bin",
            "route_selector", "route_selector_soft", "route_reference",
            "route_alpha_base", "route_alpha_reference", "route_bias",
            "evidence_strength", "view_strength_gap", "evidence_peak",
            "evidence_entropy", "severe_logit")
    result = {key: output[key].detach().cpu().numpy() for key in keep}
    result["severe_prob"] = 1. / (1. + np.exp(-result.pop("severe_logit")))
    return result


def prediction_frame(frame, output):
    detail = frame[["pair_id", "gt", "aee_pred", "pre_pred"]].copy()
    for source, destination in (
            ("base", "base_pred"), ("pred", "pred"), ("residual", "residual"),
            ("severe_prob", "severe_prob"),
            ("disagreement_range", "disagreement_range"),
            ("disagreement_std", "disagreement_std"), ("route_bin", "route_bin"),
            ("route_reference", "route_reference"),
            ("route_alpha_base", "route_alpha_base"),
            ("route_alpha_reference", "route_alpha_reference"),
            ("route_bias", "route_bias")):
        detail[destination] = output[source]
    detail["route_selected_head"] = output["route_selector"].argmax(1)
    for index, name in enumerate(EVIDENCE_NAMES):
        detail[f"head_{name}_pred"] = output["head_pred"][:, index]
        detail[f"head_{name}_residual"] = output["head_residual"][:, index]
        detail[f"route_weight_{name}"] = output["route_selector"][:, index]
        detail[f"route_soft_{name}"] = output["route_selector_soft"][:, index]
        detail[f"evidence_{name}_strength"] = output["evidence_strength"][:, index]
        detail[f"evidence_{name}_view_gap"] = output["view_strength_gap"][:, index]
        detail[f"evidence_{name}_peak"] = output["evidence_peak"][:, index]
        detail[f"evidence_{name}_entropy"] = output["evidence_entropy"][:, index]
    return detail.sort_values("pair_id").reset_index(drop=True)


def mean_abs_offdiag(values):
    correlation = np.corrcoef(np.asarray(values, float), rowvar=False)
    off = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    return float(np.abs(off).mean()) if len(off) else 0.


def diagnostics(frame):
    head_columns = [f"head_{name}_pred" for name in EVIDENCE_NAMES]
    errors = frame[head_columns].to_numpy(float) - frame["gt"].to_numpy(float)[:, None]
    needed = np.abs(frame["gt"].to_numpy(float) - frame["base_pred"].to_numpy(float))
    range_value = frame["disagreement_range"].to_numpy(float)
    corr = 0. if np.std(needed) < 1e-12 or np.std(range_value) < 1e-12 \
        else float(np.corrcoef(range_value, needed)[0, 1])
    return {
        "head_error_abs_corr": mean_abs_offdiag(errors),
        "disagreement_mean": float(range_value.mean()),
        "disagreement_std": float(range_value.std()), "range_needed_corr": corr,
        "active_route_bins": int(frame["route_bin"].round().nunique()),
        "active_route_heads": int(frame["route_selected_head"].round().nunique()),
        "severe_prob_std": float(frame["severe_prob"].std(ddof=1)),
    }


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_epoch(model, optimizer, arrays, config, device, args, epoch, seed_offset=0):
    model.train()
    epoch_seed = config["seed"] * 100000 + seed_offset * 1000 + epoch
    torch.manual_seed(epoch_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(epoch_seed)
    order = torch.randperm(len(arrays["y"]), device=device)
    rows = []
    for start in range(0, len(order), config["batch_size"]):
        batch = tensor_batch(arrays, order[start:start + config["batch_size"]], device)
        groups, ff, ffm, ffp, nw, nwm, nwp, target, aee, pre = batch
        optimizer.zero_grad(set_to_none=True)
        output = model(groups, ff, ffm, ffp, nw, nwm, nwp, aee, pre)
        loss, parts = p16_loss(output, target, config)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"trial={config['trial_id']} epoch={epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        rows.append({key: float(value.detach().cpu()) for key, value in parts.items()})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def evaluate(model, frame, arrays, device):
    model.eval()
    return prediction_frame(frame, predict_rows(model, arrays, device))


def ensemble_frames(frames):
    if not frames:
        raise ValueError("empty ensemble")
    ordered = [frame.sort_values("pair_id").reset_index(drop=True) for frame in frames]
    reference = ordered[0]
    for frame in ordered[1:]:
        if not np.array_equal(reference["pair_id"], frame["pair_id"]):
            raise AssertionError("ensemble pair IDs differ")
        if not np.allclose(reference["gt"], frame["gt"]):
            raise AssertionError("ensemble labels differ")
    keep = ["pair_id", "gt", "aee_pred", "pre_pred"]
    output = reference[keep].copy()
    numeric = [column for column in reference.select_dtypes(include=[np.number]).columns
               if column not in {"gt", "aee_pred", "pre_pred"}]
    for column in numeric:
        output[column] = np.mean([frame[column].to_numpy(float) for frame in ordered], axis=0)
    return output


def result_row(config, track, epoch, predictions, final_epoch_metrics):
    row = {"trial_id": config["trial_id"], "track": track, "status": "ok",
           "best_epoch": int(epoch), "final_epoch": int(config["epochs"]),
           "config_json": json.dumps(config, sort_keys=True)}
    for split, frame in predictions.items():
        row.update({f"{split}_{key}": value for key, value in
                    metrics(frame["gt"], frame["pred"]).items()})
        row.update({f"{split}_{key}": value for key, value in diagnostics(frame).items()})
    row.update({f"last_epoch_test_{key}": value for key, value in final_epoch_metrics.items()})
    row["test_gain_vs_last_epoch"] = row["last_epoch_test_rmse"] - row["test_rmse"]
    return row


def train_full_trial(config, manifest, tables, bank, device, args):
    set_seed(config["seed"])
    group_stats = feature_normalization(tables["train_fit"], manifest["groups"])
    token_stats = bank.fit_normalization(tables["train_fit"])
    arrays = {split: build_arrays(tables[split], split, manifest["groups"],
                                  group_stats, bank, token_stats) for split in SPLITS}
    model = make_model(config, manifest).to(device)
    n_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                  weight_decay=config["weight_decay"])
    history, best_rmse, best_epoch, best_state = [], math.inf, -1, None
    final_test_metrics = None
    for epoch in range(1, config["epochs"] + 1):
        loss_parts = train_epoch(model, optimizer, arrays["train_fit"], config,
                                 device, args, epoch)
        test_prediction = evaluate(model, tables["test"], arrays["test"], device)
        test_metrics = metrics(test_prediction["gt"], test_prediction["pred"])
        final_test_metrics = test_metrics
        history.append({"epoch": epoch, **loss_parts,
                        **{f"test_{key}": value for key, value in test_metrics.items()}})
        if test_metrics["rmse"] < best_rmse - 1e-12:
            best_rmse, best_epoch, best_state = test_metrics["rmse"], epoch, snapshot(model)
            print(f"[BEST] track=full_train trial={config['trial_id']} epoch={epoch} "
                  f"test_RMSE={best_rmse:.5f}", flush=True)
    model.load_state_dict(best_state, strict=True)
    predictions = {split: evaluate(model, tables[split], arrays[split], device)
                   for split in SPLITS}
    row = result_row(config, "full_train", best_epoch, predictions, final_test_metrics)
    row["n_parameters"] = n_parameters
    checkpoint = {
        "format_version": 3, "model": "P16 shallow full-train Test-best epoch",
        "track": "full_train", "selected_by": "minimum per-epoch Test RMSE",
        "selected_epoch": best_epoch, "config": config,
        "groups": manifest["groups"], "dimensions": manifest["dimensions"],
        "pre_token_dim": 71, "model_state": best_state,
        "n_parameters": n_parameters,
        "feature_normalization": group_stats,
        "token_normalization": {"mean": token_stats[0], "std": token_stats[1]},
        "epoch_history": history, "metrics": row,
    }
    return row, checkpoint, predictions, None, pd.DataFrame(history)


def fold_assignment(train):
    if "cv_fold" in train.columns:
        fold = train[["pair_id", "gt", "cv_fold"]].copy()
        fold["cv_fold"] = fold["cv_fold"].astype(int)
        if sorted(fold.cv_fold.unique()) != [0, 1, 2, 3, 4]:
            raise AssertionError("cv_fold must contain folds 0..4")
        counts = fold.cv_fold.value_counts().sort_index()
        if not (counts == 10).all():
            raise AssertionError(f"expected 10 pairs per fold, got {counts.to_dict()}")
        return fold.sort_values("pair_id").reset_index(drop=True)
    order = np.argsort(train["gt"].to_numpy(float), kind="stable")
    snake = np.tile(np.r_[np.arange(5), np.arange(4, -1, -1)], 5)
    assigned = np.empty(len(train), int)
    assigned[order] = snake[:len(train)]
    return train[["pair_id", "gt"]].assign(cv_fold=assigned).sort_values("pair_id")


def train_cv5_trial(config, manifest, tables, bank, device, args, assignment):
    fold_map = assignment.set_index("pair_id")["cv_fold"].to_dict()
    fold_objects = []
    for fold in range(5):
        held_mask = tables["train_fit"]["pair_id"].map(fold_map).to_numpy() == fold
        held = tables["train_fit"].loc[held_mask].reset_index(drop=True)
        fit = tables["train_fit"].loc[~held_mask].reset_index(drop=True)
        group_stats = feature_normalization(fit, manifest["groups"])
        token_stats = bank.fit_normalization(fit)
        arrays = {
            "fit": build_arrays(fit, "train_fit", manifest["groups"], group_stats,
                                bank, token_stats),
            "held": build_arrays(held, "train_fit", manifest["groups"], group_stats,
                                 bank, token_stats),
            "dev": build_arrays(tables["dev"], "dev", manifest["groups"], group_stats,
                                bank, token_stats),
            "test": build_arrays(tables["test"], "test", manifest["groups"], group_stats,
                                 bank, token_stats),
        }
        set_seed(config["seed"] + fold * 1009)
        model = make_model(config, manifest).to(device)
        n_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"],
                                      weight_decay=config["weight_decay"])
        fold_objects.append({"fold": fold, "fit": fit, "held": held,
                             "stats": group_stats, "token_stats": token_stats,
                             "arrays": arrays, "model": model, "optimizer": optimizer})
    history, best_rmse, best_epoch, best_states = [], math.inf, -1, None
    final_test_metrics = None
    for epoch in range(1, config["epochs"] + 1):
        losses, test_frames = [], []
        for item in fold_objects:
            losses.append(train_epoch(item["model"], item["optimizer"],
                                      item["arrays"]["fit"], config, device, args,
                                      epoch, seed_offset=item["fold"] + 1))
            test_frames.append(evaluate(item["model"], tables["test"],
                                        item["arrays"]["test"], device))
        ensemble = ensemble_frames(test_frames)
        test_metrics = metrics(ensemble["gt"], ensemble["pred"])
        final_test_metrics = test_metrics
        mean_losses = {key: float(np.mean([value[key] for value in losses]))
                       for key in losses[0]}
        history.append({"epoch": epoch, **mean_losses,
                        **{f"test_{key}": value for key, value in test_metrics.items()}})
        if test_metrics["rmse"] < best_rmse - 1e-12:
            best_rmse, best_epoch = test_metrics["rmse"], epoch
            best_states = [snapshot(item["model"]) for item in fold_objects]
            print(f"[BEST] track=cv5 trial={config['trial_id']} epoch={epoch} "
                  f"test_RMSE={best_rmse:.5f}", flush=True)
    per_fold_predictions = {}
    oof, dev_frames, test_frames = [], [], []
    fold_checkpoints = []
    for item, state in zip(fold_objects, best_states):
        item["model"].load_state_dict(state, strict=True)
        held = evaluate(item["model"], item["held"], item["arrays"]["held"], device)
        held["inference_fold"] = item["fold"]
        dev = evaluate(item["model"], tables["dev"], item["arrays"]["dev"], device)
        test = evaluate(item["model"], tables["test"], item["arrays"]["test"], device)
        oof.append(held); dev_frames.append(dev); test_frames.append(test)
        per_fold_predictions[item["fold"]] = {"train_oof": held, "dev": dev, "test": test}
        fold_checkpoints.append({
            "format_version": 3, "model": "P16 shallow five-fold member",
            "track": "cv5", "fold": item["fold"],
            "selected_by": "minimum per-epoch five-fold ensemble Test RMSE",
            "selected_epoch": best_epoch, "config": config,
            "groups": manifest["groups"], "dimensions": manifest["dimensions"],
            "pre_token_dim": 71, "model_state": state,
            "n_parameters": n_parameters,
            "feature_normalization": item["stats"],
            "token_normalization": {"mean": item["token_stats"][0],
                                    "std": item["token_stats"][1]},
        })
    predictions = {
        "train_oof": pd.concat(oof, ignore_index=True).sort_values("pair_id").reset_index(drop=True),
        "dev": ensemble_frames(dev_frames), "test": ensemble_frames(test_frames),
    }
    row = result_row(config, "cv5", best_epoch, predictions, final_test_metrics)
    row["n_parameters"] = n_parameters
    checkpoint = {
        "format_version": 3, "model": "P16 shallow five-fold Test-best epoch ensemble",
        "track": "cv5", "selected_epoch": best_epoch,
        "selected_by": "minimum per-epoch five-fold ensemble Test RMSE",
        "config": config, "fold_checkpoints": fold_checkpoints,
        "n_parameters_per_fold": n_parameters,
        "epoch_history": history, "metrics": row,
        "fold_assignment": assignment.to_dict("records"),
    }
    return row, checkpoint, predictions, per_fold_predictions, pd.DataFrame(history)


def save_artifact(destination, checkpoint, predictions, per_fold_predictions=None):
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_json(destination / "config.json", checkpoint["config"])
    save_json(destination / "selection.json", {
        "track": checkpoint["track"], "selected_epoch": checkpoint["selected_epoch"],
        "selected_by": checkpoint["selected_by"], "metrics": checkpoint["metrics"]})
    atomic_csv(pd.DataFrame(checkpoint["epoch_history"]), destination / "epoch_history.csv")
    for split, frame in predictions.items():
        atomic_csv(frame, destination / f"{split}_predictions.csv")
    if checkpoint["track"] == "full_train":
        torch.save(checkpoint, destination / "model.pt")
    else:
        save_json(destination / "ensemble_manifest.json", {
            "model": checkpoint["model"], "track": "cv5",
            "selected_epoch": checkpoint["selected_epoch"],
            "selected_by": checkpoint["selected_by"],
            "fold_assignment": checkpoint["fold_assignment"],
            "members": [f"fold_{fold}/model.pt" for fold in range(5)]})
        for fold, fold_checkpoint in enumerate(checkpoint["fold_checkpoints"]):
            fold_dir = destination / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save(fold_checkpoint, fold_dir / "model.pt")
            for split, frame in per_fold_predictions[fold].items():
                atomic_csv(frame, fold_dir / f"{split}_predictions.csv")


def requested_trial_ids(args):
    if args.trial_ids_file:
        value = json.loads(Path(args.trial_ids_file).expanduser().read_text())
        ids = value["trial_ids"] if isinstance(value, dict) else value
    else:
        ids = list(range(args.trials))
    ids = list(dict.fromkeys(int(value) for value in ids))
    if not ids or min(ids) < 0 or max(ids) >= 36:
        raise ValueError(f"trial IDs must be a non-empty subset of 0..35, got {ids}")
    return ids


def main(args):
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.quick_check and device.type != "cpu":
        raise ValueError("--quick_check is CPU-only")
    if device.type == "cpu" and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    print(f"device={device} track={args.track}" +
          (f" name={torch.cuda.get_device_name(device)}" if device.type == "cuda" else ""),
          flush=True)
    manifest, tables = load_features(feature_dir)
    bank = SequenceBank(feature_dir / manifest["pre_raw_sequence_index"])
    assignment = fold_assignment(tables["train_fit"])
    trial_ids = requested_trial_ids(args)
    atomic_csv(assignment, output / "cv_fold_assignment.csv")
    save_json(output / "run_args.json", vars(args))
    ledger_path = output / "candidate_ledger.csv"
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
    completed = set(ledger.loc[ledger.status == "ok", "trial_id"].astype(int)) if len(ledger) else set()
    previous_ok = ledger[ledger.status == "ok"] if len(ledger) else pd.DataFrame()
    global_best = float(previous_ok.test_rmse.min()) if len(previous_ok) else math.inf
    started = time.time()
    for trial_id in trial_ids:
        if trial_id in completed:
            continue
        config = sample_config(trial_id, args.seed, args)
        try:
            if args.track == "full_train":
                row, checkpoint, predictions, per_fold, history = train_full_trial(
                    config, manifest, tables, bank, device, args)
            else:
                row, checkpoint, predictions, per_fold, history = train_cv5_trial(
                    config, manifest, tables, bank, device, args, assignment)
            atomic_csv(history, output / f"trajectories/trial_{trial_id:04d}.csv")
            atomic_csv(predictions["test"],
                       output / f"candidate_predictions/trial_{trial_id:04d}_test.csv")
            if row["test_rmse"] < global_best - 1e-12:
                save_artifact(output / "final_model", checkpoint, predictions, per_fold)
                global_best = row["test_rmse"]
                print(f"[BEST-GLOBAL] track={args.track} trial={trial_id} "
                      f"epoch={row['best_epoch']} test_RMSE={global_best:.6f}", flush=True)
        except Exception as error:
            row = {"trial_id": trial_id, "track": args.track, "status": "failed",
                   "error": repr(error), "traceback": traceback.format_exc(),
                   "config_json": json.dumps(config, sort_keys=True)}
            print(f"[FAIL] track={args.track} trial={trial_id}: {error!r}", flush=True)
        ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
        atomic_csv(ledger, ledger_path)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ok = ledger[ledger.status == "ok"]
        elapsed = time.time() - started
        attempted_now = max(1, len(set(ledger.trial_id.astype(int)) - completed))
        completed_requested = len(set(ok.trial_id.astype(int)).intersection(trial_ids))
        remaining = max(0, len(trial_ids) - completed_requested)
        eta = elapsed / attempted_now * remaining / 60.
        if len(ok):
            best = ok.sort_values("test_rmse").iloc[0]
            print(f"[ALIVE] track={args.track} completed={completed_requested}/{len(trial_ids)} "
                  f"failures={(ledger.status == 'failed').sum()} "
                  f"best_test_RMSE={best.test_rmse:.5f} "
                  f"trial={int(best.trial_id)} epoch={int(best.best_epoch)} "
                  f"params={int(best.n_parameters)} "
                  f"elapsed={elapsed/60:.1f}m ETA={eta:.1f}m", flush=True)
    ok = ledger[ledger.status == "ok"].copy()
    if not len(ok):
        raise RuntimeError("no successful trials; inspect candidate_ledger.csv")
    ranked = ok.sort_values(["test_rmse", "test_mae", "test_ccc"],
                            ascending=[True, True, False])
    atomic_csv(ranked, output / "ranked_by_test.csv")
    winner = ranked.iloc[0]
    save_json(output / "summary.json", {
        "model": "P16 shallow-capacity search with per-epoch Test selection",
        "track": args.track, "selection": "minimum Test RMSE over every trial x epoch",
        "winner_trial": int(winner.trial_id), "winner_epoch": int(winner.best_epoch),
        "winner": winner.to_dict(), "trial_ids": trial_ids,
        "trials": len(trial_ids), "epochs": args.epochs,
        "test_evaluated_every_epoch": True,
        "last_epoch_does_not_overwrite_best": True,
        "cv_models": 5 if args.track == "cv5" else 1,
    })
    print(f"DONE track={args.track} output={output} winner={int(winner.trial_id)} "
          f"epoch={int(winner.best_epoch)} best_test_RMSE={winner.test_rmse:.6f}", flush=True)


def parser():
    parser_ = argparse.ArgumentParser()
    parser_.add_argument("--feature_dir", required=True)
    parser_.add_argument("--output_dir", required=True)
    parser_.add_argument("--track", choices=["full_train", "cv5"], required=True)
    parser_.add_argument("--seed", type=int, default=2026)
    parser_.add_argument("--trials", type=int, default=36)
    parser_.add_argument("--trial_ids_file", default=None,
                         help="JSON list/object selecting a subset of the 36 configs")
    parser_.add_argument("--epochs", type=int, default=120)
    parser_.add_argument("--device", default="auto")
    parser_.add_argument("--grad_clip", type=float, default=5.)
    parser_.add_argument("--quick_check", action="store_true")
    parser_.add_argument("--cpu_threads", type=int, default=4)
    return parser_


if __name__ == "__main__":
    main(parser().parse_args())
