"""Strict pair-grouped AVEC2014 training for the full AVEC2017 PRE topology."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from pose_data import (
    NormalizedSubset, PRE2014Corpus, PatchConfig, SanitizerConfig,
    collate_pre, compute_norm, discover_records,
)
from pose_model import PREFullSyncNet


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ccc(y_true, y_pred) -> float:
    y = np.asarray(y_true, np.float64)
    p = np.asarray(y_pred, np.float64)
    if len(y) < 2:
        return 0.0
    covariance = np.mean((y - y.mean()) * (p - p.mean()))
    denominator = y.var() + p.var() + (y.mean() - p.mean()) ** 2 + 1e-8
    return float(2.0 * covariance / denominator)


def metrics(y_true, y_pred) -> Dict:
    y = np.asarray(y_true, np.float64)
    p = np.asarray(y_pred, np.float64)
    corr = float(np.corrcoef(y, p)[0, 1]) if len(y) > 1 and y.std() > 1e-8 and p.std() > 1e-8 else 0.0
    return {
        "n": int(len(y)), "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))), "ccc": ccc(y, p), "corr": corr,
        "pred_mean": float(p.mean()), "pred_std": float(p.std(ddof=1)) if len(p) > 1 else 0.0,
        "gt_mean": float(y.mean()), "gt_std": float(y.std(ddof=1)) if len(y) > 1 else 0.0,
        "bias": float(np.mean(p - y)),
    }


def bdi_bin(score: float) -> int:
    if score <= 13:
        return 0
    if score <= 19:
        return 1
    if score <= 28:
        return 2
    return 3


def bdi_bin_name(score: float) -> str:
    return ("minimal_0_13", "mild_14_19", "moderate_20_28", "severe_29_plus")[bdi_bin(score)]


def paired_frame(clip_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    if (clip_df.groupby("pair_id")["gt"].nunique() > 1).any():
        raise ValueError("Inconsistent labels across the two views of a pair")
    aggregation = {"gt": "first", pred_col: "mean", "view": "nunique"}
    for column in [
        "base_pred", "pred_full", "severe_residual", "severe_prob", "severe_bag_logit",
        "score_topk", "attn_peak", "retard_conf", "pred_std_folds",
    ]:
        if column in clip_df.columns and column != pred_col:
            aggregation[column] = "mean"
    out = clip_df.groupby("pair_id", as_index=False).agg(aggregation)
    return out.rename(columns={"view": "n_views"}).sort_values("pair_id").reset_index(drop=True)


def metric_table(clip_df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    rows = []
    paired = paired_frame(clip_df, pred_col)
    rows.append({"scope": "paired_mean", **metrics(paired["gt"], paired[pred_col])})
    rows.append({"scope": "clips_pooled", **metrics(clip_df["gt"], clip_df[pred_col])})
    for view in ("Freeform", "Northwind"):
        subset = clip_df.loc[clip_df["view"] == view]
        rows.append({"scope": view, **metrics(subset["gt"], subset[pred_col])})
    return pd.DataFrame(rows)


def make_pair_folds(corpus, train_indices, n_splits, seed):
    pair_table = pd.DataFrame([
        {"pair_id": corpus.samples[i]["pair_id"], "score": corpus.samples[i]["score"]}
        for i in train_indices
    ]).drop_duplicates("pair_id").sort_values("pair_id").reset_index(drop=True)
    labels = pair_table["score"].map(bdi_bin).to_numpy()
    counts = np.bincount(labels, minlength=4)
    if np.all(counts[counts > 0] >= n_splits):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        iterator = splitter.split(pair_table["pair_id"], labels)
        method = "bdi_clinical_bin_stratified"
    else:
        quantiles = pd.qcut(pair_table["score"].rank(method="first"), q=min(4, len(pair_table)), labels=False)
        q_counts = np.bincount(np.asarray(quantiles, int))
        if q_counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(pair_table["pair_id"], quantiles)
            method = "train_quantile_stratified"
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            iterator = splitter.split(pair_table["pair_id"])
            method = "group_kfold_fallback"
    indices_by_pair: Dict[str, List[int]] = {}
    for index in train_indices:
        indices_by_pair.setdefault(corpus.samples[index]["pair_id"], []).append(index)
    folds = []
    for fold, (fit_pos, val_pos) in enumerate(iterator):
        fit_pairs = sorted(pair_table.iloc[fit_pos]["pair_id"].tolist())
        val_pairs = sorted(pair_table.iloc[val_pos]["pair_id"].tolist())
        assert not set(fit_pairs).intersection(val_pairs)
        fit_indices = sorted(i for pair in fit_pairs for i in indices_by_pair[pair])
        val_indices = sorted(i for pair in val_pairs for i in indices_by_pair[pair])
        folds.append((fold, fit_indices, val_indices, fit_pairs, val_pairs))
    return folds, method


def ccc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_tensor(0.0)
    covariance = torch.mean((pred - pred.mean()) * (target - target.mean()))
    denominator = pred.var(unbiased=False) + target.var(unbiased=False) + (pred.mean() - target.mean()).square() + 1e-8
    return 1.0 - 2.0 * covariance / denominator


def bdi_to_model(value, score_max: float):
    return value * (24.0 / score_max)


def model_to_bdi(value, score_max: float):
    return value * (score_max / 24.0)


def module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    found = False
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        total += float(parameter.grad.detach().square().sum().cpu())
        found = True
    return float(math.sqrt(total)) if found else 0.0


def collect_grad_stats(model: PREFullSyncNet) -> Dict[str, float]:
    """Track the same failure-prone model regions as the final 2017 trainer."""
    return {
        "grad_total": module_grad_norm(model),
        "grad_input_proj": module_grad_norm(model.input_proj),
        "grad_temporal": module_grad_norm(model.temporal),
        "grad_token_score": module_grad_norm(model.token_score),
        "grad_token_attn": module_grad_norm(model.token_attn),
        "grad_video_proj": module_grad_norm(model.video_proj),
        "grad_reg_head": module_grad_norm(model.reg_head),
        "grad_severe_token_head": module_grad_norm(model.severe_token_head),
        "grad_severe_residual_head": module_grad_norm(model.severe_residual_head),
    }


def compute_loss(output: Dict, target_bdi: torch.Tensor, args) -> Tuple[torch.Tensor, Dict]:
    target = bdi_to_model(target_bdi, args.bdi_score_max)
    pred = output["pred"].view(-1)
    weights = torch.ones_like(target)
    weights = torch.where(target_bdi >= 14.0, weights * args.loss_mild_weight, weights)
    weights = torch.where(target_bdi >= 20.0, weights * args.loss_moderate_weight, weights)
    weights = torch.where(target_bdi >= args.severe_threshold_bdi, weights * args.loss_severe_weight, weights)
    weights = weights / weights.mean().clamp_min(1e-6)

    huber_each = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
    huber = (huber_each * weights).mean()
    mse = (((pred - target).square()) * weights).mean()
    ccc_term = ccc_loss(pred, target)
    total = args.w_huber * huber + args.w_mse * mse + args.w_ccc * ccc_term

    severe_label = (target_bdi >= args.severe_threshold_bdi).float()
    probability = output["severe_prob"].view(-1).clamp(1e-5, 1.0 - 1e-5)
    severe_bce = F.binary_cross_entropy(probability, severe_label)
    total = total + args.w_severe_bce * severe_bce

    residual = output["severe_residual"].view(-1)
    non_mask = target_bdi < args.severe_threshold_bdi
    severe_mask = ~non_mask
    res_non = residual.new_tensor(0.0)
    res_severe = residual.new_tensor(0.0)
    if non_mask.any():
        res_non = residual[non_mask].square().mean()
        total = total + args.w_res_nonsevere * res_non
    if severe_mask.any():
        res_severe = (residual[severe_mask] - args.severe_res_target).square().mean()
        total = total + args.w_res_severe * res_severe
    res_global = residual.square().mean()
    total = total + args.w_res_global * res_global

    stats = {
        "total": float(total.detach().cpu()), "huber": float(huber.detach().cpu()),
        "mse": float(mse.detach().cpu()), "ccc_loss": float(ccc_term.detach().cpu()),
        "severe_bce": float(severe_bce.detach().cpu()),
        "res_nonsevere": float(res_non.detach().cpu()),
        "res_severe": float(res_severe.detach().cpu()), "res_global": float(res_global.detach().cpu()),
        "pred_std": float(pred.detach().std(unbiased=False).cpu()) if pred.numel() > 1 else 0.0,
    }
    return total, stats


def make_loader(dataset, batch_size, shuffle, workers, sampler=None):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(shuffle and sampler is None), sampler=sampler,
        num_workers=workers, collate_fn=collate_pre, pin_memory=torch.cuda.is_available(), drop_last=False,
    )


def make_sampler(corpus, indices, args):
    weights = []
    for index in indices:
        score = corpus.samples[index]["score"]
        if score >= args.severe_threshold_bdi:
            weight = args.severe_oversample
        elif score >= 20:
            weight = args.moderate_oversample
        elif score >= 14:
            weight = args.mild_oversample
        else:
            weight = 1.0
        weights.append(weight)
    return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


@torch.no_grad()
def predict(model, loader, device, score_max: float, fold: int) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in loader:
        output = model(batch["tokens"].to(device), batch["mask"].to(device))
        arrays = {
            key: output[key].detach().cpu().numpy().reshape(-1)
            for key in [
                "pred", "base_pred", "severe_residual", "severe_prob", "severe_bag_logit",
                "score_topk", "attn_peak", "retard_conf",
            ]
        }
        gt = batch["score"].numpy()
        for i in range(len(gt)):
            base_bdi = float(model_to_bdi(arrays["base_pred"][i], score_max))
            residual_bdi = float(model_to_bdi(arrays["severe_residual"][i], score_max))
            rows.append({
                "fold": fold, "split": batch["split"][i], "pair_id": batch["pair_id"][i],
                "view": batch["view"][i], "clip_id": batch["clip_id"][i], "gt": float(gt[i]),
                "base_pred": base_bdi, "severe_residual": residual_bdi,
                "pred_full": base_bdi + residual_bdi,
                "severe_prob": float(arrays["severe_prob"][i]),
                "severe_bag_logit": float(arrays["severe_bag_logit"][i]),
                "score_topk": float(arrays["score_topk"][i]),
                "attn_peak": float(arrays["attn_peak"][i]),
                "retard_conf": float(arrays["retard_conf"][i]),
            })
    return pd.DataFrame(rows)


def select_score(metric_row: Dict, args) -> float:
    scale = 24.0 / args.bdi_score_max
    return float(
        metric_row["rmse"] * scale
        + args.select_std_lambda * abs(metric_row["pred_std"] - metric_row["gt_std"]) * scale
        + args.select_bias_lambda * abs(metric_row["bias"]) * scale
        - args.select_corr_lambda * metric_row["corr"]
    )


def train_fold(fold, corpus, fit_indices, val_indices, args, device, fold_dir):
    set_seed(args.seed + fold)
    feature_mean, feature_std = compute_norm(corpus, fit_indices)
    fit_ds = NormalizedSubset(corpus, fit_indices, feature_mean, feature_std)
    val_ds = NormalizedSubset(corpus, val_indices, feature_mean, feature_std)
    sampler = make_sampler(corpus, fit_indices, args) if args.use_weighted_sampler else None
    fit_loader = make_loader(fit_ds, args.batch_size, True, args.num_workers, sampler)
    val_loader = make_loader(val_ds, args.eval_batch_size, False, args.num_workers)
    model = PREFullSyncNet(
        in_dim=corpus.feature_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        dropout=args.dropout, topk_ratio=args.topk_ratio, min_topk=args.min_topk,
        severe_topk_ratio=args.severe_topk_ratio, severe_min_topk=args.severe_min_topk,
        severe_residual_max=args.severe_residual_max,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_score, best_epoch, patience = math.inf, -1, 0
    best_path = fold_dir / "best_model.pt"
    history, gradient_history = [], []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_index, batch in enumerate(
            tqdm(fit_loader, desc=f"fold{fold} epoch{epoch}", ncols=100, leave=False), start=1
        ):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch["tokens"].to(device), batch["mask"].to(device))
            loss, loss_stats = compute_loss(output, batch["score"].to(device), args)
            loss.backward()
            grad_stats = collect_grad_stats(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(loss_stats)
            gradient_history.append({"epoch": epoch, "batch": batch_index, **grad_stats})
        scheduler.step()
        val_df = predict(model, val_loader, device, args.bdi_score_max, fold)
        paired = paired_frame(val_df, "pred_full")
        val_metric = metrics(paired["gt"], paired["pred_full"])
        current_score = select_score(val_metric, args)
        row = {
            "epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"]),
            **{f"train_{key}": float(np.mean([x[key] for x in losses])) for key in losses[0]},
            "val_select_score": current_score,
            **{f"val_paired_{key}": value for key, value in val_metric.items()},
            "val_residual_mean": float(val_df["severe_residual"].mean()),
            "val_severe_prob_mean": float(val_df["severe_prob"].mean()),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(fold_dir / "history.csv", index=False)
        pd.DataFrame(gradient_history).to_csv(fold_dir / "gradient_flow.csv", index=False)
        if current_score < best_score - args.min_delta:
            best_score, best_epoch, patience = current_score, epoch, 0
            torch.save({
                "model_state": model.state_dict(), "feature_mean": feature_mean, "feature_std": feature_std,
                "best_epoch": best_epoch, "best_select_score": best_score, "fold": fold,
                "seed": args.seed + fold, "args": vars(args),
                "protocol": "fullsync_train_group_oof_select__dev_test_ensemble_report",
            }, best_path)
        else:
            patience += 1
        elapsed = time.time() - started
        eta = elapsed / epoch * max(0, args.epochs - epoch)
        print(
            f"fold={fold} epoch={epoch} select={current_score:.4f} "
            f"RMSE={val_metric['rmse']:.4f} CCC={val_metric['ccc']:.4f} "
            f"res={val_df['severe_residual'].mean():.3f} prob={val_df['severe_prob'].mean():.3f} "
            f"best={best_score:.4f}@{best_epoch} ETA_max={eta/60:.1f}m", flush=True,
        )
        print(f"APNET_PROGRESS Fold {fold + 1}/{args.n_splits} | Epoch {epoch}/{args.epochs} | "
              f"train_loss={row['train_total']:.4f} | lr={row['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)
        if epoch >= args.min_epochs and patience >= args.early_stop:
            print(f"fold={fold} early_stop epoch={epoch}", flush=True)
            break

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    val_df = predict(model, val_loader, device, args.bdi_score_max, fold)
    val_df.to_csv(fold_dir / "oof_clip_predictions.csv", index=False)
    return model, feature_mean, feature_std, val_df


def ensemble_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    joined = pd.concat(frames, ignore_index=True)
    keys = ["split", "pair_id", "view", "clip_id", "gt"]
    value_cols = [
        "base_pred", "severe_residual", "pred_full", "severe_prob", "severe_bag_logit",
        "score_topk", "attn_peak", "retard_conf",
    ]
    aggregation = {column: "mean" for column in value_cols}
    aggregation.update({"fold": "nunique"})
    out = joined.groupby(keys, as_index=False).agg(aggregation).rename(columns={"fold": "n_models"})
    pred_spread = joined.groupby(keys)["pred_full"].std().reset_index(name="pred_std_folds")
    out = out.merge(pred_spread, on=keys, how="left")
    out["pred_std_folds"] = out["pred_std_folds"].fillna(0.0)
    if not (out["n_models"] == len(frames)).all():
        raise AssertionError("Incomplete fold ensemble")
    return out.sort_values(["pair_id", "view"]).reset_index(drop=True)


def choose_residual_alpha(dev: pd.DataFrame, alphas: Sequence[float]) -> Tuple[float, pd.DataFrame]:
    rows = []
    for alpha in alphas:
        candidate = dev.copy()
        candidate["candidate_pred"] = candidate["base_pred"] + alpha * candidate["severe_residual"]
        paired = paired_frame(candidate, "candidate_pred")
        rows.append({"alpha": alpha, **metrics(paired["gt"], paired["candidate_pred"])})
    table = pd.DataFrame(rows).sort_values(["rmse", "alpha"]).reset_index(drop=True)
    return float(table.iloc[0]["alpha"]), table


def save_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, dict):
            return {str(k): convert(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(v) for v in item]
        if isinstance(item, np.generic):
            return item.item()
        return item
    with path.open("w", encoding="utf-8") as handle:
        json.dump(convert(value), handle, indent=2, ensure_ascii=False)


def load_checkpoint(path: Path, device):
    """Support both old PyTorch (no weights_only kwarg) and new defaults."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> List[str]:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_report(path, summary, candidate_table):
    lines = [
        "# AVEC2014 PRE-P1 FullSync report", "", "## Protocol", "",
        "- Architecture: complete final AVEC2017 severe-aware PRE topology.",
        "- Input: pose-only 71-D descriptors at 10 Hz, 4 s windows, 2 s stride.",
        "- Score adaptation: BDI-II is linearly mapped to the AVEC2017 0-24 optimization scale.",
        "- Split: pair-grouped train OOF; dev selects residual alpha; test is final-only.", "",
        "## Dev residual selection", "",
        *markdown_table(candidate_table), "",
        f"Selected residual alpha: **{summary['selected_residual_alpha']:.2f}**", "",
        "## Primary results", "",
        "| Split | Scope | n | RMSE | MAE | CCC | Corr | Pred mean | Pred std |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train_oof", "dev", "test"):
        for scope, item in summary[split].items():
            lines.append(
                f"| {split} | {scope} | {int(item['n'])} | {item['rmse']:.4f} | {item['mae']:.4f} | "
                f"{item['ccc']:.4f} | {item['corr']:.4f} | {item['pred_mean']:.4f} | {item['pred_std']:.4f} |"
            )
    lines.extend([
        "", "## Interpretation rule", "",
        "`selected_pred` is the only test result eligible for comparison. Raw base/full columns are retained for transparent diagnostics.",
        "Run `scripts/04_run_forensics.sh` after training to identify the failing component if performance is weak.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args):
    set_seed(args.seed)
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    records = discover_records(args.data_root, args.labels_csv, strict=True)
    records.to_csv(output / "resolved_manifest.csv", index=False)
    sanitizer = SanitizerConfig(
        target_hz=args.target_hz, conf_th=args.conf_th, median_kernel=args.median_kernel,
        smooth_kernel=args.smooth_kernel, ema_alpha=args.ema_alpha, deadband_pose=args.deadband_pose,
    )
    patch = PatchConfig(window_sec=args.window_sec, stride_sec=args.stride_sec, max_tokens=args.max_tokens)
    corpus = PRE2014Corpus(records, args.cache_root, sanitizer, patch, use_cache=not args.no_cache)
    if corpus.feature_dim != 71:
        raise AssertionError(f"Expected 71-D descriptors, got {corpus.feature_dim}")
    quality = corpus.quality_frame()
    quality.to_csv(output / "data_quality.csv", index=False)
    train_indices = corpus.indices_for_split("train")
    dev_indices = corpus.indices_for_split("dev")
    test_indices = corpus.indices_for_split("test")
    folds, fold_method = make_pair_folds(corpus, train_indices, args.n_splits, args.seed)

    oof_frames, dev_frames, test_frames, fold_manifest = [], [], [], []
    for fold, fit_idx, val_idx, fit_pairs, val_pairs in folds:
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_manifest.append({
            "fold": fold, "fit_pairs": fit_pairs, "val_pairs": val_pairs,
            "n_fit_clips": len(fit_idx), "n_val_clips": len(val_idx),
        })
        model, mean, std, oof_df = train_fold(fold, corpus, fit_idx, val_idx, args, device, fold_dir)
        oof_frames.append(oof_df)
        dev_ds = NormalizedSubset(corpus, dev_indices, mean, std)
        test_ds = NormalizedSubset(corpus, test_indices, mean, std)
        dev_df = predict(model, make_loader(dev_ds, args.eval_batch_size, False, args.num_workers),
                         device, args.bdi_score_max, fold)
        test_df = predict(model, make_loader(test_ds, args.eval_batch_size, False, args.num_workers),
                          device, args.bdi_score_max, fold)
        dev_df.to_csv(fold_dir / "dev_clip_predictions.csv", index=False)
        test_df.to_csv(fold_dir / "test_clip_predictions_unscored.csv", index=False)
        dev_frames.append(dev_df)
        test_frames.append(test_df)

    save_json(output / "fold_manifest.json", {"method": fold_method, "folds": fold_manifest})
    oof = pd.concat(oof_frames, ignore_index=True).sort_values(["pair_id", "view"]).reset_index(drop=True)
    if oof["clip_id"].nunique() != len(train_indices):
        raise AssertionError("OOF coverage mismatch")
    dev = ensemble_frames(dev_frames)
    test = ensemble_frames(test_frames)
    alpha_grid = [float(x) for x in args.residual_alpha_grid.split(",")]
    selected_alpha, candidates = choose_residual_alpha(dev, alpha_grid)
    candidates.to_csv(output / "dev_residual_alpha_candidates.csv", index=False)
    for frame in (oof, dev, test):
        frame["selected_pred"] = frame["base_pred"] + selected_alpha * frame["severe_residual"]

    for name, frame in (("train_oof", oof), ("dev_ensemble", dev), ("test_ensemble", test)):
        frame.to_csv(output / f"{name}_clip_predictions.csv", index=False)
        paired_frame(frame, "selected_pred").to_csv(output / f"{name}_paired_predictions.csv", index=False)
        metric_table(frame, "selected_pred").to_csv(output / f"{name}_metrics_by_view.csv", index=False)

    summary = {
        "protocol": "fullsync_train_group_oof__dev_select_residual_alpha__test_final_only",
        "architecture": "final_AVEC2017_PRE_severe_aware_topology",
        "score_adaptation": f"BDI-II x (24/{args.bdi_score_max}) during optimization",
        "selected_residual_alpha": selected_alpha, "fold_method": fold_method,
        "seed": args.seed, "feature_dim": corpus.feature_dim,
        "sanitizer": asdict(sanitizer), "patch": asdict(patch),
    }
    for split, frame in (("train_oof", oof), ("dev", dev), ("test", test)):
        table = metric_table(frame, "selected_pred")
        summary[split] = {row.scope: row._asdict() for row in table.itertuples(index=False)}
    save_json(output / "summary.json", summary)
    write_report(output / "report.md", summary, candidates)
    print((output / "report.md").read_text(encoding="utf-8"))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/AVEC2014")
    p.add_argument("--labels_csv", default="data/AVEC2014/avec2014_labels.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--cache_root", default="cache_pre2014_pose71")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--target_hz", type=float, default=10.0)
    p.add_argument("--conf_th", type=float, default=0.75)
    p.add_argument("--median_kernel", type=int, default=5)
    p.add_argument("--smooth_kernel", type=int, default=5)
    p.add_argument("--ema_alpha", type=float, default=0.65)
    p.add_argument("--deadband_pose", type=float, default=1e-3)
    p.add_argument("--window_sec", type=float, default=4.0)
    p.add_argument("--stride_sec", type=float, default=2.0)
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--topk_ratio", type=float, default=0.10)
    p.add_argument("--min_topk", type=int, default=3)
    p.add_argument("--severe_topk_ratio", type=float, default=0.15)
    p.add_argument("--severe_min_topk", type=int, default=4)
    p.add_argument("--severe_residual_max", type=float, default=5.0)
    p.add_argument("--bdi_score_max", type=float, default=64.0)
    p.add_argument("--severe_threshold_bdi", type=float, default=29.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--w_huber", type=float, default=0.85)
    p.add_argument("--w_mse", type=float, default=0.10)
    p.add_argument("--w_ccc", type=float, default=0.05)
    p.add_argument("--loss_mild_weight", type=float, default=1.0)
    p.add_argument("--loss_moderate_weight", type=float, default=1.10)
    p.add_argument("--loss_severe_weight", type=float, default=1.20)
    p.add_argument("--w_severe_bce", type=float, default=0.05)
    p.add_argument("--w_res_nonsevere", type=float, default=0.20)
    p.add_argument("--w_res_severe", type=float, default=0.01)
    p.add_argument("--w_res_global", type=float, default=0.01)
    p.add_argument("--severe_res_target", type=float, default=2.5)
    p.add_argument("--use_weighted_sampler", action="store_true", default=True)
    p.add_argument("--no_weighted_sampler", action="store_false", dest="use_weighted_sampler")
    p.add_argument("--mild_oversample", type=float, default=1.0)
    p.add_argument("--moderate_oversample", type=float, default=1.5)
    p.add_argument("--severe_oversample", type=float, default=2.0)
    p.add_argument("--select_std_lambda", type=float, default=0.15)
    p.add_argument("--select_bias_lambda", type=float, default=0.10)
    p.add_argument("--select_corr_lambda", type=float, default=0.30)
    p.add_argument("--residual_alpha_grid", default="0,0.25,0.5,0.75,1.0")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
