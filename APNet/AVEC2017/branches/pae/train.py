# train_ard_retardation_2017_v5_clean_weaksevere.py
# Clean weak-severe training for the Psychomotor Retardation / Numbness Branch.
# Protocol: train only -> select best on dev -> report test once with best-dev checkpoint.

import os
import json
import math
import time
import argparse
import random
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import torch
from checkpoint import load_checkpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from branches.pae.dataset import SanitizerConfig, PatchConfig, collate_ard_retardation
from branches.pae.dataset_fast import ARDRetardationDataset2017Fast
from branches.pae.model import ARDSevereAwareNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cfg_to_dict(cfg):
    """Serialize dataclass or simple config object safely.

    This function must be defined before main() is called, because the training
    loop saves sanitizer_cfg and patch_cfg whenever a new best-dev checkpoint is found.
    """
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def video_id_from_path(path: str) -> str:
    b = os.path.basename(path)
    for suf in ["_AUs.csv", "_pose.csv", "_gaze.csv", ".csv"]:
        if b.endswith(suf):
            return b[:-len(suf)]
    return b


def read_split_table(txt_path: str) -> pd.DataFrame:
    rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p, s = line.strip().split()[:2]
            rows.append({"video_id": video_id_from_path(p), "score": float(s), "path": p})
    return pd.DataFrame(rows)


def bin_name(y: float) -> str:
    if y < 6:
        return "minimal_0_5"
    if y < 12:
        return "mild_6_11"
    if y < 18:
        return "moderate_12_17"
    return "severe_18_24"


def ccc_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) == 0:
        return float("nan")
    mx, my = np.mean(y_pred), np.mean(y_true)
    vx, vy = np.var(y_pred), np.var(y_true)
    cov = np.mean((y_pred - mx) * (y_true - my))
    return float((2.0 * cov) / (vx + vy + (mx - my) ** 2 + 1e-8))


def metrics_np(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) == 0:
        return {"n": 0, "rmse": None, "mae": None, "ccc": None, "corr": None,
                "pred_mean": None, "pred_std": None, "gt_mean": None, "gt_std": None, "bias": None}
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    ccc = ccc_np(y_true, y_pred)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 and np.std(y_pred) > 1e-8 else 0.0
    return {
        "n": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "ccc": ccc,
        "corr": corr,
        "pred_mean": float(np.mean(y_pred)),
        "pred_std": float(np.std(y_pred, ddof=1)) if len(y_pred) > 1 else 0.0,
        "gt_mean": float(np.mean(y_true)),
        "gt_std": float(np.std(y_true, ddof=1)) if len(y_true) > 1 else 0.0,
        "bias": float(np.mean(y_pred - y_true)),
    }


def bin_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if len(df) == 0:
        return pd.DataFrame(rows)
    total_sse = float(((df["pred"] - df["gt"]) ** 2).sum()) + 1e-8
    for b, g in df.groupby("bin"):
        m = metrics_np(g["gt"].values, g["pred"].values)
        row = {
            "bin": b,
            "count": int(len(g)),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "ccc": m["ccc"],
            "corr": m["corr"],
            "pred_mean": m["pred_mean"],
            "pred_std": m["pred_std"],
            "gt_mean": m["gt_mean"],
            "gt_std": m["gt_std"],
            "bias": m["bias"],
            "mean_abs_error": float(np.mean(np.abs(g["pred"] - g["gt"]))),
            "sse_contribution": float(((g["pred"] - g["gt"]) ** 2).sum() / total_sse),
        }
        for c in ["base_pred", "severe_residual", "severe_prob", "score_topk", "cls_pred", "pred_reg"]:
            if c in g.columns:
                row[f"mean_{c}"] = float(g[c].mean())
                row[f"std_{c}"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        order = ["minimal_0_5", "mild_6_11", "moderate_12_17", "severe_18_24"]
        out["bin"] = pd.Categorical(out["bin"], categories=order, ordered=True)
        out = out.sort_values("bin")
    return out


def ccc_loss_torch(pred, target):
    x = pred.view(-1)
    y = target.view(-1)
    xm, ym = x.mean(), y.mean()
    cov = torch.mean((x - xm) * (y - ym))
    denom = torch.var(x, unbiased=False) + torch.var(y, unbiased=False) + (xm - ym) ** 2 + 1e-8
    return 1.0 - (2.0 * cov) / denom


def get_out(out: dict, *names, default=None):
    for n in names:
        if isinstance(out, dict) and n in out:
            return out[n]
    return default


def build_configs(args):
    san = SanitizerConfig(
        target_hz=args.target_hz,
        conf_th=args.conf_th,
        median_kernel=args.median_kernel,
        smooth_kernel=args.smooth_kernel,
        ema_alpha=args.ema_alpha,
        deadband_pose=args.deadband_pose,
        use_gaze=args.use_gaze,
    )
    patch = PatchConfig(
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    return san, patch


def build_dataset(txt_path, san, patch, args, split_name: str, verbose=True):
    return ARDRetardationDataset2017Fast(
        txt_path,
        san,
        patch,
        args.cache_root,
        use_cache=(not args.no_cache),
        max_tokens=args.max_tokens,
        verbose=verbose,
    )


def compute_and_set_norm(train_ds, *other_datasets):
    mean, std = train_ds.compute_feature_norm()
    train_ds.set_feature_norm(mean, std)
    for ds in other_datasets:
        ds.set_feature_norm(mean, std)
    return mean, std


def get_dataset_scores(ds, txt_path):
    for attr in ["scores", "labels", "video_scores", "ys"]:
        if hasattr(ds, attr):
            try:
                arr = np.asarray(getattr(ds, attr), dtype=np.float64).reshape(-1)
                if len(arr) == len(ds):
                    return arr
            except Exception:
                pass
    tbl = read_split_table(txt_path)
    arr = tbl["score"].values.astype(np.float64)
    if len(arr) != len(ds):
        print(f"[WARN] score length mismatch: txt={len(arr)} dataset={len(ds)}; truncating.")
        arr = arr[:min(len(arr), len(ds))]
    return arr


def make_sampler(ds, txt_path, args):
    if not args.use_weighted_sampler:
        return None
    scores = get_dataset_scores(ds, txt_path)
    weights = []
    for y in scores:
        w = 1.0
        if y >= 18:
            w *= args.severe_oversample
        elif y >= 12:
            w *= args.moderate_oversample
        elif y >= 6:
            w *= args.mild_oversample
        weights.append(w)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(ds):
        print("[WARN] Weighted sampler disabled due to length mismatch.")
        return None
    print("[Sampler] weights | "
          f"min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f} | "
          f"severe={args.severe_oversample}, moderate={args.moderate_oversample}")
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def compute_loss(out, target, args):
    pred = get_out(out, "pred", "y_pred")
    if pred is None:
        raise KeyError("Model output must contain 'pred' or 'y_pred'.")
    target = target.view(-1).float()
    pred = pred.view(-1).float()

    weight = torch.ones_like(target)
    weight = torch.where(target >= 6.0, weight * args.loss_mild_weight, weight)
    weight = torch.where(target >= 12.0, weight * args.loss_moderate_weight, weight)
    weight = torch.where(target >= 18.0, weight * args.loss_severe_weight, weight)
    weight = weight / weight.mean().clamp_min(1e-6)

    huber_each = F.smooth_l1_loss(pred, target, beta=args.huber_beta, reduction="none")
    l_huber = (huber_each * weight).mean()
    l_mse = (((pred - target) ** 2) * weight).mean()
    l_ccc = ccc_loss_torch(pred, target) if pred.numel() > 1 else pred.new_tensor(0.0)
    loss = args.w_huber * l_huber + args.w_mse * l_mse + args.w_ccc * l_ccc

    stats = {
        "loss_huber": float(l_huber.detach().cpu()),
        "loss_mse": float(l_mse.detach().cpu()),
        "loss_ccc": float(l_ccc.detach().cpu()),
    }

    severe_prob = get_out(out, "severe_prob", "prob_severe", default=None)
    severe_res = get_out(out, "severe_residual", "residual_severe", default=None)

    if severe_prob is not None and args.w_severe_bce > 0:
        severe_label = (target >= args.severe_threshold).float()
        prob = severe_prob.view(-1).float().clamp(1e-5, 1.0 - 1e-5)
        l_bce = F.binary_cross_entropy(prob, severe_label)
        loss = loss + args.w_severe_bce * l_bce
        stats["loss_severe_bce"] = float(l_bce.detach().cpu())

    if severe_res is not None:
        res = severe_res.view(-1).float()
        non_mask = (target < args.severe_threshold)
        sev_mask = (target >= args.severe_threshold)
        if non_mask.any() and args.w_res_nonsevere > 0:
            l_res_non = torch.mean(res[non_mask] ** 2)
            loss = loss + args.w_res_nonsevere * l_res_non
            stats["loss_res_nonsevere"] = float(l_res_non.detach().cpu())
        if sev_mask.any() and args.w_res_severe > 0:
            l_res_sev = torch.mean((res[sev_mask] - args.severe_res_target) ** 2)
            loss = loss + args.w_res_severe * l_res_sev
            stats["loss_res_severe"] = float(l_res_sev.detach().cpu())
        if args.w_res_global > 0:
            l_res_global = torch.mean(res ** 2)
            loss = loss + args.w_res_global * l_res_global
            stats["loss_res_global"] = float(l_res_global.detach().cpu())

    if args.w_tolerance > 0:
        tol_err = torch.relu(torch.abs(pred - target) - args.tolerance_margin)
        l_tol = torch.mean(tol_err ** 2)
        loss = loss + args.w_tolerance * l_tol
        stats["loss_tolerance"] = float(l_tol.detach().cpu())

    stats["loss_total"] = float(loss.detach().cpu())
    stats["batch_pred_mean"] = float(pred.detach().cpu().mean())
    stats["batch_pred_std"] = float(pred.detach().cpu().std(unbiased=False)) if pred.numel() > 1 else 0.0
    return loss, stats


def grad_norm(module):
    total = 0.0
    count = 0
    if module is None:
        return 0.0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).cpu())
        count += 1
    return float(math.sqrt(total)) if count > 0 else 0.0


def collect_grad_stats(model):
    candidates = {
        "grad_input_proj": ["input_proj", "embed", "token_embed", "encoder_in"],
        "grad_encoder": ["encoder", "temporal", "transformer"],
        "grad_token_score": ["token_score", "score_head", "evidence_score"],
        "grad_token_attn": ["token_attn", "attn_head", "attention"],
        "grad_video_head": ["video_head", "head", "regressor"],
        "grad_offset_head": ["offset_head", "residual_head", "severe_residual_head"],
        "grad_severe_head": ["severe_head", "severe_prob_head", "severity_head"],
        "grad_prototypes": ["prototypes", "prototype"],
    }
    modules = dict(model.named_modules())
    out = {}
    for out_name, keys in candidates.items():
        value = 0.0
        used = False
        for name, mod in modules.items():
            lname = name.lower()
            if any(k.lower() in lname for k in keys):
                value += grad_norm(mod)
                used = True
        out[out_name] = float(value) if used else 0.0
    out["grad_total"] = grad_norm(model)
    return out


@torch.no_grad()
def evaluate(model, loader, split_name: str):
    model.eval()
    rows = []
    for batch in tqdm(loader, desc=f"eval {split_name}", ncols=100, leave=False):
        tokens = batch["tokens"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)
        score = batch["score"].to(DEVICE).view(-1).float()
        out = model(tokens, mask)
        pred = get_out(out, "pred", "y_pred").view(-1).detach().cpu().numpy()
        gt = score.detach().cpu().numpy()

        def arr_or_none(x):
            if x is None:
                return None
            return x.detach().cpu().numpy().reshape(len(pred), -1)

        base_arr = arr_or_none(get_out(out, "base_pred", "base", default=None))
        res_arr = arr_or_none(get_out(out, "severe_residual", "residual_severe", default=None))
        prob_arr = arr_or_none(get_out(out, "severe_prob", "prob_severe", default=None))
        topk_arr = arr_or_none(get_out(out, "score_topk", "topk_score", default=None))
        video_ids = batch.get("video_ids", [str(i) for i in range(len(pred))])

        for i in range(len(pred)):
            row = {
                "split": split_name,
                "video_id": video_ids[i],
                "gt": float(gt[i]),
                "pred": float(pred[i]),
                "abs_error": float(abs(pred[i] - gt[i])),
                "sq_error": float((pred[i] - gt[i]) ** 2),
                "bin": bin_name(float(gt[i])),
            }
            if base_arr is not None:
                row["base_pred"] = float(base_arr[i].reshape(-1)[0])
            if res_arr is not None:
                row["severe_residual"] = float(res_arr[i].reshape(-1)[0])
            if prob_arr is not None:
                row["severe_prob"] = float(prob_arr[i].reshape(-1)[0])
            if topk_arr is not None:
                row["score_topk"] = float(topk_arr[i].reshape(-1)[0])
            rows.append(row)
    df = pd.DataFrame(rows)
    return metrics_np(df["gt"].values, df["pred"].values), bin_metrics_df(df), df


def select_score(metrics: dict, args) -> float:
    rmse = metrics["rmse"]
    corr = metrics["corr"] if metrics["corr"] is not None else 0.0
    pred_std = metrics["pred_std"] if metrics["pred_std"] is not None else 0.0
    gt_std = metrics["gt_std"] if metrics["gt_std"] is not None else 1.0
    pred_mean = metrics["pred_mean"] if metrics["pred_mean"] is not None else 0.0
    gt_mean = metrics["gt_mean"] if metrics["gt_mean"] is not None else 0.0
    return float(rmse + args.select_std_lambda * abs(pred_std - gt_std)
                 + args.select_bias_lambda * abs(pred_mean - gt_mean)
                 - args.select_corr_lambda * corr)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "forensics"), exist_ok=True)

    san, patch = build_configs(args)
    print("[Config] Sanitizer:", san)
    print("[Config] Patch:", patch)

    train_ds = build_dataset(args.train_txt, san, patch, args, "train", verbose=True)
    dev_ds = build_dataset(args.dev_txt, san, patch, args, "dev", verbose=True)
    test_ds = build_dataset(args.test_txt, san, patch, args, "test", verbose=True)
    feature_mean, feature_std = compute_and_set_norm(train_ds, dev_ds, test_ds)

    feature_dim = int(getattr(train_ds, "feature_dim", getattr(train_ds, "in_dim", 0)))
    if feature_dim <= 0:
        sample = train_ds[0]
        feature_dim = int(sample["tokens"].shape[-1])
    print(f"[Feature] feature_dim={feature_dim}")

    sampler = make_sampler(train_ds, args.train_txt, args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler,
                              num_workers=args.num_workers, collate_fn=collate_ard_retardation,
                              pin_memory=True, drop_last=False)
    dev_loader = DataLoader(dev_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers_eval,
                            collate_fn=collate_ard_retardation, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers_eval,
                             collate_fn=collate_ard_retardation, pin_memory=True)

    model = ARDSevereAwareNet(
        in_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_severity_bins=4,
        topk_ratio=args.topk_ratio,
        min_topk=args.min_topk,
        severe_topk_ratio=args.severe_topk_ratio,
        severe_min_topk=args.severe_min_topk,
        severe_residual_max=args.severe_residual_max,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history, grad_history = [], []
    best_score, best_dev_rmse, best_epoch, patience = 1e9, 1e9, -1, 0
    best_path = os.path.join(args.save_dir, "pae_best.pth")

    progress_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_rows, epoch_grad_rows = [], []
        pbar = tqdm(train_loader, desc=f"ARD-V5 Ep{epoch}", ncols=120)
        for batch_idx, batch in enumerate(pbar, start=1):
            tokens = batch["tokens"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
            score = batch["score"].to(DEVICE).view(-1).float()
            optimizer.zero_grad(set_to_none=True)
            out = model(tokens, mask)
            loss, loss_stats = compute_loss(out, score, args)
            loss.backward()
            grad_stats = collect_grad_stats(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_stats["epoch"] = epoch
            loss_stats["batch_idx"] = batch_idx
            grad_stats["epoch"] = epoch
            grad_stats["batch_idx"] = batch_idx
            epoch_loss_rows.append(loss_stats)
            epoch_grad_rows.append(grad_stats)
            pbar.set_postfix(loss=f"{loss_stats['loss_total']:.4f}", pred_std=f"{loss_stats['batch_pred_std']:.2f}")

        scheduler.step()
        loss_df_ep = pd.DataFrame(epoch_loss_rows)
        grad_df_ep = pd.DataFrame(epoch_grad_rows)
        train_row = {"epoch": epoch, "lr": float(optimizer.param_groups[0]["lr"])}
        for c in loss_df_ep.columns:
            if c not in ["epoch", "batch_idx"]:
                train_row[f"train_{c}"] = float(loss_df_ep[c].mean())
        for c in grad_df_ep.columns:
            if c not in ["epoch", "batch_idx"]:
                train_row[c] = float(grad_df_ep[c].mean())

        dev_m, dev_bm, dev_df = evaluate(model, dev_loader, "dev")
        cur_score = select_score(dev_m, args)
        row = {**train_row,
               "dev_select_score": cur_score,
               "dev_rmse": dev_m["rmse"],
               "dev_mae": dev_m["mae"],
               "dev_ccc": dev_m["ccc"],
               "dev_corr": dev_m["corr"],
               "dev_pred_mean": dev_m["pred_mean"],
               "dev_pred_std": dev_m["pred_std"],
               "dev_gt_mean": dev_m["gt_mean"],
               "dev_gt_std": dev_m["gt_std"],
               "dev_bias": dev_m["bias"]}
        for _, br in dev_bm.iterrows():
            prefix = str(br["bin"])
            row[f"dev_{prefix}_rmse"] = br["rmse"]
            row[f"dev_{prefix}_mae"] = br["mae"]
            row[f"dev_{prefix}_ccc"] = br["ccc"]
            row[f"dev_{prefix}_corr"] = br["corr"]
            row[f"dev_{prefix}_pred_mean"] = br["pred_mean"]
        history.append(row)
        grad_history.extend(epoch_grad_rows)
        pd.DataFrame(history).to_csv(os.path.join(args.save_dir, "history.csv"), index=False)
        pd.DataFrame(grad_history).to_csv(os.path.join(args.save_dir, "gradient_flow.csv"), index=False)
        print(f"p{epoch} | DEV score={cur_score:.4f} | RMSE={dev_m['rmse']:.4f} | MAE={dev_m['mae']:.4f} | "
              f"CCC={dev_m['ccc']:.4f} | corr={dev_m['corr']:.4f} | pred_std={dev_m['pred_std']:.3f} | bias={dev_m['bias']:.3f}", flush=True)

        dev_df.to_csv(os.path.join(args.save_dir, "forensics", f"dev_predictions_epoch{epoch:03d}.csv"), index=False)
        dev_bm.to_csv(os.path.join(args.save_dir, "forensics", f"dev_bin_metrics_epoch{epoch:03d}.csv"), index=False)

        eta = (time.monotonic() - progress_start) / epoch * (args.epochs - epoch)
        print(f"APNET_PROGRESS Epoch {epoch}/{args.epochs} | train_loss={train_row['train_loss_total']:.4f} | "
              f"lr={train_row['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)
        if cur_score < best_score:
            best_score = cur_score
            best_dev_rmse = dev_m["rmse"]
            best_epoch = epoch
            patience = 0
            torch.save({
                "model_state": model.state_dict(),
                "args": vars(args),
                "feature_dim": feature_dim,
                "feature_mean": feature_mean,
                "feature_std": feature_std,
                "sanitizer_cfg": cfg_to_dict(san),
                "patch_cfg": cfg_to_dict(patch),
                "best_epoch": best_epoch,
                "best_dev_rmse": best_dev_rmse,
                "best_dev_select_score": best_score,
                "protocol": "train_only_dev_select_test_report",
                "checkpoint_type": "ard_retardation_v5_clean_weaksevere",
            }, best_path)
            dev_df.to_csv(os.path.join(args.save_dir, "best_dev_predictions.csv"), index=False)
            dev_bm.to_csv(os.path.join(args.save_dir, "best_dev_bin_metrics.csv"), index=False)
        else:
            patience += 1
        if epoch >= args.min_epochs and patience >= args.early_stop:
            print(f"Early stop at epoch {epoch}.")
            break

    ckpt = load_checkpoint(best_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=True)
    dev_m, dev_bm, dev_df = evaluate(model, dev_loader, "dev")
    test_m, test_bm, test_df = evaluate(model, test_loader, "test")
    dev_df.to_csv(os.path.join(args.save_dir, "final_dev_predictions.csv"), index=False)
    test_df.to_csv(os.path.join(args.save_dir, "final_test_predictions.csv"), index=False)
    dev_bm.to_csv(os.path.join(args.save_dir, "final_dev_bin_metrics.csv"), index=False)
    test_bm.to_csv(os.path.join(args.save_dir, "final_test_bin_metrics.csv"), index=False)
    summary = {
        "protocol": "train_only_dev_select_test_report",
        "best_epoch": int(best_epoch),
        "best_dev_select_score": float(best_score),
        "best_dev_rmse": float(best_dev_rmse),
        "dev": {"all": dev_m, "bins": dev_bm.to_dict(orient="records")},
        "test": {"all": test_m, "bins": test_bm.to_dict(orient="records")},
        "recommend_checkpoint": best_path,
        "seed": args.seed,
        "notes": {
            "selection": "Dev-only score = RMSE + std penalty + bias penalty - corr reward. Test is not used during training.",
            "weak_severe": "Severe oversampling and residual loss are intentionally weakened for train-only clean protocol."
        }
    }
    with open(os.path.join(args.save_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFinal TEST bin metrics:")
    print(test_bm.to_string(index=False))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", type=str, default="train_2017.txt")
    p.add_argument("--dev_txt", type=str, default="dev_2017.txt")
    p.add_argument("--test_txt", type=str, default="test_2017.txt")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--cache_root", type=str, default="cache_ard_retardation_2017_v5_clean")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--use_gaze", action="store_true", default=False)
    p.add_argument("--target_hz", type=float, default=10.0)
    p.add_argument("--conf_th", type=float, default=0.75)
    p.add_argument("--median_kernel", type=int, default=5)
    p.add_argument("--smooth_kernel", type=int, default=5)
    p.add_argument("--ema_alpha", type=float, default=0.65)
    p.add_argument("--deadband_pose", type=float, default=1e-3)
    p.add_argument("--window_sec", type=float, default=4.0)
    p.add_argument("--stride_sec", type=float, default=2.0)
    p.add_argument("--use_fft", action="store_true", default=False)  # kept for CLI compatibility; PatchConfig in this codebase does not accept it
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--topk_ratio", type=float, default=0.10)
    p.add_argument("--min_topk", type=int, default=3)
    p.add_argument("--severe_topk_ratio", type=float, default=0.15)
    p.add_argument("--severe_min_topk", type=int, default=4)
    p.add_argument("--severe_residual_max", type=float, default=5.0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--min_epochs", type=int, default=10)
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_workers_eval", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--use_weighted_sampler", action="store_true", default=True)
    p.add_argument("--no_weighted_sampler", action="store_false", dest="use_weighted_sampler")
    p.add_argument("--mild_oversample", type=float, default=1.0)
    p.add_argument("--moderate_oversample", type=float, default=1.5)
    p.add_argument("--severe_oversample", type=float, default=2.0)
    p.add_argument("--huber_beta", type=float, default=1.0)
    p.add_argument("--w_huber", type=float, default=0.85)
    p.add_argument("--w_mse", type=float, default=0.10)
    p.add_argument("--w_ccc", type=float, default=0.05)
    p.add_argument("--loss_mild_weight", type=float, default=1.0)
    p.add_argument("--loss_moderate_weight", type=float, default=1.10)
    p.add_argument("--loss_severe_weight", type=float, default=1.20)
    p.add_argument("--severe_threshold", type=float, default=18.0)
    p.add_argument("--w_severe_bce", type=float, default=0.05)
    p.add_argument("--w_res_nonsevere", type=float, default=0.20)
    p.add_argument("--w_res_severe", type=float, default=0.01)
    p.add_argument("--w_res_global", type=float, default=0.01)
    p.add_argument("--severe_res_target", type=float, default=2.5)
    p.add_argument("--w_tolerance", type=float, default=0.0)
    p.add_argument("--tolerance_margin", type=float, default=1.5)
    p.add_argument("--select_std_lambda", type=float, default=0.15)
    p.add_argument("--select_bias_lambda", type=float, default=0.10)
    p.add_argument("--select_corr_lambda", type=float, default=0.30)
    return p


if __name__ == "__main__":
    main(parser().parse_args())

