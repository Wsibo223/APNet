#!/usr/bin/env python3
"""Build one-row-per-pair P16 features and raw 71-D PRE token sequences.

The locked AEE-V4 branch supplies the AEE anchor and latent context. PRE-P1 is
used only for its published preprocessing configuration and baseline score.
P16 trains a capacity-scaled temporal encoder directly from unnormalised
71-D pose descriptors. P2/P3/P5 outputs and teacher targets are forbidden.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import pandas as pd

from temporal_common import (
    assert_aligned, atomic_csv, load_torch, locate, save_json,
    validate_feature_table, validate_pair_table,
)


OUTPUT_SPLITS = ("train_fit", "dev", "test")
BRANCH_SPLITS = ("train_oof", "dev", "test")
SOURCE_OF = {"train_fit": "train_oof", "dev": "dev", "test": "test"}
SCORE_COLUMNS = [
    "aee_pred", "pre_pred", "mean_pred", "pred_min", "pred_max",
    "pred_gap", "pred_abs_gap", "pred_prod", "aee_sq", "pre_sq",
]


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _pair_view_diagnostics(frame, view_column, prefix, split):
    frame = frame.copy()
    frame["pair_id"] = frame["pair_id"].astype(str)
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns
               if c not in {"gt", "fold"}]
    rows = []
    for pair_id, group in frame.groupby("pair_id", sort=True):
        if set(group[view_column].astype(str)) != {"Freeform", "Northwind"}:
            raise AssertionError(f"{split}/{pair_id}: incomplete views")
        ff = group[group[view_column].astype(str) == "Freeform"].iloc[0]
        nw = group[group[view_column].astype(str) == "Northwind"].iloc[0]
        row = {"pair_id": str(pair_id), "gt": float(ff["gt"])}
        for column in numeric:
            left, right = float(ff[column]), float(nw[column])
            row[f"{prefix}ff_{column}"] = left
            row[f"{prefix}nw_{column}"] = right
            row[f"{prefix}mean_{column}"] = .5 * (left + right)
            row[f"{prefix}gap_{column}"] = abs(left - right)
        rows.append(row)
    return validate_pair_table(pd.DataFrame(rows), split)


def _numeric_pair_diagnostics(frame, prefix):
    excluded = {"gt", "fold", "n_views", "selected_pred", "anchor_pred", "pred"}
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns
               if c not in excluded]
    return frame[["pair_id"] + numeric].rename(
        columns={c: f"{prefix}{c}" for c in numeric})


def score_tables(bundle_root: Path) -> Dict[str, pd.DataFrame]:
    aee_root = locate(bundle_root, "v4_safe_oof_learned_seed2026")
    pre_root = locate(bundle_root, "pre_p1_fullsync_seed2026")
    result = {}
    for output_split in OUTPUT_SPLITS:
        split = SOURCE_OF[output_split]
        aee_pair_name = {
            "train_oof": "train_oof_paired_predictions.csv",
            "dev": "dev_paired_predictions.csv", "test": "test_paired_predictions.csv"}[split]
        aee_clip_name = {
            "train_oof": "train_oof_clip_predictions.csv",
            "dev": "dev_clip_predictions.csv", "test": "test_clip_predictions.csv"}[split]
        pre_pair_name = {
            "train_oof": "train_oof_paired_predictions.csv",
            "dev": "dev_ensemble_paired_predictions.csv",
            "test": "test_ensemble_paired_predictions.csv"}[split]
        pre_clip_name = {
            "train_oof": "train_oof_clip_predictions.csv",
            "dev": "dev_ensemble_clip_predictions.csv",
            "test": "test_ensemble_clip_predictions.csv"}[split]
        aee = validate_pair_table(pd.read_csv(aee_root / aee_pair_name), output_split)
        pre = validate_pair_table(pd.read_csv(pre_root / pre_pair_name), output_split)
        assert_aligned(aee, pre, f"AEE/PRE {output_split}")
        frame = aee[["pair_id", "gt"]].copy()
        frame["aee_pred"] = aee["anchor_pred"].to_numpy(float)
        frame["pre_pred"] = pre["selected_pred"].to_numpy(float)
        frame["mean_pred"] = .5 * (frame["aee_pred"] + frame["pre_pred"])
        frame["pred_min"] = np.minimum(frame["aee_pred"], frame["pre_pred"])
        frame["pred_max"] = np.maximum(frame["aee_pred"], frame["pre_pred"])
        frame["pred_gap"] = frame["aee_pred"] - frame["pre_pred"]
        frame["pred_abs_gap"] = frame["pred_gap"].abs()
        frame["pred_prod"] = frame["aee_pred"] * frame["pre_pred"]
        frame["aee_sq"] = frame["aee_pred"] ** 2
        frame["pre_sq"] = frame["pre_pred"] ** 2
        frame = frame.merge(_numeric_pair_diagnostics(aee, "aee_pair_"), on="pair_id")
        frame = frame.merge(_numeric_pair_diagnostics(pre, "pre_pair_"), on="pair_id")
        frame = frame.merge(_pair_view_diagnostics(
            pd.read_csv(aee_root / aee_clip_name), "task", "aee_view_", output_split
        ).drop(columns="gt"), on="pair_id")
        frame = frame.merge(_pair_view_diagnostics(
            pd.read_csv(pre_root / pre_clip_name), "view", "pre_view_", output_split
        ).drop(columns="gt"), on="pair_id")
        result[output_split] = validate_pair_table(frame, output_split)
    return result


def pairify_aee(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    embedding = [c for c in rows if c.startswith("e") and c[1:].isdigit()]
    diagnostic = [c for c in rows.select_dtypes(include=[np.number]).columns
                  if c not in {"fold", "gt", "branch_pred"} and c not in embedding]
    output = []
    for (fold, pair_id), group in rows.groupby(["fold", "pair_id"], sort=True):
        if set(group["view"].astype(str)) != {"Freeform", "Northwind"}:
            raise AssertionError(f"{split}/{pair_id}: incomplete AEE views")
        ff = group[group["view"].astype(str) == "Freeform"].iloc[0]
        nw = group[group["view"].astype(str) == "Northwind"].iloc[0]
        row = {"pair_id": str(pair_id), "gt": float(ff["gt"]), "branch_fold": int(fold)}
        ff_value = ff[embedding].to_numpy(np.float32)
        nw_value = nw[embedding].to_numpy(np.float32)
        for name, values in (("ff", ff_value), ("nw", nw_value)):
            row.update({f"aeeemb_{name}_{i:03d}": float(v)
                        for i, v in enumerate(values)})
        for column in diagnostic:
            left, right = float(ff[column]), float(nw[column])
            row[f"aeeemb_diag_ff_{column}"] = left
            row[f"aeeemb_diag_nw_{column}"] = right
            row[f"aeeemb_diag_mean_{column}"] = .5 * (left + right)
            row[f"aeeemb_diag_gap_{column}"] = abs(left - right)
        output.append(row)
    return pd.DataFrame(output)


def collapse_aee(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    numeric = [c for c in frame.select_dtypes(include=[np.number]).columns
               if c not in {"gt", "branch_fold"}]
    aggregation = {c: "mean" for c in numeric}
    aggregation["gt"] = "first"
    if split == "train_fit":
        fold_count = frame.groupby("pair_id")["branch_fold"].nunique()
        if not (fold_count == 1).all():
            raise AssertionError("train_fit: each pair must have one AEE OOF fold")
        aggregation["branch_fold"] = "first"
    paired = frame.groupby("pair_id", as_index=False).agg(aggregation)
    if split == "train_fit":
        paired = paired.rename(columns={"branch_fold": "cv_fold"})
    return validate_pair_table(paired, split)


def extract_aee_embeddings(bundle_root: Path, data_root: Path, labels_csv: Path,
                           cache_root: Path, device) -> Dict[str, pd.DataFrame]:
    import torch
    project = Path(__file__).resolve().parents[1] / "branches/aee"
    code = project
    sys.path.insert(0, str(code))
    core = import_module("aee_core", code / "aee_core.py")
    train_mod = import_module("p14_aee_train", code / "train.py")
    build = import_module("p14_aee_manifest", code / "build_manifest.py")
    core.DEVICE = device
    results = locate(bundle_root, "v4_safe_oof_learned_seed2026")
    manifest_dir = cache_root / "aee_manifests"
    if not (manifest_dir / "train_2014.txt").exists():
        manifest_args = SimpleNamespace(labels_csv=str(labels_csv), score_col=None,
                                        split_col=None, task_col=None, key_col=None)
        build.write_outputs(build.attach_labels(
            build.scan_features(data_root), build.read_labels(manifest_args)), manifest_dir)
    metas = {split: core.read_txt(manifest_dir / f"{split}_2014.txt")
             for split in ("train", "dev", "test")}
    first = load_torch(results / "fold_0/model_ema.pth", device)
    first_args = SimpleNamespace(**first["args"])
    first_args.au_columns_list = [c.strip() for c in first_args.au_columns.split(",") if c.strip()]
    arrays, paths = {}, {}
    for split in ("train", "dev", "test"):
        arrays[split], paths[split] = train_mod.load_split(metas[split], first_args, split)
    collected = {split: [] for split in BRANCH_SPLITS}
    for fold in range(5):
        checkpoint = load_torch(results / f"fold_{fold}/model_ema.pth", device)
        args = SimpleNamespace(**checkpoint["args"])
        args.au_columns_list = [c.strip() for c in args.au_columns.split(",") if c.strip()]
        fold_prediction = {
            "train_oof": pd.read_csv(results / f"fold_{fold}/oof_clip_predictions.csv"),
            "dev": pd.read_csv(results / f"fold_{fold}/dev_clip_predictions.csv"),
            "test": pd.read_csv(results / f"fold_{fold}/test_clip_predictions.csv")}
        heldout = set(fold_prediction["train_oof"].pair_id.astype(str))
        fit_ids = [i for i, pair_id in enumerate(metas["train"].pair_id.astype(str))
                   if pair_id not in heldout]
        fit_arrays = [arrays["train"][i] for i in fit_ids]
        fit_meta = metas["train"].iloc[fit_ids].reset_index(drop=True)
        raw_mean, raw_std = core.fit_global_stats(fit_arrays)
        fit_descriptors = np.stack([
            core.make_descriptor(value, raw_mean, raw_std, task)
            for value, task in zip(fit_arrays, fit_meta.task)])
        _, descriptor_mean, descriptor_std = core.standardize(fit_descriptors)
        for split in BRANCH_SPLITS:
            raw_split = "train" if split == "train_oof" else split
            prediction = fold_prediction[split].copy()
            prediction["video_id"] = prediction["video_id"].astype(str)
            diagnostic_columns = [
                c for c in prediction.select_dtypes(include=[np.number]).columns
                if c not in {"fold", "gt", "anchor_pred"}]
            diagnostic_map = prediction.set_index("video_id")[diagnostic_columns].to_dict("index")
            if split == "train_oof":
                ids = [i for i, pair_id in enumerate(metas["train"].pair_id.astype(str))
                       if pair_id in heldout]
            else:
                ids = list(range(len(metas[raw_split])))
            metadata = metas[raw_split].iloc[ids].reset_index(drop=True)
            values = [arrays[raw_split][i] for i in ids]
            source_paths = [paths[raw_split][i] for i in ids]
            anchor_map = prediction.set_index("video_id")["anchor_pred"].to_dict()
            anchors = np.asarray([anchor_map[str(video_id)]
                                  for video_id in metadata.video_id], np.float32)
            videos = core.build_videos(
                metadata, values, source_paths, anchors, raw_mean, raw_std,
                descriptor_mean, descriptor_std, args, f"p14_{split}_fold{fold}")
            model = train_mod.model_from_videos(videos, args)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            capture: List[np.ndarray] = []

            def hook(_module, inputs):
                capture.append(inputs[0].detach().cpu().numpy())

            handle = model.res_head[0].register_forward_pre_hook(hook)
            rows = []
            with torch.no_grad():
                for video_index, video in enumerate(videos):
                    batch = core.collate([core.ExprDataset([video], train=False, args=args)[0]])
                    capture.clear()
                    model(batch["x"].to(device), batch["mask"].to(device),
                          batch["bag_mask"].to(device), batch["desc"].to(device),
                          batch["anchor"].to(device))
                    vector = capture[-1][0]
                    video_id = str(metadata.iloc[video_index].video_id)
                    row = {"fold": fold, "pair_id": str(video.pair_id),
                           "view": str(video.task), "gt": float(video.score),
                           "branch_pred": float(anchors[video_index])}
                    row.update({c: float(v) for c, v in diagnostic_map[video_id].items()})
                    row.update({f"e{i:04d}": float(v) for i, v in enumerate(vector)})
                    rows.append(row)
            handle.remove()
            collected[split].append(pd.DataFrame(rows))
    output = {}
    for output_split in OUTPUT_SPLITS:
        source = SOURCE_OF[output_split]
        clips = pd.concat(collected[source], ignore_index=True)
        output[output_split] = collapse_aee(pairify_aee(clips, output_split), output_split)
    return output


def extract_raw_pre_sequences(bundle_root: Path, data_root: Path, labels_csv: Path,
                              cache_root: Path):
    project = Path(__file__).resolve().parents[1] / "branches/pae"
    sys.path.insert(0, str(project))
    from pose_data import PRE2014Corpus, PatchConfig, SanitizerConfig, discover_records

    results = locate(bundle_root, "pre_p1_fullsync_seed2026")
    first = load_torch(results / "fold_0/best_model.pt", "cpu")
    args = first["args"]
    sanitizer = SanitizerConfig(
        target_hz=args["target_hz"], conf_th=args["conf_th"],
        median_kernel=args["median_kernel"], smooth_kernel=args["smooth_kernel"],
        ema_alpha=args["ema_alpha"], deadband_pose=args["deadband_pose"])
    patch = PatchConfig(window_sec=args["window_sec"], stride_sec=args["stride_sec"],
                        max_tokens=args["max_tokens"])
    records = discover_records(data_root, labels_csv, strict=True)
    corpus = PRE2014Corpus(records, cache_root / "pre_raw_cache", sanitizer, patch,
                          use_cache=True)
    sequence_rows, quality_rows = [], []
    sequence_root = cache_root / "pre_raw_sequences"
    for sample in corpus.samples:
        split = "train_fit" if sample["split"].lower() == "train" else sample["split"].lower()
        if split not in OUTPUT_SPLITS:
            continue
        safe_pair = sample["pair_id"].replace("/", "_")
        safe_view = sample["view"].replace("/", "_")
        path = sequence_root / split / f"{safe_pair}__{safe_view}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, descriptors=sample["tokens"].astype(np.float32))
        sequence_rows.append({
            "split": split, "pair_id": str(sample["pair_id"]), "view": str(sample["view"]),
            "gt": float(sample["score"]), "n_tokens": int(len(sample["tokens"])),
            "descriptor_dim": int(sample["tokens"].shape[1]), "path": str(path.resolve())})
        quality_rows.append({
            "split": split, "pair_id": str(sample["pair_id"]), "view": str(sample["view"]),
            "gt": float(sample["score"]), **{k: float(v) for k, v in sample["quality"].items()}})
    index = pd.DataFrame(sequence_rows)
    if len(index) != 300 or index.duplicated(["split", "pair_id", "view"]).any():
        raise AssertionError(f"expected 300 unique raw PRE sequences, got {len(index)}")
    quality = {}
    quality_frame = pd.DataFrame(quality_rows)
    for split in OUTPUT_SPLITS:
        quality[split] = _pair_view_diagnostics(
            quality_frame[quality_frame.split == split], "view", "quality_", split)
    return index, quality, args


def feature_groups(columns: List[str]):
    groups = {
        "aee_ff": [c for c in columns if c.startswith("aeeemb_ff_")],
        "aee_nw": [c for c in columns if c.startswith("aeeemb_nw_")],
        "aee_evidence": [c for c in columns if c.startswith("aeeemb_diag_")],
        "score_context": [c for c in SCORE_COLUMNS if c in columns],
        "quality_context": [c for c in columns if c.startswith("quality_")],
    }
    evidence = groups["aee_evidence"]
    salient_terms = ("top", "attn", "temporal", "proto", "match", "mass", "segment")
    severe_terms = ("severe", "residual", "base_pred", "pred_full", "gap", "null")
    groups["salient_evidence"] = [c for c in evidence if any(t in c for t in salient_terms)]
    groups["severe_evidence"] = [c for c in evidence if any(t in c for t in severe_terms)]
    if not groups["salient_evidence"]:
        groups["salient_evidence"] = evidence[:min(8, len(evidence))]
    if not groups["severe_evidence"]:
        groups["severe_evidence"] = evidence[-min(8, len(evidence)):]
    for name, values in groups.items():
        if not values:
            raise AssertionError(f"empty P16 feature group: {name}")
    if len(groups["aee_ff"]) != len(groups["aee_nw"]):
        raise AssertionError("AEE Freeform/Northwind dimensions differ")
    return groups


def main(args):
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle = Path(args.branch_bundle).expanduser().resolve()
    if not bundle.is_dir():
        raise NotADirectoryError(bundle)
    import torch
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = Path(args.data_root).expanduser().resolve()
    labels_csv = Path(args.labels_csv).expanduser().resolve()
    cache_root = output / "branch_cache"
    scores = score_tables(bundle)
    print(f"device={device}; [1/2] extracting fold-averaged AEE context", flush=True)
    aee = extract_aee_embeddings(bundle, data_root, labels_csv, cache_root, device)
    print("[2/2] extracting unnormalised 71-D PRE sequences", flush=True)
    sequence_index, quality, pre_args = extract_raw_pre_sequences(
        bundle, data_root, labels_csv, cache_root)
    atomic_csv(sequence_index, output / "pre_raw_sequence_index.csv")
    common_columns = None
    for split in OUTPUT_SPLITS:
        frame = aee[split].merge(scores[split], on=["pair_id", "gt"], how="inner")
        frame = frame.merge(quality[split], on=["pair_id", "gt"], how="inner")
        frame = validate_feature_table(frame, split)
        current = [c for c in frame if c not in {"pair_id", "gt", "cv_fold"}]
        common_columns = current if common_columns is None else [
            c for c in common_columns if c in current]
        atomic_csv(frame, output / f"{split}_features.csv")
    forbidden = [c for c in common_columns
                 if c.startswith(("trial_", "p2", "p3", "p5", "teacher"))]
    if forbidden:
        raise AssertionError(f"external input leaked into P16: {forbidden[:10]}")
    groups = feature_groups(common_columns)
    save_json(output / "feature_manifest.json", {
        "contract": "P16 one-row-per-pair AEE context plus raw PRE 71-D sequences",
        "pair_aware": True, "full_train_pairs": 50,
        "single_final_model": True, "single_shared_pre_stem": True,
        "internal_temporal_scales": 6, "independent_score_heads": 6,
        "sample_adaptive_router": True, "external_teacher_or_distillation": False,
        "p2_p3_p5_inputs": False, "outer_oof_fusion": False,
        "pre_raw_sequence_index": "pre_raw_sequence_index.csv",
        "pre_token_dim": 71, "descriptor_priors_are_label_free": True,
        "preprocessing": {k: pre_args[k] for k in (
            "target_hz", "window_sec", "stride_sec", "max_tokens")},
        "train_rows": 50, "dev_rows": 50, "test_rows": 50,
        "cv_fold_source": "original AEE-V4 held-out fold",
        "feature_columns": common_columns, "groups": groups,
        "dimensions": {name: len(values) for name, values in groups.items()},
        "score_columns": SCORE_COLUMNS,
        "forbidden_external_inputs": ["P2", "P3", "P5", "teacher prediction"],
    })
    print(f"DONE P16 features={output} n_features={len(common_columns)}", flush=True)


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--branch_bundle", required=True,
                   help="Directory containing the freshly trained AEE and PAE outputs")
    p.add_argument("--data_root", default="data/AVEC2014")
    p.add_argument("--labels_csv", default="data/AVEC2014/avec2014_labels.csv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
