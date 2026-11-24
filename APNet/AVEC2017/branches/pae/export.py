# branches/pae/export.py
# Export train/dev/test predictions from the old ARD retardation V5 clean weak-severe checkpoint.
#
# This script does NOT retrain the model. It only:
#   1) imports the original training script,
#   2) rebuilds the dataset with checkpoint preprocessing configs,
#   3) loads pae_best.pth,
#   4) runs model.eval() on train/dev/test,
#   5) saves train_predictions.csv / dev_predictions.csv / test_predictions.csv.
#
# Usage:
#   cd ~/cv_project/2017
#   python branches/pae/export.py \
#     --script_module branches.pae.train \
#     --ckpt pae_training/pae_best.pth \
#     --out_dir fusion_inputs_2017/numb_v5

import os
import json
import inspect
import argparse
import importlib
from types import SimpleNamespace
from dataclasses import make_dataclass, field

import numpy as np
import pandas as pd
import torch
from checkpoint import load_checkpoint
from torch.utils.data import DataLoader


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def dict_to_dataclass_config(d, class_name):
    """Convert a checkpoint config dict into a dataclass instance.

    The original dataset calls dataclasses.asdict(cfg) and also accesses cfg.xxx.
    SimpleNamespace supports cfg.xxx but fails asdict(cfg), so we build a lightweight
    dataclass dynamically with the checkpoint keys.
    """
    if not isinstance(d, dict):
        return d
    fields = []
    values = {}
    for k, v in d.items():
        vv = dict_to_dataclass_config(v, f"{class_name}_{k}") if isinstance(v, dict) else v
        values[k] = vv
        fields.append((k, type(vv), field(default=vv)))
    Cfg = make_dataclass(class_name, fields)
    return Cfg(**values)


def safe_metrics_from_df(df):
    if not {"gt", "pred"}.issubset(df.columns):
        return {}
    y = df["gt"].values.astype(float)
    p = df["pred"].values.astype(float)
    corr = 0.0 if len(y) < 2 or np.std(y) < 1e-8 or np.std(p) < 1e-8 else float(np.corrcoef(y, p)[0, 1])
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "mae": float(np.mean(np.abs(p - y))),
        "corr": corr,
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
        "gt_mean": float(np.mean(y)),
        "gt_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
    }


def construct_model(mod, ckpt, ckpt_args):
    if not hasattr(mod, "ARDSevereAwareNet"):
        raise AttributeError(
            f"{mod.__name__} does not expose ARDSevereAwareNet. "
            "Run grep -n \"class .*nn.Module\\|ARDSevereAwareNet\" on the training script."
        )

    cls = mod.ARDSevereAwareNet
    sig = inspect.signature(cls)
    params = sig.parameters

    # Common name mapping. This is deliberately broad to tolerate minor version differences.
    value_map = {
        "feature_dim": ckpt.get("feature_dim"),
        "input_dim": ckpt.get("feature_dim"),
        "in_dim": ckpt.get("feature_dim"),
        "feat_dim": ckpt.get("feature_dim"),
        "hidden_dim": getattr(ckpt_args, "hidden_dim", 64),
        "hid": getattr(ckpt_args, "hidden_dim", 64),
        "num_layers": getattr(ckpt_args, "num_layers", 1),
        "n_layers": getattr(ckpt_args, "num_layers", 1),
        "dropout": getattr(ckpt_args, "dropout", 0.10),
        "topk_ratio": getattr(ckpt_args, "topk_ratio", 0.20),
        "min_topk": getattr(ckpt_args, "min_topk", 1),
        "severe_topk_ratio": getattr(ckpt_args, "severe_topk_ratio", 0.20),
        "severe_min_topk": getattr(ckpt_args, "severe_min_topk", 1),
        "severe_residual_max": getattr(ckpt_args, "severe_residual_max", 8.0),
        "severe_threshold": getattr(ckpt_args, "severe_threshold", 18.0),
    }

    kwargs = {}
    missing_required = []
    for name, p in params.items():
        if name == "self":
            continue
        if name in value_map and value_map[name] is not None:
            kwargs[name] = value_map[name]
        elif p.default is inspect._empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            missing_required.append(name)

    print("[Model signature]", sig)
    print("[Model kwargs]", kwargs)
    if missing_required:
        print("[WARN] Missing required constructor args:", missing_required)

    try:
        return cls(**kwargs)
    except TypeError as e:
        print("[WARN] Keyword construction failed:", repr(e))
        # Fallback for the most common signature:
        # ARDSevereAwareNet(feature_dim, hidden_dim, num_layers, dropout, topk_ratio, min_topk,
        #                   severe_topk_ratio, severe_min_topk, severe_residual_max)
        common = [
            ckpt.get("feature_dim"),
            getattr(ckpt_args, "hidden_dim", 64),
            getattr(ckpt_args, "num_layers", 1),
            getattr(ckpt_args, "dropout", 0.10),
            getattr(ckpt_args, "topk_ratio", 0.20),
            getattr(ckpt_args, "min_topk", 1),
            getattr(ckpt_args, "severe_topk_ratio", 0.20),
            getattr(ckpt_args, "severe_min_topk", 1),
            getattr(ckpt_args, "severe_residual_max", 8.0),
        ]
        try:
            return cls(*common)
        except TypeError as e2:
            raise TypeError(
                "Could not construct ARDSevereAwareNet automatically.\n"
                f"Signature: {sig}\n"
                f"Keyword error: {e}\n"
                f"Positional error: {e2}\n"
                "Please paste the ARDSevereAwareNet class __init__ signature."
            )


def build_loader_for_split(mod, txt_path, ckpt, ckpt_args, split_name, batch_size, num_workers):
    # The original dataset expects config objects that support both:
    #   cfg.xxx attribute access
    #   dataclasses.asdict(cfg)
    # Checkpoints store them as plain dicts, so convert dict -> lightweight dataclass.
    san = dict_to_dataclass_config(ckpt["sanitizer_cfg"], "SanitizerCfgFromCheckpoint")
    patch = dict_to_dataclass_config(ckpt["patch_cfg"], "PatchCfgFromCheckpoint")

    ds = mod.build_dataset(txt_path, san, patch, ckpt_args, split_name, verbose=True)

    # In V5 scripts, normalization is usually set after dataset construction.
    if hasattr(ds, "set_feature_norm"):
        ds.set_feature_norm(ckpt["feature_mean"], ckpt["feature_std"])
    else:
        print("[WARN] Dataset has no set_feature_norm; assuming build_dataset already handles normalization.")

    if not hasattr(mod, "collate_ard_retardation"):
        raise AttributeError(f"{mod.__name__} has no collate_ard_retardation")

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=mod.collate_ard_retardation,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, loader


def run_eval(mod, model, loader, split_name):
    model.eval()
    with torch.no_grad():
        out = mod.evaluate(model, loader, split_name)

    if isinstance(out, tuple) and len(out) == 3:
        m, bm, df = out
    elif isinstance(out, tuple) and len(out) == 2:
        m, df = out
        bm = None
    else:
        raise RuntimeError(f"Unexpected evaluate output type/len: {type(out)}")

    return m, bm, df


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print("[Device]", device)

    mod = importlib.import_module(args.script_module)
    if hasattr(mod, "DEVICE"):
        mod.DEVICE = device

    ckpt = load_checkpoint(args.ckpt, map_location=device)
    ckpt_args = SimpleNamespace(**ckpt["args"])

    # Override split paths if user provides them.
    ckpt_args.train_txt = args.train_txt
    ckpt_args.dev_txt = args.dev_txt
    ckpt_args.test_txt = args.test_txt
    ckpt_args.save_dir = args.out_dir
    if args.cache_root:
        ckpt_args.cache_root = args.cache_root
    if args.no_cache:
        ckpt_args.no_cache = True

    print("[Checkpoint]", args.ckpt)
    print("[Checkpoint type]", ckpt.get("checkpoint_type", ""))
    print("[Best epoch]", ckpt.get("best_epoch", None))
    print("[Feature dim]", ckpt.get("feature_dim", None))

    model = construct_model(mod, ckpt, ckpt_args).to(device)
    state = ckpt.get("model_state", ckpt.get("state_dict", None))
    if state is None:
        raise KeyError("Checkpoint has neither model_state nor state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[load_state_dict] missing:", missing)
    print("[load_state_dict] unexpected:", unexpected)

    split_paths = {
        "train": ckpt_args.train_txt,
        "dev": ckpt_args.dev_txt,
        "test": ckpt_args.test_txt,
    }

    summary = {
        "ckpt": args.ckpt,
        "script_module": args.script_module,
        "checkpoint_type": ckpt.get("checkpoint_type", ""),
        "best_epoch": ckpt.get("best_epoch", None),
        "splits": {},
    }

    for split, txt_path in split_paths.items():
        print(f"\n========== Export {split}: {txt_path} ==========")
        _, loader = build_loader_for_split(
            mod,
            txt_path,
            ckpt,
            ckpt_args,
            split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        m, bm, df = run_eval(mod, model, loader, split)

        # Normalize common column names for fusion.
        if "prediction" in df.columns and "pred" not in df.columns:
            df = df.rename(columns={"prediction": "pred"})
        if "score" in df.columns and "gt" not in df.columns:
            df = df.rename(columns={"score": "gt"})

        out_csv = os.path.join(args.out_dir, f"{split}_predictions.csv")
        df.to_csv(out_csv, index=False)
        if bm is not None:
            try:
                bm.to_csv(os.path.join(args.out_dir, f"{split}_bin_metrics.csv"), index=False)
            except Exception:
                pass

        mm = m if isinstance(m, dict) else safe_metrics_from_df(df)
        if not isinstance(mm, dict) or "rmse" not in mm:
            mm = safe_metrics_from_df(df)
        summary["splits"][split] = {"metrics": to_jsonable(mm), "csv": out_csv}
        print(f"[{split}] saved: {out_csv}")
        print(f"[{split}] metrics:", json.dumps(to_jsonable(summary["splits"][split]["metrics"]), indent=2))

    with open(os.path.join(args.out_dir, "export_summary.json"), "w", encoding="utf-8") as f:
        json.dump(to_jsonable(summary), f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--script_module", default="branches.pae.train")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--train_txt", default="train_2017.txt")
    p.add_argument("--dev_txt", default="dev_2017.txt")
    p.add_argument("--test_txt", default="test_2017.txt")
    p.add_argument("--cache_root", default="")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
