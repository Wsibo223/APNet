#!/usr/bin/env python3
"""Export PAE predictions and embeddings from a specified checkpoint."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import train as tr
from .dataset import collate_ard_retardation
from .model import ARDSevereAwareNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_ckpt(run_dir):
    path = Path(run_dir) / "pae_best.pth"
    if not path.is_file():
        raise FileNotFoundError(f"Missing PAE checkpoint: {path}")
    return path


def build_model(ckpt, ns):
    feature_dim = int(ckpt.get("feature_dim", getattr(ns, "feature_dim", 0)))
    kwargs = dict(
        hidden_dim=int(ns.hidden_dim),
        num_layers=int(ns.num_layers),
        dropout=float(ns.dropout),
        topk_ratio=float(ns.topk_ratio),
        min_topk=int(ns.min_topk),
        severe_topk_ratio=float(ns.severe_topk_ratio),
        severe_min_topk=int(ns.severe_min_topk),
        severe_residual_max=float(ns.severe_residual_max),
    )
    last = None
    for maker in [
        lambda: ARDSevereAwareNet(feature_dim=feature_dim, **kwargs),
        lambda: ARDSevereAwareNet(input_dim=feature_dim, **kwargs),
        lambda: ARDSevereAwareNet(feature_dim, **kwargs),
    ]:
        try:
            model = maker().to(DEVICE)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.eval()
            return model
        except TypeError as e:
            last = e
    raise last


def build_dataset(txt, ckpt, ns, cache_root, split):
    ns2 = SimpleNamespace(**vars(ns))
    ns2.cache_root = str(cache_root)
    ns2.no_cache = False
    ns2.num_workers = 0
    ns2.num_workers_eval = 0
    ns2.eval_batch_size = int(getattr(ns2, "eval_batch_size", 8))
    san, patch = tr.build_configs(ns2)
    ds = tr.build_dataset(str(txt), san, patch, ns2, split, verbose=True)

    # Critical: use training normalization from ckpt.
    if "feature_mean" in ckpt and "feature_std" in ckpt:
        fm = np.asarray(ckpt["feature_mean"], dtype=np.float32)
        fs = np.asarray(ckpt["feature_std"], dtype=np.float32)
        for n in ["feature_mean", "feature_mu", "mean", "feat_mean"]:
            try: setattr(ds, n, fm)
            except Exception: pass
        for n in ["feature_std", "feature_sigma", "std", "feat_std"]:
            try: setattr(ds, n, fs)
            except Exception: pass
    return ds


@torch.no_grad()
def export_split(model, ckpt, ns, txt, cache_root, split, out_csv):
    ds = build_dataset(txt, ckpt, ns, cache_root, split)
    loader = DataLoader(ds, batch_size=int(getattr(ns, "eval_batch_size", 8)), shuffle=False,
                        num_workers=0, collate_fn=collate_ard_retardation)

    rows = []
    for batch in loader:
        tokens = batch["tokens"].to(DEVICE).float()
        mask = batch["mask"].to(DEVICE)
        out = model(tokens, mask)
        pred = out["pred"].detach().cpu().numpy().reshape(-1)
        gt = batch["score"].detach().cpu().numpy().reshape(-1)

        vids = batch.get("video_ids", None)
        if vids is None:
            vids = [str(len(rows) + i) for i in range(len(pred))]

        emb = out.get("embedding", None)
        emb_np = emb.detach().cpu().numpy() if emb is not None else None

        for i in range(len(pred)):
            row = {
                "video_id": str(vids[i]),
                "gt": float(gt[i]),
                "pred": float(pred[i]),
            }
            for key in ["base_pred", "severe_residual", "severe_prob", "severe_bag_logit", "score_topk", "retard_conf"]:
                if key in out:
                    arr = out[key].detach().cpu().numpy().reshape(-1)
                    row[key] = float(arr[i])
            if emb_np is not None:
                for j, val in enumerate(emb_np[i].reshape(-1)):
                    row[f"emb_{j:03d}"] = float(val)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[EXPORT] {split}: {out_csv} shape={df.shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--train_txt", required=True)
    ap.add_argument("--dev_txt", required=True)
    ap.add_argument("--test_txt", required=True)
    ap.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--cache_root", default="cache_pae_export")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = find_ckpt(args.run_dir)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    ns = SimpleNamespace(**ckpt["args"])
    ns.eval_batch_size = int(getattr(ns, "eval_batch_size", 8))
    ns.batch_size = int(getattr(ns, "batch_size", 8))
    ns.num_workers = 0
    ns.num_workers_eval = 0

    model = build_model(ckpt, ns)

    meta = {
        "run_dir": str(args.run_dir),
        "ckpt": str(ckpt_path),
        "source_args": ckpt.get("args", {}),
    }
    with open(out_dir / "export_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for split in args.splits:
        txt = getattr(args, split + "_txt")
        export_split(
            model=model,
            ckpt=ckpt,
            ns=ns,
            txt=txt,
            cache_root=Path(args.cache_root) / out_dir.name / split,
            split=split,
            out_csv=out_dir / f"{split}_predictions.csv",
        )


if __name__ == "__main__":
    main()
