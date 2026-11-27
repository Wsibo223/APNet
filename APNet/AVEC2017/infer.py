#!/usr/bin/env python3
"""Evaluate fixed AVEC2017 branch and fusion checkpoints without fitting."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from console_progress import run_stage, print_test_metrics
from fusion import MLPResidual, load_split, metrics_np, standardize_apply

HERE = Path(__file__).resolve().parent


def predict(export_root, checkpoint, output, device):
    try:
        frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # older PyTorch
        frozen = torch.load(checkpoint, map_location="cpu")
    if frozen.get("format_version") != 1:
        raise ValueError("Expected a fusion_model.pth saved by this release")
    opts = SimpleNamespace(expr_test="", numb_test="", **frozen["input_options"],
                           expr_dir=export_root / "aee_predictions", numb_dir=export_root / "pae_predictions",
                           expr_embed_dir=export_root / "aee_embeddings", numb_embed_dir=export_root / "pae_embeddings")
    test, _ = load_split(opts, "test")
    if len(test) != 47 or test.video_id.nunique() != 47:
        raise ValueError("Expected 47 aligned AVEC2017 Test participants")
    st = frozen["standardizer"]
    st = {**st, "mu": np.asarray(st["mu"], np.float32), "sd": np.asarray(st["sd"], np.float32)}
    x = torch.tensor(standardize_apply(test, st), device=device, dtype=torch.float32)
    expr, numb = [torch.tensor(test[c].values.astype(np.float32), device=device)
                  for c in ("expr_pred", "numb_pred")]
    model = MLPResidual(**frozen["model_kwargs"]).to(device)
    model.load_state_dict(frozen["model_state"], strict=True)
    model.eval()
    with torch.no_grad():
        pred = torch.clamp(model(x, expr, numb), 0, 24).cpu().numpy()
    test[["video_id", "gt"]].assign(pred=pred).to_csv(output / "test_predictions.csv", mode="x", index=False)
    return metrics_np(test["gt"], pred)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path, help="Directory containing saved branch and fusion models")
    p.add_argument("--splits", type=Path, help="Prepared AVEC2017 split lists")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--exports", type=Path, help="Optional existing branch prediction and embedding directories")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    run = a.run.resolve(strict=True)
    cp = (run / "fusion/fusion_model.pth").resolve(strict=True)
    out = a.out.resolve()
    if a.exports is None and a.splits is None:
        p.error("Supply --splits or --exports")
    if a.exports is None:
        ac = (run / "aee_training/aee_best.pth").resolve(strict=True)
        pc = (run / "pae_training/pae_best.pth").resolve(strict=True)
        manifests = [v for s in ("train", "dev", "test") for v in
                     (f"--{s}_txt", (a.splits / f"{s}_2017.txt").resolve(strict=True))]
    out.mkdir(parents=True, exist_ok=False)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if a.exports is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        def stage(name, module, *args):
            run_stage(name, [sys.executable, "-B", "-u", "-m", module, *map(str, args)], out, env)
        cache = out / "pae_cache"
        stage("aee_export", "branches.aee.export", *manifests, "--ckpt", ac,
              "--out_dir", out / "aee_predictions", "--module_name", "branches.aee.train")
        stage("pae_export", "branches.pae.export", *manifests, "--ckpt", pc,
              "--out_dir", out / "pae_predictions", "--cache_root", cache, "--script_module", "branches.pae.train")
        stage("embeddings", "export_embeddings", *manifests, "--branch", "both", "--seed", 2026,
              "--expr_script", HERE / "branches/aee/train.py", "--expr_ckpt", ac,
              "--expr_out_dir", out / "aee_embeddings", "--numb_train_script", HERE / "branches/pae/train.py",
              "--numb_ckpt", pc, "--numb_out_dir", out / "pae_embeddings", "--numb_cache_root", cache, "--batch_size", 8)
    result = predict(a.exports.resolve(strict=True) if a.exports else out, cp, out, device)
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print_test_metrics(result)
    print("Results:", out)


if __name__ == "__main__":
    main()
