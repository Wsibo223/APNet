#!/usr/bin/env python3
"""Train AVEC2017 AEE, PAE and the fixed residual fusion head."""
import argparse
import json
import os
from pathlib import Path
import sys
from console_progress import run_stage, print_test_metrics

HERE = Path(__file__).resolve().parent

def config(name):
    return json.loads((HERE / "configs" / (name + ".json")).read_text())

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pae-config", type=Path,
                   help="Optional PAE argument JSON override; defaults to configs/pae.json.")
    p.add_argument("--allow-default-pae", action="store_true",
                   help="Train with script defaults if exact historical PAE arguments are unavailable.")
    a = p.parse_args()
    manifests = []
    for split in ("train", "dev", "test"):
        manifests += [f"--{split}_txt", str((a.splits / f"{split}_2017.txt").resolve(strict=True))]
    recipe = json.loads(a.pae_config.read_text()) if a.pae_config else config("pae")
    if recipe.get("status") != "historical_checkpoint_args" and not a.allow_default_pae:
        p.error("PAE configuration must have historical_checkpoint_args status; use --allow-default-pae only for an unverified run.")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "pae_config.json").write_text(json.dumps(recipe, indent=2))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "4")
    def stage(name, target, *args):
        run_stage(name, [sys.executable, "-B", "-u", "-m", target,
                         *map(str, args)], out, env)
    aee, pae = out / "aee_training", out / "pae_training"
    cache = out / "pae_cache"
    ac = aee / "aee_best.pth"
    pc = pae / "pae_best.pth"
    ae, pe = out / "aee_predictions", out / "pae_predictions"
    av, pv = out / "aee_embeddings", out / "pae_embeddings"
    stage("aee", "branches.aee.train", *config("aee")["args"], *manifests,
          "--save_dir", aee, "--seed", 2026)
    stage("pae", "branches.pae.train", *recipe["args"], *manifests,
          "--save_dir", pae, "--cache_root", cache, "--seed", 2026)
    stage("aee_export", "branches.aee.export", *manifests, "--ckpt", ac,
          "--out_dir", ae, "--module_name", "branches.aee.train")
    stage("pae_export", "branches.pae.export", *manifests, "--ckpt", pc,
          "--out_dir", pe, "--cache_root", cache, "--script_module", "branches.pae.train")
    stage("embeddings", "export_embeddings", *manifests, "--branch", "both", "--seed", 2026,
          "--expr_script", HERE / "branches/aee/train.py", "--expr_ckpt", ac, "--expr_out_dir", av,
          "--numb_train_script", HERE / "branches/pae/train.py", "--numb_ckpt", pc,
          "--numb_out_dir", pv, "--numb_cache_root", cache, "--batch_size", 8)
    stage("fusion", "fusion", *config("fusion")["args"], "--seed", 2026,
          "--expr_dir", ae, "--numb_dir", pe, "--expr_embed_dir", av,
          "--numb_embed_dir", pv, "--save_dir", out / "fusion")
    summary = json.loads((out / "fusion/ngrc_summary.json").read_text())
    print_test_metrics(summary["test_metrics"])
    print("Results:", out)

if __name__ == "__main__":
    main()
