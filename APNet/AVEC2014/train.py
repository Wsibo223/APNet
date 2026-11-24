#!/usr/bin/env python3
"""Reinitialize the AVEC2014 upstream branches and final temporal model."""
import argparse
import json
import os
from pathlib import Path
import sys
from console_progress import run_stage, print_test_metrics

HERE = Path(__file__).resolve().parent
def flags(name):
    return json.loads((HERE / "configs" / f"{name}.json").read_text())["args"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    a = p.parse_args()
    data = a.data_root.resolve(strict=True)
    labels = a.labels.resolve(strict=True)
    if any(c.isspace() for c in str(data)):
        p.error("AVEC2014 manifest paths must not contain spaces")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    if a.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    def stage(name, script, *args):
        run_stage(name, [sys.executable, "-B", "-u", str(HERE / script),
                         *map(str, args)], out, env)
    m = out / "manifests"
    upstream = out / "upstream"
    stage("prepare", "branches/aee/build_manifest.py", "--data_root", data,
          "--labels_csv", labels, "--out_dir", m)
    stage("preflight", "branches/aee/preflight.py", "--manifest_dir", m)
    manifest_flags = [v for s in ("train", "dev", "test") for v in (f"--{s}_txt", m / f"{s}_2014.txt")]
    stage("aee", "branches/aee/train.py", *flags("aee"), *manifest_flags, "--save_dir",
          upstream / "v4_safe_oof_learned_seed2026")
    stage("pae", "branches/pae/train.py", *flags("pae"), "--data_root", data,
          "--labels_csv", labels, "--output_dir", upstream / "pre_p1_fullsync_seed2026",
          "--cache_root", out / "pae_cache")
    stage("features", "fusion/extract_features.py", "--branch_bundle", upstream,
          "--data_root", data, "--labels_csv", labels, "--output_dir", out / "features",
          "--device", a.device)
    stage("temporal", "fusion/train.py", "--features", out / "features",
          "--out", out / "full_train/final_model", "--device", a.device)
    stage("calibration", "calibration/predict.py", "--rule", HERE / "configs/calibration.json",
          "--p16_result", out, "--split", "test", "--output_csv", out / "test_predictions.csv")
    # The final rule is a frozen historical calibration, not reselected on Test.
    import pandas as pd
    sys.path.insert(0, str(HERE / "calibration"))
    from calibration_common import metrics
    prediction = pd.read_csv(out / "test_predictions.csv")
    result = metrics(prediction["gt"], prediction["p19_pred"])
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print_test_metrics(result)
    print("Results:", out)

if __name__ == "__main__":
    main()
