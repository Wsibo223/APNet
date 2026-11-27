#!/usr/bin/env python3
"""Evaluate saved AEE, PAE and Ridge models without fitting."""
import argparse
import os
from pathlib import Path

from train import HERE, export_args, stage, file_hash
from fusion import score_test, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--compat-root", type=Path)
    p.add_argument("--delta-root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--exports", type=Path, help="Optional existing AEE and PAE Test exports")
    a = p.parse_args()
    run = a.run.resolve(strict=True)
    model = (run / "fusion/fusion_model.json").resolve(strict=True)
    out = a.out.resolve()
    if a.exports is None:
        if a.compat_root is None or a.delta_root is None:
            p.error("Supply both --compat-root and --delta-root, or --exports")
        inputs = {"aee": a.compat_root, "pae": a.delta_root}
        paths = {k: (run / f"{k}_training/{k}_best.pth").resolve(strict=True) for k in inputs}
        manifests = {k: {"test": (v / "test_2019.txt").resolve(strict=True)} for k, v in inputs.items()}
    out.mkdir(parents=True, exist_ok=False)
    if a.exports is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(HERE) + os.pathsep + env.get("PYTHONPATH", "")
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        placeholders = out / "unused_manifest.txt"
        dirs = {k: out / f"{k}_test_export" for k in inputs}
        report = {"stages": {}}
        for k in inputs:
            stage(f"branches.{k}.export", export_args(k, paths[k], manifests[k], dirs[k],
                  out / "export_cache", ("test",), placeholders), out, f"export_{k}", report, env)
    else:
        dirs = {k: a.exports.resolve(strict=True) / f"{k}_test_export" for k in ("aee", "pae")}
    before = file_hash(model)
    result = score_test(dirs, model, set(), output_csv=out / "test_predictions.csv")
    assert file_hash(model) == before
    write_json(out / "test_metrics.json", result)
    print("| Split | n | MAE | RMSE | CCC | PCC |\n|---|---:|---:|---:|---:|---:|")
    print(f"| Test | {result['n']} | {result['mae']:.8f} | {result['rmse']:.8f} | {result['ccc']:.8f} | {result['corr']:.8f} |")
    print("Results:", out)


if __name__ == "__main__":
    main()
