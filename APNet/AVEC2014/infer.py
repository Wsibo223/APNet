#!/usr/bin/env python3
"""Evaluate the frozen AVEC2014 final temporal model and calibration."""
import argparse
import json
from pathlib import Path
import sys
import subprocess

import pandas as pd
from console_progress import print_test_metrics

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Completed run containing the temporal checkpoint")
    p.add_argument("--features", type=Path, help="Prepared branch context and raw pose sequence directory; defaults to RUN/features")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    a = p.parse_args()
    run = a.run.resolve(strict=True)
    features = (a.features or run / "features").resolve(strict=True)
    artifact = (run / "full_train/final_model/model.pt").resolve(strict=True).parent
    test = (features / "test_features.csv").resolve(strict=True)
    index = (features / "pre_raw_sequence_index.csv").resolve(strict=True)
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    temporal = out / "temporal_predictions.csv"
    subprocess.run([sys.executable, "-B", str(HERE / "fusion/predict_temporal.py"),
                    "--artifact_dir", str(artifact), "--feature_csv", str(test), "--sequence_index", str(index),
                    "--split", "test", "--output_csv", str(temporal), "--device", a.device], check=True)
    sys.path.insert(0, str(HERE / "calibration"))
    from calibration_common import prediction_frame, metrics
    rule = json.loads((HERE / "configs/calibration.json").read_text())
    pred = prediction_frame(pd.read_csv(temporal), rule.get("config", rule))
    pred.to_csv(out / "test_predictions.csv", mode="x", index=False)
    result = metrics(pred["gt"], pred["p19_pred"])
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print_test_metrics(result)
    print("Results:", out)


if __name__ == "__main__":
    main()
