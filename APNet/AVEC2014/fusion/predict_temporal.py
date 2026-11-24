#!/usr/bin/env python3
"""Reproduce P16 predictions from the full-train or five-fold artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from temporal_common import atomic_csv, load_torch, validate_feature_table
from historical_search import SequenceBank, ensemble_frames, make_model, predict_rows, prediction_frame


def predict_one(checkpoint, frame, bank, split, device):
    groups = {}
    for name, columns in checkpoint["groups"].items():
        value = frame[columns].to_numpy(np.float32)
        stats = checkpoint["feature_normalization"][name]
        groups[name] = ((value - np.asarray(stats["mean"])) /
                        np.asarray(stats["std"])).astype(np.float32)
    token_mean = np.asarray(checkpoint["token_normalization"]["mean"], np.float32)
    token_std = np.asarray(checkpoint["token_normalization"]["std"], np.float32)
    data = {"groups": groups, "y": frame["gt"].to_numpy(np.float32),
            "aee": frame["aee_pred"].to_numpy(np.float32),
            "pre": frame["pre_pred"].to_numpy(np.float32)}
    data.update(bank.arrays(frame, split, token_mean, token_std))
    model = make_model(checkpoint["config"], checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return prediction_frame(frame, predict_rows(model, data, device))


def main(args):
    artifact = Path(args.artifact_dir).expanduser().resolve()
    frame = validate_feature_table(
        pd.read_csv(Path(args.feature_csv).expanduser().resolve()), args.split)
    bank = SequenceBank(Path(args.sequence_index).expanduser().resolve())
    device = torch.device(args.device if args.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if (artifact / "model.pt").is_file():
        checkpoints = [load_torch(artifact / "model.pt", "cpu")]
    else:
        checkpoints = [load_torch(artifact / f"fold_{fold}/model.pt", "cpu")
                       for fold in range(5)]
    with torch.no_grad():
        frames = [predict_one(checkpoint, frame, bank, args.split, device)
                  for checkpoint in checkpoints]
    result = frames[0] if len(frames) == 1 else ensemble_frames(frames)
    destination = Path(args.output_csv).expanduser().resolve()
    atomic_csv(result, destination)
    print(f"DONE members={len(frames)} output={destination}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", required=True)
    p.add_argument("--feature_csv", required=True)
    p.add_argument("--sequence_index", required=True)
    p.add_argument("--split", choices=["train_fit", "dev", "test"], required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
