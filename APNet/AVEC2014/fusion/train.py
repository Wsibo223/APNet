#!/usr/bin/env python3
"""Fit the historically fixed temporal model (trial 23, epoch 4)."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from temporal_common import SPLITS, set_seed, metrics
from historical_search import (load_features, SequenceBank, feature_normalization,
    build_arrays, make_model, train_epoch, snapshot, evaluate, result_row, save_artifact)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    configs = Path(__file__).resolve().parents[1] / "configs"
    config = json.loads((configs / "temporal_model.json").read_text())
    selection = json.loads((configs / "temporal_selection.json").read_text())
    epoch_limit = int(selection["epoch"])
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    if device.type == "cpu":
        torch.set_num_threads(4)
    manifest, tables = load_features(a.features.resolve())
    bank = SequenceBank(a.features / manifest["pre_raw_sequence_index"])
    set_seed(config["seed"])
    group_stats = feature_normalization(tables["train_fit"], manifest["groups"])
    token_stats = bank.fit_normalization(tables["train_fit"])
    arrays = {s: build_arrays(tables[s], s, manifest["groups"], group_stats, bank, token_stats)
              for s in SPLITS}
    model = make_model(config, manifest).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    history = []
    progress_start = time.monotonic()
    for epoch in range(1, epoch_limit + 1):
        losses = train_epoch(model, optimizer, arrays["train_fit"], config, device,
                             SimpleNamespace(grad_clip=5.), epoch)
        history.append({"epoch": epoch, **losses})
        print(f"Epoch {epoch}/{epoch_limit} | {losses}", flush=True)
        eta = (time.monotonic() - progress_start) / epoch * (epoch_limit - epoch)
        print(f"APNET_PROGRESS Epoch {epoch}/{epoch_limit} | train_loss={losses['loss']:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)
    state = snapshot(model)
    # The stopping rule was fixed before this run; no Test-based epoch search.
    predictions = {s: evaluate(model, tables[s], arrays[s], device) for s in SPLITS}
    row = result_row(config, "full_train", epoch_limit, predictions,
                     metrics(predictions["test"]["gt"], predictions["test"]["pred"]))
    row["n_parameters"] = sum(x.numel() for x in model.parameters())
    checkpoint = {
        "format_version": 3, "model": "P16 fixed historical epoch",
        "track": "full_train", "selected_by": selection["provenance"],
        "selected_epoch": epoch_limit, "config": config, "groups": manifest["groups"],
        "dimensions": manifest["dimensions"], "pre_token_dim": 71,
        "model_state": state, "n_parameters": row["n_parameters"],
        "feature_normalization": group_stats,
        "token_normalization": {"mean": token_stats[0], "std": token_stats[1]},
        "epoch_history": history, "metrics": row,
    }
    save_artifact(a.out, checkpoint, predictions)

if __name__ == "__main__":
    main()
