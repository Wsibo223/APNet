#!/usr/bin/env python3
"""Export only hyperparameters from a trusted local historical PAE checkpoint."""
import argparse
import json
from pathlib import Path
from checkpoint import load_checkpoint
from branches.pae.train import parser

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    saved = load_checkpoint(a.checkpoint)["args"]
    omit = {"train_txt", "dev_txt", "test_txt", "save_dir", "cache_root"}
    flags = []
    known = set()
    for act in parser()._actions:
        key = act.dest
        if key == "help" or key in omit or key not in saved:
            continue
        val = saved[key]
        if isinstance(act, argparse._StoreTrueAction):
            if val:
                flags.append(act.option_strings[0])
            known.add(key)
        elif isinstance(act, argparse._StoreFalseAction):
            if not val:
                flags.append(act.option_strings[0])
            known.add(key)
        elif val is not None:
            flags += [act.option_strings[0], str(val)]
            known.add(key)
    unknown = set(saved) - omit - known
    if unknown:
        raise ValueError(f"Unrecognized historical arguments: {sorted(unknown)}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump({"status": "historical_checkpoint_args", "args": flags}, f, indent=2)
    print(a.out)

if __name__ == "__main__":
    main()
