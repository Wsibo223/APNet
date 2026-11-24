"""Export AEE features using the checkpoint trained by this run."""

import os
import argparse
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    mod = importlib.import_module(args.module_name)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = SimpleNamespace(**ckpt["args"])
    ckpt_args.train_txt = args.train_txt
    ckpt_args.dev_txt = args.dev_txt
    ckpt_args.test_txt = args.test_txt
    ckpt_args.data_root = args.data_root
    ckpt_args.save_dir = args.out_dir

    au_cols = ckpt.get("au_columns", [c.strip() for c in ckpt_args.au_columns.split(",") if c.strip()])
    raw_mean = ckpt["raw_mean"]
    raw_std = ckpt["raw_std"]
    desc_mu = ckpt["desc_mu"]
    desc_sig = ckpt["desc_sig"]
    anchor_beta = ckpt["anchor_beta"]

    def load_split(txt_path, split):
        meta = mod.read_txt(txt_path)
        arrays, paths = [], []
        for _, r in tqdm(meta.iterrows(), total=len(meta), desc=f"load {split}", ncols=100):
            p = mod.find_au_file(r["raw_path"], ckpt_args.data_root)
            paths.append(p)
            arrays.append(mod.load_au(p, au_cols, success_filter=not getattr(ckpt_args, "no_success_filter", False)))

        # train-only anchor from checkpoint final ridge beta.
        desc_raw = np.stack([mod.make_descriptor(a, raw_mean, raw_std) for a in arrays])
        desc_z = (desc_raw - desc_mu.reshape(1, -1)) / desc_sig.reshape(1, -1)
        anchors = mod.ridge_pred(anchor_beta, desc_z).astype(np.float32)

        videos = mod.build_videos(meta, arrays, paths, anchors, raw_mean, raw_std, desc_mu, desc_sig, ckpt_args, split)
        return videos

    datasets = [(split, load_split(getattr(ckpt_args, split + "_txt"), split)) for split in args.splits]

    # Build model.
    first_video = datasets[0][1][0]
    input_dim = first_video.segments.shape[-1]
    desc_dim = first_video.desc.shape[0]
    model = mod.V17(
        input_dim,
        desc_dim,
        ckpt_args.hidden_dim,
        ckpt_args.n_proto,
        ckpt_args.expr_score_cap,
        ckpt_args.residual_max,
        ckpt_args.dropout,
        ckpt_args.proto_tau,
        ckpt_args.attn_temp,
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # Ensure module device agrees if it uses global DEVICE in evaluate.
    try:
        mod.DEVICE = device
    except Exception:
        pass

    for split, videos in datasets:
        df, m, bm = mod.evaluate(model, videos, ckpt_args, split)
        df.to_csv(os.path.join(args.out_dir, f"{split}_predictions.csv"), index=False)
        bm.to_csv(os.path.join(args.out_dir, f"{split}_bin_metrics.csv"), index=False)
        print(f"[{split}] {m}")

    print(f"Saved fusion-ready expression predictions to: {args.out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--module_name", default="branches.aee.train")
    p.add_argument("--train_txt", default="train_2019.txt")
    p.add_argument("--dev_txt", default="dev_2019.txt")
    p.add_argument("--test_txt", default="test_2019.txt")
    p.add_argument("--splits", nargs="+", choices=["train", "dev", "test"], default=["train", "dev", "test"])
    p.add_argument("--data_root", default="")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    main(args)
