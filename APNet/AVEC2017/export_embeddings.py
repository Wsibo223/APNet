# export_embeddings.py
# V27 Route-A step: export internal branch embeddings/tokens from ExprV17 and NumbV5.
#
# Purpose:
#   V21 showed CSV-level side features are not enough to stably learn oracle branch choice.
#   This script exports deeper processed representations:
#     ExprV17: att_emb, mean_emb, top_hist, att/mean prototype hist, desc embedding, stats, vf.
#     NumbV5 : temporal token pooled embeddings, top-k token embeddings, video_proj embedding,
#              plus hook-captured token_score/token_attn/severe_token_head summaries.
#
# Then run:
#   EXPR_EMBED_DIR=exports_expr_v17_embeddings_v27 \
#   NUMB_EMBED_DIR=exports_numb_v5_embeddings_v27 \
#   bash run_oracle_choice_forensic_v21_gpu0.sh 2026
#
# Important:
#   This script only exports frozen internal features. It does not train or overwrite any checkpoint.
#
# If Expr dataset construction fails:
#   Run with --branch numb first, then paste the printed "available dataset builders/classes".
#   The Numb exporter should work because V5 has a known build_dataset path.

import os
import re
import glob
from pathlib import Path
import json
import math
import inspect
import argparse
import dataclasses
import importlib.util
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from checkpoint import load_checkpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# -----------------------------
# Generic helpers
# -----------------------------

def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ns_from_dict(d):
    if isinstance(d, SimpleNamespace):
        return d
    if isinstance(d, dict):
        return SimpleNamespace(**{k: ns_from_dict(v) if isinstance(v, dict) else v for k, v in d.items()})
    return d


def as_numpy(x):
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        x = x[0]
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def take_tensor(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def get_batch_value(batch, names, default=None):
    if isinstance(batch, dict):
        for n in names:
            if n in batch:
                return batch[n]
    return default


def flatten_vecs(prefix: str, arr: np.ndarray, max_dim: int = None) -> Dict[str, np.ndarray]:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if max_dim is not None:
        arr = arr[:, :max_dim]
    out = {}
    for j in range(arr.shape[1]):
        out[f"{prefix}_{j:03d}"] = arr[:, j]
    return out


def masked_reduce_tokens(tokens: torch.Tensor, mask: torch.Tensor = None, top_score: torch.Tensor = None, topk: int = 8):
    # tokens: [B,T,H]
    if tokens is None:
        return {}
    if isinstance(tokens, (tuple, list)):
        tokens = tokens[0]
    if tokens.ndim != 3:
        return {}
    B, T, H = tokens.shape
    device = tokens.device
    if mask is None:
        mask = torch.ones(B, T, device=device, dtype=torch.bool)
    else:
        mask = mask.bool().to(device)
        if mask.ndim > 2:
            mask = mask.squeeze(-1)
        if mask.shape[1] != T:
            # If lengths mismatch, fall back to all valid.
            mask = torch.ones(B, T, device=device, dtype=torch.bool)
    valid = mask.float()
    den = valid.sum(1).clamp_min(1.0)
    mean = (tokens * valid.unsqueeze(-1)).sum(1) / den.unsqueeze(-1)
    var = ((tokens - mean.unsqueeze(1)) ** 2 * valid.unsqueeze(-1)).sum(1) / den.unsqueeze(-1)
    std = torch.sqrt(var + 1e-6)
    mx = tokens.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
    mn = tokens.masked_fill(~mask.unsqueeze(-1), 1e9).min(1).values

    out = {"tok_mean": mean, "tok_std": std, "tok_max": mx, "tok_min": mn}

    if top_score is not None:
        sc = top_score
        if isinstance(sc, (tuple, list)):
            sc = sc[0]
        if sc.ndim == 3:
            sc = sc.squeeze(-1)
        if sc.ndim == 2 and sc.shape[1] == T:
            sc = sc.masked_fill(~mask, -1e9)
            k = min(topk, T)
            idx = torch.topk(sc, k=k, dim=1).indices
            top_tok = torch.gather(tokens, 1, idx.unsqueeze(-1).expand(-1, -1, H)).mean(1)
            out["tok_topk"] = top_tok
    return out


def safe_float_array(x, n=None):
    arr = as_numpy(x)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = np.repeat(arr.reshape(1), n if n is not None else 1)
    if arr.ndim > 1 and arr.shape[-1] == 1:
        arr = arr.reshape(arr.shape[0])
    return arr


def bin_name(y):
    y = float(y)
    if y < 6:
        return "minimal_0_5"
    if y < 12:
        return "mild_6_11"
    if y < 18:
        return "moderate_12_17"
    return "severe_18_24"


def metrics_np(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((p-y)**2))),
        "mae": float(np.mean(np.abs(p-y))),
        "pred_mean": float(np.mean(p)),
        "pred_std": float(np.std(p, ddof=1)) if len(p) > 1 else 0.0,
    }


# -----------------------------
# Expr V17 export
# -----------------------------

def find_expr_script():
    hits = sorted(glob.glob(str(Path(__file__).resolve().parent / "branches/aee/train.py")))
    for h in hits:
        txt = open(h, "r", encoding="utf-8").read()
        if "class V17" in txt and "class SegmentEncoder" in txt:
            return h
    raise FileNotFoundError("No train_expr_v17*.py with class V17 found.")


def construct_expr_model(mod, ckpt):
    sd = ckpt["model_state"]
    input_dim = int(sd["enc.proj.0.weight"].shape[1])
    desc_dim = int(sd["desc.0.weight"].shape[1])
    hid = int(sd["enc.proto"].shape[1])
    K = int(sd["enc.proto"].shape[0])
    cap = float(ckpt.get("args", {}).get("expr_score_cap", 14.0)) if isinstance(ckpt.get("args", {}), dict) else 14.0
    if "proto_scores" in ckpt:
        try:
            cap = float(np.asarray(ckpt["proto_scores"]).max())
        except Exception:
            pass
    a = ckpt.get("args", {})
    residual_max = float(a.get("residual_max", 3.8)) if isinstance(a, dict) else 3.8
    drop = float(a.get("dropout", 0.08)) if isinstance(a, dict) else 0.08
    tau = float(a.get("proto_tau", 0.16)) if isinstance(a, dict) else 0.16
    attn_temp = float(a.get("attn_temp", 0.85)) if isinstance(a, dict) else 0.85
    model = mod.V17(input_dim=input_dim, desc_dim=desc_dim, hid=hid, K=K, cap=cap,
                    residual_max=residual_max, drop=drop, tau=tau, attn_temp=attn_temp)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print("[Expr] load_state missing:", miss)
    print("[Expr] load_state unexpected:", unexp)
    return model


def expr_forward_features(model, x, mask, bag, desc, anchor):
    # Same as V17.forward, but returns deeper embeddings.
    B, S, T, D = x.shape
    cap = model.proto_scores.max().clamp_min(1)
    emb, hist, seg, pscore, pc, pe, pp = model.enc(x.view(B*S, T, D), mask.view(B*S, T))
    H = emb.shape[-1]
    K = hist.shape[-1]
    emb = emb.view(B, S, H)
    hist = hist.view(B, S, K)
    seg = seg.view(B, S)
    pscore = pscore.view(B, S)
    pc = pc.view(B, S)
    pe = pe.view(B, S)
    pp = pp.view(B, S)

    valid = bag.float()
    den = valid.sum(1).clamp_min(1)
    logits = model.seg_attn(torch.cat([
        emb, hist, (seg/cap).unsqueeze(-1), (pscore/cap).unsqueeze(-1),
        pc.unsqueeze(-1), pe.unsqueeze(-1), pp.unsqueeze(-1)
    ], -1)).squeeze(-1)
    logits = logits + 0.15*(seg/cap) + 0.10*torch.log(pc.clamp_min(1e-4))
    logits = logits.masked_fill(~bag.bool(), -1e9)
    a = torch.softmax(logits/model.attn_temp, dim=1)
    a = a * valid
    a = a / a.sum(1, keepdim=True).clamp_min(1e-8)

    mean_emb = (emb * valid.unsqueeze(-1)).sum(1) / den.unsqueeze(-1)
    att_emb = (emb * a.unsqueeze(-1)).sum(1)
    mean_hist = (hist * valid.unsqueeze(-1)).sum(1) / den.unsqueeze(-1)
    att_hist = (hist * a.unsqueeze(-1)).sum(1)

    top_h, top_s, low_s, top_emb, low_emb = [], [], [], [], []
    for i in range(B):
        idx = torch.nonzero(bag[i], as_tuple=False).view(-1)
        vals = seg[i, idx]
        k = max(1, int(math.ceil(0.2 * len(vals))))
        hi = idx[torch.topk(vals, k=k).indices]
        lo = idx[torch.topk(-vals, k=k).indices]
        top_h.append(hist[i, hi].mean(0))
        top_s.append(seg[i, hi].mean())
        low_s.append(seg[i, lo].mean())
        top_emb.append(emb[i, hi].mean(0))
        low_emb.append(emb[i, lo].mean(0))
    top_hist = torch.stack(top_h)
    top_score = torch.stack(top_s)
    seg_low = torch.stack(low_s)
    top_emb = torch.stack(top_emb)
    low_emb = torch.stack(low_emb)

    seg_mean = (seg * valid).sum(1) / den
    seg_attn = (seg * a).sum(1)
    seg_var = (((seg - seg_mean.unsqueeze(1)) ** 2) * valid).sum(1) / den
    seg_std = torch.sqrt(seg_var + 1e-6)

    proto_entropy = (pe * valid).sum(1) / den
    proto_peak = (pp * valid).sum(1) / den
    attn_peak = a.max(1).values
    attn_entropy = -(a.clamp_min(1e-8) * torch.log(a.clamp_min(1e-8))).sum(1) / torch.log(den.clamp_min(2))

    low_mask = (model.proto_scores <= 4.5).float()
    mid_mask = ((model.proto_scores > 4.5) & (model.proto_scores < 9.5)).float()
    high_mask = (model.proto_scores >= 9.5).float()
    low_mass = (mean_hist * low_mask.view(1, -1)).sum(1)
    mid_mass = (mean_hist * mid_mask.view(1, -1)).sum(1)
    high_mass = (mean_hist * high_mask.view(1, -1)).sum(1)
    proto_score = (mean_hist * model.proto_scores.view(1, -1)).sum(1)
    proto_score_attn = (att_hist * model.proto_scores.view(1, -1)).sum(1)

    d = model.desc(desc)
    an = (anchor / 24).clamp(0, 1)
    mil = torch.stack([
        an, proto_score/cap, proto_score_attn/cap, seg_mean/cap, seg_attn/cap,
        top_score/cap, low_mass, mid_mass, high_mass
    ], 1)
    stats = torch.stack([
        an, proto_score/cap, proto_score_attn/cap, seg_mean/cap, seg_attn/cap,
        top_score/cap, seg_low/cap, seg_std/cap, low_mass, mid_mass, high_mass,
        proto_entropy, proto_peak, attn_peak, attn_entropy, den/float(S),
        torch.relu(anchor-6)/18, torch.relu(6-anchor)/6,
        (att_hist-mean_hist).abs().sum(1)/2
    ], 1)
    vf = torch.cat([att_emb, mean_emb, att_hist, mean_hist, top_hist, d, stats], 1)
    r1 = model.residual_max * model.res_head(vf).view(-1)
    r2 = model.residual_max * model.mil_res(mil).view(-1)
    residual = 0.75*r1 + 0.25*r2
    pred = torch.clamp(anchor + residual, 0, 24)
    p_low = torch.sigmoid(model.low_head(vf).view(-1))

    return {
        "pred": pred,
        "residual": residual,
        "residual_deep": r1,
        "residual_mil": r2,
        "p_low": p_low,
        "att_emb": att_emb,
        "mean_emb": mean_emb,
        "top_emb": top_emb,
        "low_emb": low_emb,
        "desc_emb": d,
        "vf": vf,
        "stats": stats,
        "att_hist": att_hist,
        "mean_hist": mean_hist,
        "top_hist": top_hist,
        "seg_mean": seg_mean,
        "seg_attn": seg_attn,
        "top_score": top_score,
        "seg_low": seg_low,
        "seg_std": seg_std,
        "proto_entropy": proto_entropy,
        "proto_peak": proto_peak,
        "attn_peak": attn_peak,
        "attn_entropy": attn_entropy,
        "low_mass": low_mass,
        "mid_mass": mid_mass,
        "high_mass": high_mass,
        "proto_score": proto_score,
        "proto_score_attn": proto_score_attn,
        "n_segments": den,
    }


def find_collate(mod):
    # prefer functions explicitly containing collate
    for name in dir(mod):
        obj = getattr(mod, name)
        if callable(obj) and "collate" in name.lower():
            return obj
    return None




def _is_valid_expr_dataset(ds):
    if ds is None or not hasattr(ds, "__len__") or not hasattr(ds, "__getitem__"):
        return False
    try:
        if len(ds) <= 0:
            return False
        sample = ds[0]
        return isinstance(sample, dict) and all(k in sample for k in ["x", "mask", "desc", "anchor", "y"])
    except Exception:
        return False


def _collect_valid_datasets(obj):
    found = []
    if _is_valid_expr_dataset(obj):
        found.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found += _collect_valid_datasets(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found += _collect_valid_datasets(v)
    return found


def _pick_dataset_from_result(res, split_name):
    if isinstance(res, dict):
        for key in [split_name, f"{split_name}_ds", f"{split_name}_dataset"]:
            if key in res and _is_valid_expr_dataset(res[key]):
                return res[key]
    ds_list = _collect_valid_datasets(res)
    if not ds_list:
        return None
    idx_map = {"train": 0, "dev": 1, "test": 2}
    idx = idx_map.get(split_name, 0)
    return ds_list[idx] if idx < len(ds_list) else ds_list[0]


def _extract_paths_from_meta(meta):
    # Supports DataFrame, list[dict], tuple/list returned by read_txt.
    if isinstance(meta, pd.DataFrame):
        cols = list(meta.columns)
        path_cols = [c for c in cols if any(k in str(c).lower() for k in ["au", "path", "csv", "file"])]
        if not path_cols:
            # fallback: first object column containing existing-looking strings
            path_cols = [c for c in cols if meta[c].dtype == object]
        if not path_cols:
            raise RuntimeError(f"Cannot find path column in read_txt DataFrame columns={cols}")
        pcol = path_cols[0]
        paths = meta[pcol].astype(str).tolist()
        return meta, paths

    if isinstance(meta, (list, tuple)):
        # If read_txt returns a tuple, one element may be paths.
        for obj in meta:
            if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) > 0 and isinstance(list(obj)[0], str):
                # list of paths
                return meta[0] if len(meta) > 1 else meta, [str(x) for x in obj]
        # list of dict records
        if len(meta) > 0 and isinstance(meta[0], dict):
            keys = list(meta[0].keys())
            path_keys = [k for k in keys if any(t in str(k).lower() for t in ["au", "path", "csv", "file"])]
            if not path_keys:
                raise RuntimeError(f"Cannot find path key in read_txt list-of-dict keys={keys}")
            pk = path_keys[0]
            return meta, [str(r[pk]) for r in meta]

    raise RuntimeError(f"Unsupported read_txt return type: {type(meta)}")


def _make_expr_anchor(mod, arrays, ckpt):
    # Anchor prediction used by V17 forward. Rebuild it exactly from saved beta/mu/sigma when available.
    raw_mean = ckpt.get("raw_mean")
    raw_std = ckpt.get("raw_std")
    dmu = ckpt.get("desc_mu")
    dsig = ckpt.get("desc_sig")
    beta = ckpt.get("anchor_beta", None)

    if beta is None:
        # fallback, should not happen for current checkpoint
        return np.zeros(len(arrays), dtype=np.float32)

    descs = []
    for x in arrays:
        descs.append(mod.make_descriptor(x, raw_mean, raw_std))
    D = np.stack(descs, axis=0).astype(np.float32)
    if dmu is not None and dsig is not None:
        Dz = (D - dmu) / np.maximum(dsig, 1e-6)
    else:
        Dz = D
    Xb = np.concatenate([np.ones((len(Dz), 1), dtype=np.float32), Dz.astype(np.float32)], axis=1)
    beta = np.asarray(beta, dtype=np.float32).reshape(-1)
    if beta.shape[0] != Xb.shape[1]:
        # Some scripts store beta without intercept.
        if beta.shape[0] == Dz.shape[1]:
            pred = Dz @ beta
        else:
            raise RuntimeError(f"anchor_beta dim mismatch: beta={beta.shape}, Xb={Xb.shape}, Dz={Dz.shape}")
    else:
        pred = Xb @ beta
    return np.clip(pred, 0, 24).astype(np.float32)


def _build_expr_via_build_videos(mod, txt_path, ckpt, ckpt_args, split_name):
    if not hasattr(mod, "read_txt") or not hasattr(mod, "load_au") or not hasattr(mod, "build_videos") or not hasattr(mod, "ExprDataset"):
        return None

    meta0 = mod.read_txt(txt_path)
    meta, paths = _extract_paths_from_meta(meta0)
    au_cols = ckpt.get("au_columns", None)
    if au_cols is None:
        raise RuntimeError("Expr checkpoint has no au_columns; cannot call load_au.")

    arrays = []
    good_paths = []
    for pth in paths:
        arr = mod.load_au(pth, au_cols)
        if isinstance(arr, (tuple, list)):
            arr = arr[0]
        arrays.append(arr)
        good_paths.append(pth)

    anchors = _make_expr_anchor(mod, arrays, ckpt)
    mean = ckpt.get("raw_mean")
    std = ckpt.get("raw_std")
    dmu = ckpt.get("desc_mu")
    dsig = ckpt.get("desc_sig")

    print(f"[Expr] calling build_videos(meta, arrays, paths, anchors, mean, std, dmu, dsig, args, split), n_paths={len(good_paths)}")
    videos = mod.build_videos(meta, arrays, good_paths, anchors, mean, std, dmu, dsig, ckpt_args, split_name)
    if isinstance(videos, tuple):
        # Usually first item is the video list.
        videos = videos[0]
    ds = mod.ExprDataset(videos, train=False, args=ckpt_args)
    if _is_valid_expr_dataset(ds):
        print(f"[Expr] built ExprDataset from build_videos, split={split_name}, n={len(ds)}")
        return ds
    # Some ExprDataset may only work after passing train arg positionally.
    ds = mod.ExprDataset(videos, False, ckpt_args)
    if _is_valid_expr_dataset(ds):
        print(f"[Expr] built ExprDataset positional from build_videos, split={split_name}, n={len(ds)}")
        return ds
    raise RuntimeError("ExprDataset built from build_videos but sample validation failed.")


def _try_expr_make_datasets(mod, ckpt, split_name, train_txt, dev_txt, test_txt):
    if not hasattr(mod, "make_datasets"):
        return None
    ckpt_args = ns_from_dict(ckpt.get("args", {}))
    ckpt_args.train_txt = train_txt
    ckpt_args.dev_txt = dev_txt
    ckpt_args.test_txt = test_txt
    try:
        res = mod.make_datasets(ckpt_args)
        ds = _pick_dataset_from_result(res, split_name)
        if ds is not None:
            print(f"[Expr] built dataset via make_datasets(args), split={split_name}")
            return ds
    except Exception as e:
        print("[Expr] make_datasets failed:", repr(e)[:300])
    return None


def build_expr_dataset_auto(mod, txt_path, ckpt_args, split_name):
    ckpt = getattr(build_expr_dataset_auto, "_ckpt")
    all_txt = getattr(build_expr_dataset_auto, "_all_txt", {})

    # Current V17 script path: read_txt -> load_au -> build_videos -> ExprDataset.
    try:
        ds = _build_expr_via_build_videos(mod, txt_path, ckpt, ckpt_args, split_name)
        if ds is not None:
            return ds
    except Exception as e:
        print("[Expr] build_videos path failed:", repr(e)[:800])

    ds = _try_expr_make_datasets(
        mod, ckpt, split_name,
        all_txt.get("train", "train_2017.txt"),
        all_txt.get("dev", "dev_2017.txt"),
        all_txt.get("test", "test_2017.txt")
    )
    if ds is not None:
        return ds

    print("\n[Expr] Could not auto-build dataset.")
    print("[Expr] Relevant names:", [n for n in dir(mod) if any(k in n.lower() for k in ["dataset", "video", "sample", "collate", "make", "build", "load", "read"])][:160])
    raise RuntimeError("Expr dataset auto-build failed. Please paste read_txt/build_videos/ExprDataset source.")


@torch.no_grad()
def export_expr(args):
    out_dir = args.expr_out_dir
    os.makedirs(out_dir, exist_ok=True)

    script = args.expr_script or find_expr_script()
    mod = load_module(script, "expr_v17_mod")
    ckpt = load_checkpoint(args.expr_ckpt, map_location="cpu")
    ckpt_args = ns_from_dict(ckpt.get("args", {}))
    build_expr_dataset_auto._ckpt = ckpt
    build_expr_dataset_auto._all_txt = {"train": args.train_txt, "dev": args.dev_txt, "test": args.test_txt}

    model = construct_expr_model(mod, ckpt)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model.to(device).eval()

    collate = find_collate(mod)
    if collate is None:
        print("[Expr] WARNING: no collate function found; using default DataLoader collate.")

    split_paths = {"train": args.train_txt, "dev": args.dev_txt, "test": args.test_txt}
    meta = {"script": script, "ckpt": args.expr_ckpt, "out_dir": out_dir}

    for split, txt in split_paths.items():
        ds = build_expr_dataset_auto(mod, txt, ckpt_args, split)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate if collate is not None else None)

        rows = []
        for batch in loader:
            if not isinstance(batch, dict):
                raise RuntimeError("Expr batch is not a dict; need custom collate. Paste batch type/structure.")
            x = batch["x"].to(device).float()
            mask = batch["mask"].to(device).bool()
            bag_key = "bag" if "bag" in batch else "bag_mask"
            if bag_key not in batch:
                raise RuntimeError(f"Expr batch has no bag/bag_mask. batch keys={list(batch.keys())}")
            bag = batch[bag_key].to(device).bool()
            desc = batch["desc"].to(device).float()
            anchor = batch["anchor"].to(device).float()
            y = batch["y"].detach().cpu().numpy().astype(float)
            vids = [str(v) for v in batch.get("video_id", [f"{split}_{i}" for i in range(len(y))])]
            out = expr_forward_features(model, x, mask, bag, desc, anchor)

            base = {
                "video_id": vids,
                "gt": y,
                "pred": as_numpy(out["pred"]).reshape(-1),
                "bin": [bin_name(v) for v in y],
            }
            df = pd.DataFrame(base)
            # Scalars
            for k in ["residual", "residual_deep", "residual_mil", "p_low", "seg_mean", "seg_attn", "top_score",
                      "seg_low", "seg_std", "proto_entropy", "proto_peak", "attn_peak", "attn_entropy",
                      "low_mass", "mid_mass", "high_mass", "proto_score", "proto_score_attn", "n_segments"]:
                df[f"expr_{k}"] = as_numpy(out[k]).reshape(-1)

            # Vectors
            for prefix, key, mdim in [
                ("expr_att_emb", "att_emb", args.max_emb_dims),
                ("expr_mean_emb", "mean_emb", args.max_emb_dims),
                ("expr_top_emb", "top_emb", args.max_emb_dims),
                ("expr_low_emb", "low_emb", args.max_emb_dims),
                ("expr_desc_emb", "desc_emb", args.max_emb_dims),
                ("expr_att_hist", "att_hist", None),
                ("expr_mean_hist", "mean_hist", None),
                ("expr_top_hist", "top_hist", None),
                ("expr_stats", "stats", None),
                ("expr_vf", "vf", args.max_vf_dims),
            ]:
                vecs = flatten_vecs(prefix, as_numpy(out[key]), mdim)
                for c, v in vecs.items():
                    df[c] = v
            rows.append(df)

        all_df = pd.concat(rows, axis=0, ignore_index=True)
        all_df.to_csv(os.path.join(out_dir, f"{split}_embeddings.csv"), index=False)
        print(f"[Expr] {split}: saved {len(all_df)} rows -> {out_dir}/{split}_embeddings.csv | pred_rmse={metrics_np(all_df['gt'], all_df['pred'])['rmse']:.4f}")

    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# -----------------------------
# Numb V5 export
# -----------------------------

def construct_numb_model(ckpt, device):
    from branches.pae.model import ARDSevereAwareNet
    sig = inspect.signature(ARDSevereAwareNet.__init__)
    accepted = set(sig.parameters.keys())
    a = ckpt.get("args", {})
    kwargs = {
        "in_dim": int(ckpt["feature_dim"]),
        "hidden_dim": int(a.get("hidden_dim", 128)),
        "num_layers": int(a.get("num_layers", 4)),
        "dropout": float(a.get("dropout", 0.15)),
        "n_severity_bins": 4,
        "topk_ratio": float(a.get("topk_ratio", 0.1)),
        "min_topk": int(a.get("min_topk", 3)),
        "severe_topk_ratio": float(a.get("severe_topk_ratio", 0.15)),
        "severe_min_topk": int(a.get("severe_min_topk", 4)),
        "severe_residual_max": float(a.get("severe_residual_max", 5.0)),
    }
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    model = ARDSevereAwareNet(**kwargs).to(device)
    miss, unexp = model.load_state_dict(ckpt["model_state"], strict=False)
    print("[Numb] init kwargs:", kwargs)
    print("[Numb] load_state missing:", miss)
    print("[Numb] load_state unexpected:", unexp)
    return model



def _module_dataclass_candidates(*mods):
    cands = []
    for m in mods:
        if m is None:
            continue
        for name in dir(m):
            obj = getattr(m, name)
            if inspect.isclass(obj) and dataclasses.is_dataclass(obj):
                cands.append((name, obj))
    return cands


def _load_optional_module(path, name):
    if os.path.exists(path):
        try:
            return load_module(path, name)
        except Exception:
            return None
    try:
        return __import__(path.replace(".py", ""))
    except Exception:
        return None


def _cfg_to_dataclass(cfg, purpose, train_mod):
    if dataclasses.is_dataclass(cfg):
        return cfg
    if isinstance(cfg, SimpleNamespace):
        cfg_dict = vars(cfg)
    elif isinstance(cfg, dict):
        cfg_dict = dict(cfg)
    else:
        return cfg

    ds_mod = None
    try:
        import branches.pae.dataset_fast as ds_mod_imported
        ds_mod = ds_mod_imported
    except Exception:
        ds_mod = _load_optional_module(str(Path(__file__).resolve().parent / "branches/pae/dataset_fast.py"), "dataset_ard_fast_mod")

    best = None
    best_score = -1
    for name, Cls in _module_dataclass_candidates(train_mod, ds_mod):
        fields = {f.name: f for f in dataclasses.fields(Cls)}
        overlap = len(set(fields.keys()) & set(cfg_dict.keys()))
        lname = name.lower()
        bonus = 0
        if purpose == "sanitizer" and any(k in lname for k in ["san", "clean", "filter"]):
            bonus += 5
        if purpose == "patch" and any(k in lname for k in ["patch", "window", "token"]):
            bonus += 5
        score = overlap + bonus
        if score > best_score:
            best = Cls
            best_score = score

    if best is None or best_score <= 0:
        print(f"[Numb] WARNING: no dataclass found for {purpose}; using SimpleNamespace.")
        return ns_from_dict(cfg_dict)

    kwargs = {}
    for f in dataclasses.fields(best):
        if f.name in cfg_dict:
            kwargs[f.name] = cfg_dict[f.name]
    try:
        obj = best(**kwargs)
        print(f"[Numb] {purpose} cfg dataclass: {best.__name__} | fields={list(kwargs.keys())}")
        return obj
    except Exception as e:
        print(f"[Numb] dataclass instantiate failed for {best.__name__}: {e}; using SimpleNamespace.")
        return ns_from_dict(cfg_dict)


def build_numb_dataset(train_mod, txt_path, ckpt, split_name, args):
    ckpt_args = ns_from_dict(ckpt.get("args", {}))
    if args.numb_cache_root:
        ckpt_args.cache_root = args.numb_cache_root

    san = _cfg_to_dataclass(ckpt.get("sanitizer_cfg", {}), "sanitizer", train_mod)
    patch = _cfg_to_dataclass(ckpt.get("patch_cfg", {}), "patch", train_mod)

    if hasattr(train_mod, "build_dataset"):
        f = train_mod.build_dataset
        attempts = [
            ("full_verbose", lambda: f(txt_path, san, patch, ckpt_args, split_name, verbose=True)),
            ("full_noverbose", lambda: f(txt_path, san, patch, ckpt_args, split_name)),
            ("kw", lambda: f(txt_path=txt_path, san=san, patch=patch, args=ckpt_args, split_name=split_name, verbose=True)),
        ]
        errors = []
        for name, call in attempts:
            try:
                ds = call()
                if hasattr(ds, "__len__"):
                    print(f"[Numb] built dataset via build_dataset attempt={name}, split={split_name}, n={len(ds)}")
                    return ds
            except Exception as e:
                errors.append(f"{name}: {repr(e)[:500]}")
        print("[Numb] build_dataset attempts failed:")
        for e in errors:
            print("  ", e)
        raise RuntimeError("Numb build_dataset failed. See errors above.")
    raise RuntimeError("Numb train module does not expose build_dataset.")



def call_numb_model(model, x, mask=None):
    # Try common forward signatures.
    for args in [(x, mask), (x,), (x, mask, False)]:
        try:
            if args[1] is None if len(args) > 1 else False:
                continue
            return model(*args)
        except Exception:
            pass
    try:
        return model(x, mask)
    except Exception as e:
        raise RuntimeError(f"Could not call numb model forward with x/mask. Error: {e}")


@torch.no_grad()
def export_numb(args):
    out_dir = args.numb_out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_mod = load_module(args.numb_train_script, "numb_v5_train_mod")
    ckpt = load_checkpoint(args.numb_ckpt, map_location="cpu")
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = construct_numb_model(ckpt, device)
    model.eval()

    # Hooks for internals.
    captured = {}
    handles = []
    module_dict = dict(model.named_modules())

    def make_hook(name):
        def hook(module, inp, out):
            captured[name] = take_tensor(out).detach()
        return hook

    for name in ["input_proj", "temporal", "video_proj", "token_score", "token_attn", "severe_token_head", "reg_head"]:
        if name in module_dict:
            handles.append(module_dict[name].register_forward_hook(make_hook(name)))
            print("[Numb] hook:", name)
        else:
            print("[Numb] hook missing:", name)

    collate = getattr(train_mod, "collate_ard_retardation", None)
    if collate is None:
        # fallback: any collate function
        for n in dir(train_mod):
            if "collate" in n.lower():
                collate = getattr(train_mod, n)
                print("[Numb] using collate:", n)
                break

    split_paths = {"train": args.train_txt, "dev": args.dev_txt, "test": args.test_txt}
    meta = {"train_script": args.numb_train_script, "ckpt": args.numb_ckpt, "out_dir": out_dir}

    for split, txt in split_paths.items():
        ds = build_numb_dataset(train_mod, txt, ckpt, split, args)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate if collate is not None else None)

        rows = []
        for batch in loader:
            if not isinstance(batch, dict):
                raise RuntimeError("Numb batch is not a dict; need custom collate.")
            x = get_batch_value(batch, ["x", "tokens", "feat", "features", "input"])
            if x is None:
                raise RuntimeError(f"Numb batch has no feature tensor. batch keys={list(batch.keys())}")
            x = x.to(device).float()
            mask = get_batch_value(batch, ["mask", "valid_mask", "attention_mask"], None)
            if mask is not None:
                mask = mask.to(device).bool()
            y_val = get_batch_value(batch, ["y", "gt", "score", "label"])
            if y_val is None:
                raise RuntimeError(f"Numb batch has no label tensor. batch keys={list(batch.keys())}")
            y_t = y_val.detach().cpu().numpy().astype(float).reshape(-1)
            vids = batch.get("video_id", batch.get("video_ids", batch.get("vid", [f"{split}_{i}" for i in range(len(y_t))])))
            vids = [str(v) for v in vids]

            captured.clear()
            out = call_numb_model(model, x, mask)
            if isinstance(out, dict):
                pred = out.get("pred", out.get("prediction", out.get("score", None)))
            else:
                pred = out
            pred_np = safe_float_array(pred, n=len(y_t)).reshape(-1)

            df = pd.DataFrame({
                "video_id": vids,
                "gt": y_t,
                "pred": pred_np,
                "bin": [bin_name(v) for v in y_t],
            })

            # Dict outputs if available.
            if isinstance(out, dict):
                for k, v in out.items():
                    if k in ["pred", "prediction", "score"]:
                        continue
                    arr = safe_float_array(v, n=len(y_t))
                    if arr is None:
                        continue
                    if arr.ndim == 1 and len(arr) == len(y_t):
                        df[f"numb_{k}"] = arr
                    elif arr.ndim == 2 and arr.shape[0] == len(y_t) and arr.shape[1] <= args.max_small_vec_dims:
                        for c, val in flatten_vecs(f"numb_out_{k}", arr).items():
                            df[c] = val

            # Hooked video-level embeddings.
            for hook_name in ["video_proj", "reg_head"]:
                if hook_name in captured:
                    arr = as_numpy(captured[hook_name])
                    if arr.ndim == 2 and arr.shape[0] == len(y_t):
                        for c, val in flatten_vecs(f"numb_{hook_name}", arr, args.max_emb_dims).items():
                            df[c] = val

            # Hooked temporal tokens.
            tok = captured.get("temporal", None)
            score_for_top = captured.get("token_score", None)
            if tok is not None:
                red = masked_reduce_tokens(tok, mask=mask, top_score=score_for_top, topk=args.topk_tokens)
                for name, ten in red.items():
                    arr = as_numpy(ten)
                    for c, val in flatten_vecs(f"numb_{name}", arr, args.max_emb_dims).items():
                        df[c] = val

            # Token head summaries.
            for hook_name in ["token_score", "token_attn", "severe_token_head"]:
                if hook_name in captured:
                    arr = captured[hook_name]
                    if isinstance(arr, (tuple, list)):
                        arr = arr[0]
                    if arr.ndim == 3 and arr.shape[-1] == 1:
                        arr = arr.squeeze(-1)
                    if arr.ndim == 2:
                        if mask is not None and mask.shape == arr.shape:
                            valid = mask.float()
                            den = valid.sum(1).clamp_min(1.0)
                            arr_masked = arr.masked_fill(~mask, 0.0)
                            mean = (arr_masked * valid).sum(1) / den
                            mx = arr.masked_fill(~mask, -1e9).max(1).values
                            mn = arr.masked_fill(~mask, 1e9).min(1).values
                            df[f"numb_{hook_name}_mean"] = as_numpy(mean)
                            df[f"numb_{hook_name}_max"] = as_numpy(mx)
                            df[f"numb_{hook_name}_min"] = as_numpy(mn)
                        else:
                            df[f"numb_{hook_name}_mean"] = as_numpy(arr.mean(1))
                            df[f"numb_{hook_name}_max"] = as_numpy(arr.max(1).values)
                            df[f"numb_{hook_name}_min"] = as_numpy(arr.min(1).values)

            rows.append(df)

        all_df = pd.concat(rows, axis=0, ignore_index=True)
        all_df.to_csv(os.path.join(out_dir, f"{split}_embeddings.csv"), index=False)
        print(f"[Numb] {split}: saved {len(all_df)} rows -> {out_dir}/{split}_embeddings.csv | pred_rmse={metrics_np(all_df['gt'], all_df['pred'])['rmse']:.4f}")

    for h in handles:
        h.remove()

    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# -----------------------------
# Main
# -----------------------------

def main(args):
    if args.branch in ["expr", "both"]:
        export_expr(args)
    if args.branch in ["numb", "both"]:
        export_numb(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--branch", choices=["expr", "numb", "both"], default="both")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu", action="store_true")

    p.add_argument("--train_txt", default="train_2017.txt")
    p.add_argument("--dev_txt", default="dev_2017.txt")
    p.add_argument("--test_txt", default="test_2017.txt")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=0)

    p.add_argument("--expr_script", default="")
    p.add_argument("--expr_ckpt", default="aee_training/aee_best.pth")
    p.add_argument("--expr_out_dir", default="exports_expr_v17_embeddings_v27")

    p.add_argument("--numb_train_script", default="branches/pae/train.py")
    p.add_argument("--numb_ckpt", default="pae_training/pae_best.pth")
    p.add_argument("--numb_out_dir", default="exports_numb_v5_embeddings_v27")
    p.add_argument("--numb_cache_root", default="")

    p.add_argument("--max_emb_dims", type=int, default=128)
    p.add_argument("--max_vf_dims", type=int, default=241)
    p.add_argument("--max_small_vec_dims", type=int, default=64)
    p.add_argument("--topk_tokens", type=int, default=8)

    args = p.parse_args()
    main(args)
