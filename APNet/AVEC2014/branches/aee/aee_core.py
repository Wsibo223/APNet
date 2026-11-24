#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AVEC2014 AEE V3.1: evidence-gated adaptation of the AVEC2017 AEE branch.

核心思想：
1. 不再让深层 AU 表情分支从零预测 BDI-II 0~63。
2. 先用 train-only shallow AU descriptor ridge 得到稳定 anchor。
   - train anchor 使用 K-fold OOF，避免 train label 过拟合。
   - dev/test anchor 使用 train-only full ridge 推理。
3. 深层 temporal-MIL + compressed expression lexicon 只学习 bounded residual:
      pred = anchor + residual, residual in [-R, R]
4. 保留“压缩表情词典 + 有界残差”的主架构；仅把分数相关常数映射到 BDI-II。
5. AVEC2014 适配：17 AU 强度 + 17 AU occurrence、30 Hz 时间戳重采样、
   OpenFace confidence 质量通道、等视频权重标准化、近自然严重度采样。
6. 支持 shared / Freeform / Northwind 三种训练模式；shared 模式按 pair 成批，
   默认按配对均值选模并同时报告 paired_mean、各任务和 clips_pooled。
7. V3 保留 2017 的压缩表情词典和双层时序注意力，但修复四个 V2 实证故障：
   绝对相似度缺失、窗口注意力均匀、词典分离失败、残差错误修正 anchor。

输出包括 train/dev/test 的逐片段、逐 pair、按任务和按严重度指标。
"""

import os, re, glob, json, math, argparse, random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_MIN = 0.0
SCORE_MAX = 63.0
SEVERITY_CUTS = (14.0, 20.0, 29.0)
SEVERITY_NAMES = (
    "minimal_0_13", "mild_14_19", "moderate_20_28", "severe_29_63"
)

AU_CODES = ("01", "02", "04", "05", "06", "07", "09", "10", "12", "14",
            "15", "17", "20", "23", "25", "26", "45")
DEFAULT_AU_COLUMNS = [f"AU{code}_r" for code in AU_CODES] + [f"AU{code}_c" for code in AU_CODES]
QUALITY_COLUMN = "openface_quality"


# -----------------------------
# Basic metrics
# -----------------------------

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def to_jsonable(x):
    if isinstance(x, dict): return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, (np.floating,)): return float(x)
    if isinstance(x, (np.integer,)): return int(x)
    return x


def load_checkpoint(path,map_location):
    """Load our own checkpoint across PyTorch versions (2.6 changed the default)."""
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except TypeError: return torch.load(path,map_location=map_location)


def safe_corr(a, b):
    a=np.asarray(a,dtype=np.float64).reshape(-1); b=np.asarray(b,dtype=np.float64).reshape(-1)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12: return 0.0
    return float(np.corrcoef(a,b)[0,1])


def ccc_np(y, p):
    y=np.asarray(y,dtype=np.float64).reshape(-1); p=np.asarray(p,dtype=np.float64).reshape(-1)
    my,mp=y.mean(),p.mean(); vy,vp=np.var(y),np.var(p); cov=np.mean((y-my)*(p-mp))
    return float(2*cov/(vy+vp+(my-mp)**2+1e-8))


def metrics_np(y, p):
    y=np.asarray(y,dtype=np.float64).reshape(-1); p=np.asarray(p,dtype=np.float64).reshape(-1)
    if len(y)==0: return {"n":0}
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean((p-y)**2))),
        "mae": float(np.mean(np.abs(p-y))),
        "ccc": ccc_np(y,p),
        "corr": safe_corr(y,p),
        "pred_mean": float(p.mean()),
        "pred_std": float(np.std(p,ddof=1)) if len(p)>1 else 0.0,
        "gt_mean": float(y.mean()),
        "gt_std": float(np.std(y,ddof=1)) if len(y)>1 else 0.0,
        "bias": float(np.mean(p-y)),
    }


def bin_id(y):
    return 0 if y < 14 else (1 if y < 20 else (2 if y < 29 else 3))


def bin_name(y):
    return SEVERITY_NAMES[bin_id(y)]


def bin_metrics(df, pred_col="pred"):
    rows=[]; total=float(((df[pred_col]-df["gt"])**2).sum())+1e-8
    for b in SEVERITY_NAMES:
        g=df[df["bin"]==b]
        if len(g)==0: continue
        m=metrics_np(g["gt"], g[pred_col])
        row={"bin":b, "count":int(len(g)), **m, "sse_contribution":float(((g[pred_col]-g["gt"])**2).sum()/total)}
        for c in ["anchor_pred","anchor_error","residual","residual_raw","residual_gate","residual_target","residual_target_scaled","proto_score","proto_score_attn","seg_attn","top_score","low_mass","mid_mass","high_mass","p_low","attn_peak","attn_entropy","proto_entropy","proto_peak","abs_match","null_reject_rate","raw_sim_max"]:
            if c in g.columns: row[c]=float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Loading AU
# -----------------------------

def video_id_from_path(path):
    b=os.path.basename(path); b=re.sub(r"\.csv$","",b)
    for suf in ["_AUs","_aus","_AU","_pose","_gaze","_features","_openface","_video"]:
        if b.endswith(suf): b=b[:-len(suf)]
    return b


def metadata_from_path(path):
    stem=video_id_from_path(path)
    m=re.match(r"^(?P<pair>.+?)_(?P<task>Freeform|Northwind)$", stem, flags=re.I)
    if not m:
        raise ValueError(f"Cannot infer pair/task from filename: {path}")
    task="Freeform" if m.group("task").lower()=="freeform" else "Northwind"
    return stem, m.group("pair"), task


def read_txt(txt):
    rows=[]
    with open(txt,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            parts=line.strip().split()
            if len(parts)<2: raise ValueError(f"Bad manifest line in {txt}: {line!r}")
            p,y=parts[:2]; y=float(y)
            vid,inferred_pair,inferred_task=metadata_from_path(p)
            task=parts[2] if len(parts)>2 else inferred_task
            pair_id=parts[3] if len(parts)>3 else inferred_pair
            split=parts[4] if len(parts)>4 else ""
            if not SCORE_MIN <= y <= SCORE_MAX:
                raise ValueError(f"BDI-II score outside [0,63]: path={p}, score={y}")
            rows.append({"raw_path":p,"score":y,"video_id":vid,"pair_id":str(pair_id),
                         "task":str(task),"declared_split":str(split),"bin":bin_name(y)})
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError(f"Empty manifest: {txt}")
    dup=out["video_id"].duplicated(keep=False)
    if dup.any(): raise ValueError(f"Duplicate video_id in {txt}: {out.loc[dup,'video_id'].tolist()}")
    return out


def candidate_paths(raw_path, data_root=""):
    out=[]
    def add(p):
        if p and p not in out: out.append(p)
    add(raw_path)
    if data_root and not os.path.isabs(raw_path): add(os.path.join(data_root,raw_path))
    for p in list(out):
        d=os.path.dirname(p); vid=video_id_from_path(p)
        for a,b in [("_pose.csv","_AUs.csv"),("_gaze.csv","_AUs.csv"),("pose.csv","AUs.csv"),("gaze.csv","AUs.csv"),("pose","AUs"),("gaze","AUs")]:
            if a in p: add(p.replace(a,b))
        for pat in [os.path.join(d,f"{vid}*AU*.csv"), os.path.join(d,f"{vid}*au*.csv"), os.path.join(d,f"{vid}*_AUs.csv"), os.path.join(d,f"{vid}.csv")]:
            for q in glob.glob(pat): add(q)
    return out


def has_au_columns(csv_path):
    try:
        cols=[c.strip() for c in pd.read_csv(csv_path,nrows=1).columns]
        return any(c.startswith("AU") and c.endswith("_r") for c in cols)
    except Exception:
        return False


def find_au_file(raw_path, data_root=""):
    tried=candidate_paths(raw_path,data_root)
    for p in tried:
        if os.path.exists(p) and os.path.isfile(p) and has_au_columns(p): return p
    raise FileNotFoundError(f"Cannot find AU csv for {raw_path}; tried={tried[:12]}")


def load_au(path, au_cols, success_filter=True, allow_missing=False,
            confidence_threshold=0.55, resample_hz=30.0, max_interp_gap=0.25):
    """Load OpenFace features without creating artificial deltas across failed frames.

    Valid observations are interpolated onto a fixed time grid. Points farther than
    ``max_interp_gap`` from a valid observation are filled with the video's median and
    receive quality=0; the final feature channel is always OpenFace quality in [0,1].
    """
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    if len(df)==0: raise ValueError(f"Empty csv: {path}")
    missing=[c for c in au_cols if c not in df.columns]
    if missing and not allow_missing:
        raise ValueError(f"Missing required AU columns in {path}: {missing}")

    values=np.stack([
        pd.to_numeric(df[c],errors="coerce").fillna(0).to_numpy(np.float32)
        if c in df.columns else np.zeros(len(df),dtype=np.float32)
        for c in au_cols
    ],axis=1)
    success=(pd.to_numeric(df["success"],errors="coerce").fillna(0).to_numpy(float)
             if "success" in df.columns else np.ones(len(df),dtype=float))
    confidence=(pd.to_numeric(df["confidence"],errors="coerce").fillna(0).to_numpy(float)
                if "confidence" in df.columns else np.ones(len(df),dtype=float))
    valid=np.isfinite(values).all(1) & np.isfinite(confidence)
    if success_filter: valid &= success>=0.5
    valid &= confidence>=confidence_threshold
    if valid.sum()==0:
        raise ValueError(f"No valid OpenFace rows in {path} at confidence>={confidence_threshold}")

    if "timestamp" not in df.columns or resample_hz<=0:
        x=values[valid]
        q=np.clip(confidence[valid],0,1).astype(np.float32)[:,None]
        return np.concatenate([x,q],axis=1).astype(np.float32)

    timestamps=pd.to_numeric(df["timestamp"],errors="coerce").to_numpy(float)
    valid &= np.isfinite(timestamps)
    if valid.sum()==0: raise ValueError(f"No finite valid timestamps in {path}")
    t=timestamps[valid]; x=values[valid]; q=np.clip(confidence[valid],0,1)
    order=np.argsort(t); t=t[order]; x=x[order]; q=q[order]
    keep=np.r_[True,np.diff(t)>1e-8]; t=t[keep]; x=x[keep]; q=q[keep]
    if len(t)==1:
        return np.concatenate([x,q.astype(np.float32)[:,None]],axis=1).astype(np.float32)

    step=1.0/float(resample_hz)
    grid=np.arange(t[0],t[-1]+0.5*step,step,dtype=np.float64)
    out_cols=[]
    nearest_right=np.searchsorted(t,grid,side="left").clip(0,len(t)-1)
    nearest_left=np.maximum(nearest_right-1,0)
    choose_right=np.abs(grid-t[nearest_right])<np.abs(grid-t[nearest_left])
    nearest_idx=np.where(choose_right,nearest_right,nearest_left)
    for j,c in enumerate(au_cols):
        # Occurrence outputs are binary decisions; linear interpolation invents activations.
        out_cols.append(x[nearest_idx,j] if c.endswith("_c") else np.interp(grid,t,x[:,j]))
    out=np.stack(out_cols,axis=1)
    out_q=np.interp(grid,t,q)
    nearest=np.minimum(np.abs(grid-t[nearest_left]),np.abs(grid-t[nearest_right]))
    long_gap=nearest>max_interp_gap
    if long_gap.any():
        out[long_gap]=np.median(x,axis=0,keepdims=True)
        out_q[long_gap]=0.0
    return np.concatenate([out,np.clip(out_q,0,1)[:,None]],axis=1).astype(np.float32)


def fit_global_stats(arrays):
    # Equal-video moments: a four-minute recording must not dominate a six-second one.
    means=np.stack([x.mean(0) for x in arrays])
    seconds=np.stack([(x.astype(np.float64)**2).mean(0) for x in arrays])
    mean=means.mean(0,keepdims=True).astype(np.float32)
    var=np.maximum(seconds.mean(0,keepdims=True)-mean.astype(np.float64)**2,1e-12)
    return mean,np.sqrt(var).astype(np.float32)


# -----------------------------
# Descriptor anchor and sequence features
# -----------------------------

def task_code(task):
    return -1.0 if str(task)=="Freeform" else 1.0


def make_descriptor(x_raw, mean, std, task):
    x=np.clip(x_raw.astype(np.float32),0,5)
    z=((x-mean)/std).astype(np.float32)
    dz=np.diff(z,axis=0,prepend=z[:1]); adz=np.abs(dz)
    feats=[]
    for arr in [z,dz,adz]:
        feats += list(arr.mean(0)); feats += list(arr.std(0))
        feats += list(np.percentile(arr,10,axis=0)); feats += list(np.percentile(arr,50,axis=0)); feats += list(np.percentile(arr,90,axis=0))
        feats += list(arr.max(0)); feats += list(arr.min(0)); feats += list(np.abs(arr).mean(0))
    energy=np.sum(np.abs(z),axis=1); change=np.sum(np.abs(dz),axis=1)
    feats += [energy.mean(),energy.std(),energy.max(),energy.min(),change.mean(),change.std(),change.max(),change.min()]
    # compact AU co-occurrence summary
    co=[]
    for i in range(z.shape[1]):
        for j in range(i+1,z.shape[1]):
            co.append(np.mean((z[:,i]>0.5)&(z[:,j]>0.5)))
    co=np.asarray(co,dtype=np.float32)
    feats += [co.mean() if len(co) else 0, co.std() if len(co) else 0, co.max() if len(co) else 0]
    feats += [task_code(task)]
    return np.asarray(feats,dtype=np.float32)


def standardize(Xtr, *Xs):
    mu=Xtr.mean(0,keepdims=True); sig=np.maximum(Xtr.std(0,keepdims=True),1e-6)
    return [(Xtr-mu)/sig] + [(X-mu)/sig for X in Xs], mu, sig


def ridge_fit(X,y,alpha):
    X=np.asarray(X,np.float64); y=np.asarray(y,np.float64)
    Xb=np.concatenate([np.ones((len(X),1)),X],axis=1)
    reg=np.eye(Xb.shape[1])*alpha; reg[0,0]=0
    return np.linalg.solve(Xb.T@Xb+reg, Xb.T@y)


def ridge_pred(beta,X):
    X=np.asarray(X,np.float64); Xb=np.concatenate([np.ones((len(X),1)),X],axis=1)
    return np.clip(Xb@beta,SCORE_MIN,SCORE_MAX)


def fit_anchor(Xtr,ytr,Xdev,ydev,Xte,yte,groups,args):
    alphas=[float(a) for a in args.anchor_alphas.split(",") if a.strip()]
    rng=np.random.default_rng(args.seed)
    unique_groups=np.asarray(sorted(set(map(str,groups))))
    rng.shuffle(unique_groups)
    group_folds=np.array_split(unique_groups,min(args.anchor_folds,len(unique_groups)))
    folds=[np.flatnonzero(np.isin(np.asarray(groups).astype(str),val)) for val in group_folds]
    rows=[]; best=None
    for a in alphas:
        oof=np.zeros(len(ytr))
        for val in folds:
            tr=np.setdiff1d(np.arange(len(ytr)),val)
            beta=ridge_fit(Xtr[tr],ytr[tr],a); oof[val]=ridge_pred(beta,Xtr[val])
        m=metrics_np(ytr,oof); score=m["rmse"]-0.25*m["ccc"]-0.10*m["corr"]
        rows.append({"alpha":a,"score":score,**m})
        if best is None or score<best["score"]: best={"alpha":a,"score":score,"oof":oof,"metrics":m}
    beta=ridge_fit(Xtr,ytr,best["alpha"])
    tr_fit=ridge_pred(beta,Xtr); dev=ridge_pred(beta,Xdev); te=ridge_pred(beta,Xte)
    return {
        "alpha":float(best["alpha"]), "beta":beta.astype(np.float32),
        "train_oof":best["oof"].astype(np.float32), "train_fit":tr_fit.astype(np.float32),
        "dev":dev.astype(np.float32), "test":te.astype(np.float32),
        "cv_table":rows,
        "summary":{
            "selected_alpha":float(best["alpha"]),
            "train_oof":metrics_np(ytr,best["oof"]),
            "train_fit":metrics_np(ytr,tr_fit),
            "dev":metrics_np(ydev,dev),
            "test":"withheld_until_final_report",
        }
    }


def fit_anchor_mode(Xtr,ytr,Xdev,ydev,Xte,yte,tr_meta,dev_meta,te_meta,args):
    """Fit either one shared anchor or one train-only anchor per interview task."""
    if args.anchor_mode == "shared" or args.task_mode != "shared":
        a=fit_anchor(Xtr,ytr,Xdev,ydev,Xte,yte,tr_meta["pair_id"].values,args)
        return {"train_oof":a["train_oof"],"train_fit":a["train_fit"],
                "dev":a["dev"],"test":a["test"],
                "summary":{"mode":"shared","models":{"shared":a["summary"]}},
                "cv_tables":{"shared":a["cv_table"]},
                "models":{"shared":{"alpha":a["alpha"],"beta":a["beta"]}}}

    out={k:np.zeros(n,dtype=np.float32) for k,n in
         [("train_oof",len(ytr)),("train_fit",len(ytr)),
          ("dev",len(ydev)),("test",len(yte))]}
    summaries={}; cv_tables={}; models={}
    for task in ("Freeform","Northwind"):
        it=np.flatnonzero(tr_meta["task"].to_numpy()==task)
        iv=np.flatnonzero(dev_meta["task"].to_numpy()==task)
        ie=np.flatnonzero(te_meta["task"].to_numpy()==task)
        a=fit_anchor(Xtr[it],ytr[it],Xdev[iv],ydev[iv],Xte[ie],yte[ie],
                     tr_meta.iloc[it]["pair_id"].values,args)
        out["train_oof"][it]=a["train_oof"]; out["train_fit"][it]=a["train_fit"]
        out["dev"][iv]=a["dev"]; out["test"][ie]=a["test"]
        summaries[task]=a["summary"]; cv_tables[task]=a["cv_table"]
        models[task]={"alpha":a["alpha"],"beta":a["beta"]}
    out.update({"summary":{"mode":"task_specific","models":summaries},
                "cv_tables":cv_tables,"models":models})
    return out


def make_frame_features(x_raw, mean, std, args):
    x=np.clip(x_raw.astype(np.float32),0,args.au_clip)
    raw=((x-mean)/std).astype(np.float32)
    med=np.median(x,axis=0,keepdims=True).astype(np.float32)
    mad=np.median(np.abs(x-med),axis=0,keepdims=True).astype(np.float32)
    scale=np.maximum(mad,args.robust_eps)
    rel=np.clip((x-med)/scale,-args.rel_clip,args.rel_clip).astype(np.float32)
    dx=np.diff(x,axis=0,prepend=x[:1]).astype(np.float32)
    delta=np.clip(dx/scale,-args.delta_clip,args.delta_clip).astype(np.float32)
    abs_delta=np.clip(np.abs(dx)/scale,0,args.delta_clip).astype(np.float32)
    if args.input_mode=="raw": return raw
    if args.input_mode=="raw_rel": return np.concatenate([raw,rel],axis=1)
    return np.concatenate([raw,rel,delta,abs_delta],axis=1)


def make_segments(x, seq_len, stride, max_segments):
    T,C=x.shape; starts=list(range(0,max(1,T-seq_len+1),stride)); last=max(0,T-seq_len)
    if not starts or starts[-1]!=last: starts.append(last)
    starts=sorted(set(starts))
    if max_segments and len(starts)>max_segments:
        ids=np.linspace(0,len(starts)-1,max_segments).round().astype(int); starts=[starts[i] for i in ids]
    segs=[]; masks=[]
    for s in starts:
        seg=x[s:s+seq_len]; mask=np.ones(len(seg),dtype=np.float32)
        if len(seg)<seq_len:
            pad_len=seq_len-len(seg); pad=np.repeat(seg[-1:],pad_len,axis=0) if len(seg) else np.zeros((pad_len,C),dtype=np.float32)
            seg=np.concatenate([seg,pad],axis=0); mask=np.concatenate([mask,np.zeros(pad_len,dtype=np.float32)],axis=0)
        segs.append(seg.astype(np.float32)); masks.append(mask.astype(np.float32))
    return np.stack(segs), np.stack(masks)


@dataclass
class VideoItem:
    split: str; video_id: str; pair_id: str; task: str; raw_path: str; au_path: str; score: float; bin: str; anchor_pred: float
    segments: np.ndarray; masks: np.ndarray; desc: np.ndarray


def build_videos(meta, arrays, paths, anchors, mean, std, dmu, dsig, args, split):
    videos=[]
    for i,(_,r) in enumerate(tqdm(list(meta.iterrows()),desc=f"segment {split}",ncols=100)):
        feat=make_frame_features(arrays[i],mean,std,args)
        # A constant task channel lets the shared network learn task-conditioned evidence.
        feat=np.concatenate([feat,np.full((len(feat),1),task_code(r.task),dtype=np.float32)],axis=1)
        segs,masks=make_segments(feat,args.seq_len,args.stride,args.max_segments)
        desc=((make_descriptor(arrays[i],mean,std,r.task).reshape(-1)-dmu.reshape(-1))/dsig.reshape(-1)).astype(np.float32)
        videos.append(VideoItem(split,str(r.video_id),str(r.pair_id),str(r.task),str(r.raw_path),str(paths[i]),float(r.score),str(r.bin),float(anchors[i]),segs,masks,desc))
    print(f"[{split}] videos={len(videos)} | mean_segments={np.mean([len(v.segments) for v in videos]):.1f} | feat_dim={videos[0].segments.shape[-1]}")
    print(meta["bin"].value_counts().to_string())
    return videos


# -----------------------------
# Dataset
# -----------------------------

class ExprDataset(Dataset):
    def __init__(self,videos,train=True,args=None):
        self.videos=videos; self.train=train; self.args=args
    def __len__(self): return len(self.videos)
    def __getitem__(self,idx):
        v=self.videos[idx]; segs=v.segments; masks=v.masks
        if self.train:
            if self.args.max_train_segments_per_video>0 and len(segs)>self.args.max_train_segments_per_video:
                ids=np.sort(np.random.choice(len(segs),self.args.max_train_segments_per_video,replace=False)); segs=segs[ids]; masks=masks[ids]
            if len(segs)>4 and self.args.segment_keep_ratio<1:
                keep=max(4,int(round(len(segs)*self.args.segment_keep_ratio)))
                ids=np.sort(np.random.choice(len(segs),keep,replace=False)); segs=segs[ids]; masks=masks[ids]
            segs=segs.copy()
            if self.args.noise_std>0: segs += np.random.normal(0,self.args.noise_std,segs.shape).astype(np.float32)
            if self.args.feature_dropout>0:
                segs *= (np.random.rand(segs.shape[0],1,segs.shape[2])>self.args.feature_dropout).astype(np.float32)
            if self.args.time_mask_prob>0 and np.random.rand()<self.args.time_mask_prob:
                mlen=max(1,int(segs.shape[1]*self.args.time_mask_ratio))
                for s in range(segs.shape[0]):
                    st=np.random.randint(0,max(1,segs.shape[1]-mlen+1)); segs[s,st:st+mlen]=segs[s].mean(0,keepdims=True)
        return {
            "x":torch.from_numpy(segs).float(), "mask":torch.from_numpy(masks).bool(),
            "desc":torch.from_numpy(v.desc).float(), "anchor":torch.tensor(v.anchor_pred,dtype=torch.float32),
            "y":torch.tensor(v.score,dtype=torch.float32), "video_id":v.video_id,
            "pair_id":v.pair_id,"task":v.task,"bin":v.bin
        }


def collate(batch):
    B=len(batch); S=max(b["x"].shape[0] for b in batch); T=batch[0]["x"].shape[1]; C=batch[0]["x"].shape[2]; D=batch[0]["desc"].reshape(-1).shape[0]
    x=torch.zeros(B,S,T,C); mask=torch.zeros(B,S,T,dtype=torch.bool); bag=torch.zeros(B,S,dtype=torch.bool)
    desc=torch.zeros(B,D); y=torch.zeros(B); anchor=torch.zeros(B); vids=[]; pairs=[]; tasks=[]; bins=[]
    for i,b in enumerate(batch):
        s=b["x"].shape[0]; x[i,:s]=b["x"]; mask[i,:s]=b["mask"]; bag[i,:s]=True
        desc[i]=b["desc"].reshape(-1); y[i]=b["y"]; anchor[i]=b["anchor"]
        vids.append(b["video_id"]); pairs.append(b["pair_id"]); tasks.append(b["task"]); bins.append(b["bin"])
    return {"x":x,"mask":mask,"bag_mask":bag,"desc":desc,"y":y,"anchor":anchor,
            "video_id":vids,"pair_id":pairs,"task":tasks,"bin":bins}


class BalancedSampler(Sampler):
    def __init__(self,videos,batch_size,steps,seed):
        self.bs=batch_size; self.steps=steps; self.seed=seed; self.epoch=0
        self.by={0:[],1:[],2:[],3:[]}
        for i,v in enumerate(videos): self.by[bin_id(v.score)].append(i)
        print("[BatchSampler] counts:",{k:len(v) for k,v in self.by.items()})
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); self.epoch+=1
        active=[k for k,pool in self.by.items() if pool]
        quota=[0,0,0,0]
        for j in range(self.bs): quota[active[j%len(active)]]+=1
        for _ in range(self.steps):
            batch=[]
            for k,q in enumerate(quota):
                pool=self.by[k]
                if pool: batch += rng.choice(pool,size=q,replace=len(pool)<q).tolist()
            rng.shuffle(batch); yield batch


class PairBalancedSampler(Sampler):
    """Severity-balanced batches that keep Freeform/Northwind pairs together."""
    def __init__(self,videos,batch_size,steps,seed):
        if batch_size%2: raise ValueError("shared task_mode requires an even batch_size")
        self.bs=batch_size; self.steps=steps; self.seed=seed; self.epoch=0
        grouped={}
        for i,v in enumerate(videos): grouped.setdefault(v.pair_id,[]).append(i)
        self.by={0:[],1:[],2:[],3:[]}
        for pair,ids in grouped.items():
            tasks={videos[i].task for i in ids}
            if len(ids)!=2 or tasks!={"Freeform","Northwind"}:
                raise ValueError(f"Incomplete training pair {pair}: ids={ids}, tasks={sorted(tasks)}")
            self.by[bin_id(videos[ids[0]].score)].append(ids)
        print("[PairBatchSampler] counts:",{k:len(v) for k,v in self.by.items()})
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); self.epoch+=1
        active=[k for k,pool in self.by.items() if pool]
        n_pairs=self.bs//2; quota=[0,0,0,0]
        for j in range(n_pairs): quota[active[j%len(active)]]+=1
        for _ in range(self.steps):
            batch=[]
            for k,q in enumerate(quota):
                pool=self.by[k]
                if pool:
                    for j in rng.choice(len(pool),size=q,replace=len(pool)<q): batch.extend(pool[int(j)])
            # Shuffle pair order but retain adjacent pair members.
            chunks=[batch[i:i+2] for i in range(0,len(batch),2)]; rng.shuffle(chunks)
            yield [i for chunk in chunks for i in chunk]


# -----------------------------
# Model
# -----------------------------

class SegmentEncoder(nn.Module):
    def __init__(self,input_dim,hid=64,K=10,cap=14.0,drop=0.08,tau=0.16,
                 match_center=0.15,match_scale=0.10,frame_match_weight=0.35):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(input_dim,hid),nn.LayerNorm(hid),nn.GELU())
        self.conv=nn.Sequential(nn.Conv1d(hid,hid,3,padding=1),nn.BatchNorm1d(hid),nn.GELU(),nn.Dropout(drop))
        self.gru=nn.GRU(hid,hid//2,batch_first=True,bidirectional=True)
        p=torch.empty(K,hid); nn.init.orthogonal_(p); self.proto=nn.Parameter(p)
        self.register_buffer("scores",torch.linspace(0,cap,K)); self.tau=tau
        self.match_center=float(match_center); self.match_scale=float(match_scale)
        self.frame_match_weight=float(frame_match_weight)
        self.attn=nn.Sequential(nn.Linear(hid+K+2,hid//2),nn.Tanh(),nn.Linear(hid//2,1))
        self.offset=nn.Sequential(nn.Linear(hid+K+4,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1),nn.Tanh())
    def forward(self,x,mask):
        h=self.proj(x); h=self.conv(h.transpose(1,2)).transpose(1,2); h,_=self.gru(h)
        z=F.normalize(h,dim=-1); p=F.normalize(self.proto,dim=1)
        sim=torch.matmul(z,p.T)
        # Explicit null/background state.  The K expression prototypes compete
        # against a fixed rejection logit; therefore a neutral/unmatched frame
        # is allowed to assign most probability to "none of the prototypes".
        # +log(K) prevents the K expression alternatives from winning merely
        # because there are K of them.  At max_sim ~= match_center, expression
        # versus null evidence is approximately balanced.
        expr_logits=sim/self.tau
        w=torch.softmax(expr_logits,dim=-1)
        # Corrected aggregate expression evidence competes with the null state.
        # At similarity==match_center for all K prototypes, match_logit==0.
        match_scale=max(self.match_scale,1e-4)
        match_logit=torch.logsumexp(sim/match_scale,dim=-1)-math.log(sim.shape[-1])-self.match_center/match_scale
        expr_mass=torch.sigmoid(match_logit)
        expr_prob=w*expr_mass.unsqueeze(-1)
        proto_frame=(w*self.scores.view(1,1,-1)).sum(-1)
        peak=w.max(-1).values
        ent=-(w.clamp_min(1e-8)*torch.log(w.clamp_min(1e-8))).sum(-1)/math.log(w.shape[-1])
        max_sim=sim.max(-1).values
        # V2 only used relative softmax confidence. V3 rejects frames whose absolute
        # cosine similarity is low even when softmax is artificially sharp.
        abs_conf=expr_mass
        conf=(abs_conf*peak*(1-0.35*ent).clamp_min(0.05)).clamp_min(1e-4)
        logits=self.attn(torch.cat([h,w,(proto_frame/self.scores.max()).unsqueeze(-1),conf.unsqueeze(-1)],-1)).squeeze(-1)
        logits=logits+self.frame_match_weight*torch.log(abs_conf.clamp_min(1e-4))
        logits=logits.masked_fill(~mask.bool(),-1e9)
        a=torch.softmax(logits,dim=1)
        emb=(h*a.unsqueeze(-1)).sum(1)
        # Prototype statistics are conditional on the expression-matched mass.
        # General temporal embedding still sees every valid frame, but neutral
        # frames do not dilute or fabricate the expression histogram.
        expr_a=a*abs_conf; expr_den=expr_a.sum(1,keepdim=True).clamp_min(1e-6)
        hist=(w*expr_a.unsqueeze(-1)).sum(1)/expr_den
        valid=mask.float(); den=valid.sum(1).clamp_min(1)
        pscore=(hist*self.scores.view(1,-1)).sum(1)
        pc=(abs_conf*valid).sum(1)/den; pe=(ent*valid).sum(1)/den; pp=(peak*valid).sum(1)/den
        pm=(max_sim*valid).sum(1)/den
        pr=(((abs_conf<0.5).float()*valid).sum(1)/den)
        # 2017 used +/-2 on a 0..14 expression lexicon. Keep the same normalized span.
        off_scale=float(self.scores.max())/7.0
        off=self.offset(torch.cat([emb,hist,(pscore/self.scores.max()).unsqueeze(-1),pc.unsqueeze(-1),pe.unsqueeze(-1),pp.unsqueeze(-1)],1)).view(-1)*off_scale
        seg=torch.clamp(pscore+off,0,float(self.scores.max()))
        return emb,hist,seg,pscore,pc,pe,pp,pm,pr


class AEE2014(nn.Module):
    def __init__(self,input_dim,desc_dim,hid=64,K=10,cap=1.0,residual_max=20.0,drop=0.08,tau=0.16,
                 attn_temp=0.45,max_score=63.0,low_cut=14.0,match_center=0.15,
                 match_scale=0.10,frame_match_weight=0.35,segment_match_weight=0.25):
        super().__init__()
        self.enc=SegmentEncoder(input_dim,hid,K,cap,drop,tau,match_center,match_scale,frame_match_weight)
        # V4 component diagnosis can retrain the same architecture with uniform
        # window pooling.  It is deliberately a runtime flag so old V3.1
        # checkpoints remain loadable and the learned-attention path is unchanged.
        self.force_uniform_attention=False
        self.residual_max=residual_max; self.attn_temp=attn_temp; self.segment_match_weight=float(segment_match_weight)
        self.max_score=float(max_score); self.low_cut=float(low_cut)
        self.register_buffer("proto_scores",torch.linspace(0,cap,K))
        self.desc=nn.Sequential(nn.Linear(desc_dim,hid),nn.LayerNorm(hid),nn.GELU(),nn.Dropout(drop))
        self.seg_attn=nn.Sequential(nn.Linear(hid+K+5,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1))
        # 19 original video statistics + 10 temporal primitive statistics.
        vdim=hid*2+K*3+hid+29
        self.res_head=nn.Sequential(nn.Linear(vdim,hid),nn.LayerNorm(hid),nn.GELU(),nn.Dropout(drop),nn.Linear(hid,hid//2),nn.GELU(),nn.Linear(hid//2,1),nn.Tanh())
        self.low_head=nn.Sequential(nn.Linear(vdim,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1))
        self.res_gate=nn.Sequential(nn.Linear(vdim,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1))
        nn.init.constant_(self.res_gate[-1].bias,-1.0)
        self.mil_res=nn.Sequential(nn.Linear(9,18),nn.GELU(),nn.Linear(18,1),nn.Tanh())

    @staticmethod
    def temporal_stats(seg,pc,hist,bag):
        """Differentiable persistence/transition/spectral summaries over ordered windows."""
        rows=[]
        for i in range(seg.shape[0]):
            n=max(1,int(bag[i].sum().item())); s=seg[i,:n]; q=pc[i,:n]; h=hist[i,:n]
            zero=s.new_tensor(0.0)
            d1=(s[1:]-s[:-1]).abs().mean() if n>1 else zero
            d2=(s[2:]-s[:-2]).abs().mean() if n>2 else zero
            hd=(h[1:]-h[:-1]).abs().mean() if n>1 else zero
            persist=torch.exp(-5.0*(s[1:]-s[:-1]).abs()).mean() if n>1 else s.new_tensor(1.0)
            # Fixed sinusoidal projections avoid cuFFT, which is unstable on
            # some PyTorch/CUDA/cuFFT combinations.  They retain differentiable
            # low/mid/high temporal energy without changing feature dimensions.
            centered=s-s.mean()
            freqs=s.new_tensor([0.04,0.08,0.12,0.16,0.22,0.28,0.34,0.42,0.49])
            tt=torch.arange(n,device=s.device,dtype=s.dtype).unsqueeze(1)
            phase=2*math.pi*tt*freqs.unsqueeze(0)
            cp=(centered.unsqueeze(1)*torch.cos(phase)).sum(0)
            sp=(centered.unsqueeze(1)*torch.sin(phase)).sum(0)
            power=(cp.square()+sp.square())/max(float(n*n),1.0)
            total=power.sum().clamp_min(1e-8)
            low=power[:3].sum()/total; mid=power[3:6].sum()/total; high=power[6:].sum()/total
            centroid=(freqs*power).sum()/total
            rows.append(torch.stack([d1,d2,hd,persist,q.mean(),q.std(unbiased=False),low,mid,high,centroid]))
        return torch.stack(rows)
    def forward(self,x,mask,bag,desc,anchor):
        B,S,T,D=x.shape; cap=self.proto_scores.max().clamp_min(1)
        emb,hist,seg,pscore,pc,pe,pp,pm,pr=self.enc(x.view(B*S,T,D),mask.view(B*S,T))
        H=emb.shape[-1]; K=hist.shape[-1]
        emb=emb.view(B,S,H); hist=hist.view(B,S,K); seg=seg.view(B,S); pscore=pscore.view(B,S); pc=pc.view(B,S); pe=pe.view(B,S); pp=pp.view(B,S); pm=pm.view(B,S); pr=pr.view(B,S)
        valid=bag.float(); den=valid.sum(1).clamp_min(1)
        logits=self.seg_attn(torch.cat([emb,hist,(seg/cap).unsqueeze(-1),(pscore/cap).unsqueeze(-1),pc.unsqueeze(-1),pe.unsqueeze(-1),pp.unsqueeze(-1)],-1)).squeeze(-1)
        # Do not hard-code high severity as high saliency. Saliency follows absolute
        # library match and the learned evidence network.
        logits=logits+self.segment_match_weight*torch.log(pc.clamp_min(1e-4))
        logits=logits.masked_fill(~bag.bool(),-1e9)
        if self.force_uniform_attention:
            a=valid/den.unsqueeze(1)
        else:
            a=torch.softmax(logits/self.attn_temp,dim=1); a=a*valid; a=a/a.sum(1,keepdim=True).clamp_min(1e-8)
        mean_emb=(emb*valid.unsqueeze(-1)).sum(1)/den.unsqueeze(-1); att_emb=(emb*a.unsqueeze(-1)).sum(1)
        mean_hist=(hist*valid.unsqueeze(-1)).sum(1)/den.unsqueeze(-1); att_hist=(hist*a.unsqueeze(-1)).sum(1)
        top_h=[]; top_s=[]; low_s=[]
        for i in range(B):
            idx=torch.nonzero(bag[i],as_tuple=False).view(-1); vals=seg[i,idx]; k=max(1,int(math.ceil(0.2*len(vals))))
            hi=idx[torch.topk(vals,k=k).indices]; lo=idx[torch.topk(-vals,k=k).indices]
            top_h.append(hist[i,hi].mean(0)); top_s.append(seg[i,hi].mean()); low_s.append(seg[i,lo].mean())
        top_hist=torch.stack(top_h); top_score=torch.stack(top_s); seg_low=torch.stack(low_s)
        seg_mean=(seg*valid).sum(1)/den; seg_attn=(seg*a).sum(1); seg_var=(((seg-seg_mean.unsqueeze(1))**2)*valid).sum(1)/den; seg_std=torch.sqrt(seg_var+1e-6)
        proto_entropy=(pe*valid).sum(1)/den; proto_peak=(pp*valid).sum(1)/den
        attn_peak=a.max(1).values; attn_entropy=-(a.clamp_min(1e-8)*torch.log(a.clamp_min(1e-8))).sum(1)/torch.log(den.clamp_min(2))
        low_edge=0.32*cap; high_edge=0.68*cap
        low_mask=(self.proto_scores<=low_edge).float(); mid_mask=((self.proto_scores>low_edge)&(self.proto_scores<high_edge)).float(); high_mask=(self.proto_scores>=high_edge).float()
        low_mass=(mean_hist*low_mask.view(1,-1)).sum(1); mid_mass=(mean_hist*mid_mask.view(1,-1)).sum(1); high_mass=(mean_hist*high_mask.view(1,-1)).sum(1)
        proto_score=(mean_hist*self.proto_scores.view(1,-1)).sum(1); proto_score_attn=(att_hist*self.proto_scores.view(1,-1)).sum(1)
        temporal=self.temporal_stats(seg,pc,hist,bag)
        d=self.desc(desc); an=(anchor/self.max_score).clamp(0,1)
        mil=torch.stack([an,proto_score/cap,proto_score_attn/cap,seg_mean/cap,seg_attn/cap,top_score/cap,low_mass,mid_mass,high_mass],1)
        stats=torch.stack([an,proto_score/cap,proto_score_attn/cap,seg_mean/cap,seg_attn/cap,top_score/cap,seg_low/cap,seg_std/cap,low_mass,mid_mass,high_mass,proto_entropy,proto_peak,attn_peak,attn_entropy,den/float(S),torch.relu(anchor-self.low_cut)/max(self.max_score-self.low_cut,1.0),torch.relu(self.low_cut-anchor)/max(self.low_cut,1.0), (att_hist-mean_hist).abs().sum(1)/2],1)
        stats=torch.cat([stats,temporal],1)
        vf=torch.cat([att_emb,mean_emb,att_hist,mean_hist,top_hist,d,stats],1)
        r1=self.residual_max*self.res_head(vf).view(-1); r2=self.residual_max*self.mil_res(mil).view(-1)
        residual_raw=0.75*r1+0.25*r2
        residual_gate=torch.sigmoid(self.res_gate(vf).view(-1))
        residual=residual_gate*residual_raw
        pred=torch.clamp(anchor+residual,SCORE_MIN,self.max_score); p_low=torch.sigmoid(self.low_head(vf).view(-1))
        return {"pred":pred,"residual":residual,"residual_raw":residual_raw,"residual_gate":residual_gate,"residual_deep":r1,"residual_mil":r2,"p_low":p_low,"seg_mean":seg_mean,"seg_attn":seg_attn,"top_score":top_score,"seg_low":seg_low,"seg_std":seg_std,"attn":a,"attn_hist":att_hist,"mean_hist":mean_hist,"proto_entropy":proto_entropy,"proto_peak":proto_peak,"attn_peak":attn_peak,"attn_entropy":attn_entropy,"low_mass":low_mass,"mid_mass":mid_mass,"high_mass":high_mass,"proto_score":proto_score,"proto_score_attn":proto_score_attn,"abs_match":((pc*valid).sum(1)/den),"null_reject_rate":((pr*valid).sum(1)/den),"raw_sim_max":((pm*valid).sum(1)/den),"temporal_stats":temporal}


# -----------------------------
# Loss / eval
# -----------------------------

def ccc_loss(pred,y):
    if pred.numel()<2: return pred.new_tensor(0.)
    x=pred.view(-1); t=y.view(-1); xm=x.mean(); tm=t.mean()
    cov=((x-xm)*(t-tm)).mean()
    return 1-2*cov/(torch.var(x,unbiased=False)+torch.var(t,unbiased=False)+(xm-tm)**2+1e-8)


def corr_loss(pred,y):
    if pred.numel()<2: return pred.new_tensor(0.)
    x=pred-pred.mean(); t=y-y.mean()
    return 1-torch.sum(x*t)/torch.sqrt(torch.sum(x*x)*torch.sum(t*t)+1e-8)


def rank_loss(pred,y,margin,min_gap):
    if pred.numel()<2: return pred.new_tensor(0.)
    yi=y.view(-1,1); yj=y.view(1,-1); pi=pred.view(-1,1); pj=pred.view(1,-1)
    sign=torch.sign(yi-yj); valid=torch.abs(yi-yj)>=min_gap
    if valid.sum()==0: return pred.new_tensor(0.)
    return F.relu(margin-sign*(pi-pj))[valid].mean()


def ordinal_target(y,scores,sigma):
    y=y.view(-1,1); s=scores.view(1,-1)
    t=torch.exp(-0.5*((s-y)/sigma)**2); return t/t.sum(1,keepdim=True).clamp_min(1e-8)


def proto_div_loss(proto):
    p=F.normalize(proto,dim=1); sim=p@p.T; eye=torch.eye(sim.shape[0],device=sim.device)
    return ((sim-eye)**2).mean()


def residual_target_scaled_np(y, anchor, args):
    raw = float(np.clip(y - anchor, -args.residual_max, args.residual_max))
    if y < args.minimal_cut:
        scale = args.residual_minimal_scale
    elif y < args.mild_cut:
        scale = args.residual_mild_scale
    elif y < args.moderate_cut:
        scale = args.residual_moderate_scale
    else:
        scale = args.residual_severe_scale
    return float(np.clip(raw * scale, -args.residual_max, args.residual_max))

def loss_fn(out,y,anchor,model,args,epoch,pair_ids=None):
    pred=out["pred"]; y=y.float(); anchor=anchor.float()
    # Prototype scores are latent normalized severity, never a linearly truncated PHQ score.
    y_expr=torch.clamp(y/args.score_max,0,1)

    # V17: residual target is still derived from train-only anchor, but supervision is strengthened
    # in a controlled way. We do not force full gt-anchor correction; instead we use bin-wise
    # target scaling and directional push losses.
    raw_rt=torch.clamp(y-anchor,-args.residual_max,args.residual_max)
    scale=torch.ones_like(y)*args.residual_target_scale
    scale=torch.where(y<args.minimal_cut, torch.ones_like(y)*args.residual_minimal_scale, scale)
    scale=torch.where((y>=args.minimal_cut)&(y<args.mild_cut), torch.ones_like(y)*args.residual_mild_scale, scale)
    scale=torch.where((y>=args.mild_cut)&(y<args.moderate_cut), torch.ones_like(y)*args.residual_moderate_scale, scale)
    scale=torch.where(y>=args.moderate_cut, torch.ones_like(y)*args.residual_severe_scale, scale)
    rt=torch.clamp(raw_rt*scale,-args.residual_max,args.residual_max)

    ramp=1.0 if args.library_warmup_epochs<=1 else min(1.0,epoch/float(args.library_warmup_epochs))
    w=torch.ones_like(y)
    w=torch.where(y<args.minimal_cut,w*args.loss_minimal_weight,w); w=torch.where((y>=args.minimal_cut)&(y<args.mild_cut),w*args.loss_mild_weight,w)
    w=torch.where((y>=args.mild_cut)&(y<args.moderate_cut),w*args.loss_moderate_weight,w); w=torch.where(y>=args.moderate_cut,w*args.loss_severe_weight,w)
    w=w/w.mean().clamp_min(1e-6)
    huber=(F.smooth_l1_loss(pred,y,beta=args.huber_beta,reduction="none")*w).mean()
    cl=ccc_loss(pred,y); cr=corr_loss(pred,y); rl=rank_loss(pred,y,args.rank_margin,args.rank_min_gap)

    res_w=torch.ones_like(y)
    res_w=torch.where(y<args.minimal_cut,res_w*args.residual_minimal_weight,res_w)
    res_w=torch.where((y>=args.minimal_cut)&(y<args.mild_cut),res_w*args.residual_mild_weight,res_w)
    res_w=torch.where((y>=args.mild_cut)&(y<args.moderate_cut),res_w*args.residual_moderate_weight,res_w)
    res_w=torch.where(y>=args.moderate_cut,res_w*args.residual_severe_weight,res_w)
    res_w=res_w*(1.0+args.residual_magnitude_weight*(raw_rt.abs()/max(args.residual_max,1e-6)))
    res_w=res_w/res_w.mean().clamp_min(1e-6)
    res_loss=(F.smooth_l1_loss(out["residual"],rt,beta=args.residual_beta,reduction="none")*res_w).mean()
    res_l2=(out["residual"]**2).mean()

    # Direction follows sign(gt-anchor), not severity class: an overestimating anchor must move down.
    minimal=y<args.minimal_cut; mild=(y>=args.minimal_cut)&(y<args.mild_cut); moderate=(y>=args.mild_cut)&(y<args.moderate_cut); severe=y>=args.moderate_cut
    push=torch.clamp(rt.abs()*args.residual_push_fraction, min=0.0, max=args.residual_max)
    valid_direction=raw_rt.abs()>0.5
    dir_loss=(torch.relu(push-torch.sign(raw_rt)*out["residual"]).pow(2)[valid_direction].mean()
              if valid_direction.any() else pred.new_tensor(0.))

    # Directly address the observed moderate underestimation without using a separate boost head.
    mod_under=torch.relu(y[moderate]-pred[moderate]-args.moderate_under_margin).pow(2).mean() if moderate.any() else pred.new_tensor(0.)
    severe_under=torch.relu(y[severe]-pred[severe]-args.severe_under_margin).pow(2).mean() if severe.any() else pred.new_tensor(0.)

    low_bce=F.binary_cross_entropy(out["p_low"],(y<args.minimal_cut).float())
    low_over=torch.relu(pred[minimal]-y[minimal]-args.low_over_margin).pow(2).mean() if minimal.any() else pred.new_tensor(0.)
    target=ordinal_target(y_expr,model.proto_scores,args.ordinal_sigma)
    mh=out["mean_hist"].clamp_min(1e-8); ah=out["attn_hist"].clamp_min(1e-8)
    kl_mean=(target*(torch.log(target.clamp_min(1e-8))-torch.log(mh))).sum(1).mean()
    kl_attn=(target*(torch.log(target.clamp_min(1e-8))-torch.log(ah))).sum(1).mean()
    pkl=args.proto_mean_kl_weight*kl_mean+args.proto_attn_kl_weight*kl_attn
    # The attended library path must carry most of the severity supervision; V2
    # accidentally rewarded the almost-uniform video mean more strongly.
    pscore=0.25*F.smooth_l1_loss(out["proto_score"],y_expr,beta=0.05)+0.75*F.smooth_l1_loss(out["proto_score_attn"],y_expr,beta=0.05)
    aux=0.5*F.smooth_l1_loss(out["seg_attn"],y_expr,beta=0.08)+0.5*F.smooth_l1_loss(out["top_score"],y_expr,beta=0.08)
    div=proto_div_loss(model.enc.proto); attn=out["attn_entropy"].mean(); pent=out["proto_entropy"].mean()
    attn_uniform=torch.relu(out["attn_entropy"]-args.attn_entropy_target).pow(2).mean()
    gate_target=(raw_rt.abs()/max(args.residual_max,1e-6)).clamp(0,1)
    gate_loss=F.smooth_l1_loss(out["residual_gate"],gate_target,beta=0.10)
    dist=pred.new_tensor(0.)
    if pred.numel()>1:
        scale=max(args.score_max/3.0,1.0)
        dist=((pred.mean()-y.mean())/scale).pow(2)+0.25*((pred.std(unbiased=False)-y.std(unbiased=False))/scale).pow(2)
    pair_loss=pred.new_tensor(0.)
    if pair_ids is not None:
        pair_groups=[]
        for pair_id in dict.fromkeys(map(str,pair_ids)):
            ids=[i for i,p in enumerate(pair_ids) if str(p)==pair_id]
            if len(ids)>1: pair_groups.append(ids)
        if pair_groups:
            pp=torch.stack([pred[ids].mean() for ids in pair_groups])
            yy=torch.stack([y[ids].mean() for ids in pair_groups])
            pair_loss=F.smooth_l1_loss(pp,yy,beta=args.huber_beta)+0.25*ccc_loss(pp,yy)
    loss=(args.w_huber*huber+args.w_ccc*cl+args.w_corr*cr+args.w_rank*rl+
          args.w_residual*res_loss+args.w_res_l2*res_l2+args.w_res_direction*dir_loss+
          args.w_moderate_under*mod_under+args.w_severe_under*severe_under+
          args.w_low_bce*low_bce+args.w_low_over*low_over+args.w_aux*aux+
          ramp*(args.w_proto_ordinal*pkl+args.w_proto_score*pscore+
                args.w_proto_diversity*div+args.w_attn_focus*attn+
                args.w_attn_uniform*attn_uniform+args.w_proto_entropy*pent)+
          args.w_gate*gate_loss+
          args.w_dist*dist+args.w_pair*pair_loss)
    return loss, {"loss":float(loss.detach().cpu()),"pred_std":float(pred.detach().cpu().std(unbiased=False)) if pred.numel()>1 else 0.0,"residual_std":float(out["residual"].detach().cpu().std(unbiased=False)) if pred.numel()>1 else 0.0,"residual_mean":float(out["residual"].detach().cpu().mean()),"residual_gate":float(out["residual_gate"].detach().cpu().mean()),"abs_match":float(out["abs_match"].detach().cpu().mean()),"res_loss":float(res_loss.detach().cpu()),"gate_loss":float(gate_loss.detach().cpu()),"dir_loss":float(dir_loss.detach().cpu()),"mod_under":float(mod_under.detach().cpu()),"severe_under":float(severe_under.detach().cpu()),"pair_loss":float(pair_loss.detach().cpu()),"proto_kl":float(pkl.detach().cpu()),"proto_score":float(pscore.detach().cpu()),"attn_uniform":float(attn_uniform.detach().cpu()),"ramp":float(ramp)}


@torch.no_grad()
def evaluate(model,videos,args,split):
    model.eval(); rows=[]
    for v in tqdm(videos,desc=f"eval {split}",ncols=100,leave=False):
        b=collate([ExprDataset([v],train=False,args=args)[0]])
        out=model(b["x"].to(DEVICE),b["mask"].to(DEVICE),b["bag_mask"].to(DEVICE),b["desc"].to(DEVICE),b["anchor"].to(DEVICE))
        y=float(v.score); a=float(v.anchor_pred)
        row={"split":split,"video_id":v.video_id,"pair_id":v.pair_id,"task":v.task,"gt":y,"bin":v.bin,"anchor_pred":a,"anchor_error":a-y,"residual_target":float(np.clip(y-a,-args.residual_max,args.residual_max)),"residual_target_scaled":residual_target_scaled_np(y,a,args),
             "pred":float(out["pred"].cpu().numpy()[0]),"residual":float(out["residual"].cpu().numpy()[0]),"residual_raw":float(out["residual_raw"].cpu().numpy()[0]),"residual_gate":float(out["residual_gate"].cpu().numpy()[0]),"residual_deep":float(out["residual_deep"].cpu().numpy()[0]),"residual_mil":float(out["residual_mil"].cpu().numpy()[0]),
             "p_low":float(out["p_low"].cpu().numpy()[0]),"proto_score":float(out["proto_score"].cpu().numpy()[0]),"proto_score_attn":float(out["proto_score_attn"].cpu().numpy()[0]),
             "seg_mean":float(out["seg_mean"].cpu().numpy()[0]),"seg_attn":float(out["seg_attn"].cpu().numpy()[0]),"top_score":float(out["top_score"].cpu().numpy()[0]),"seg_low":float(out["seg_low"].cpu().numpy()[0]),
             "attn_peak":float(out["attn_peak"].cpu().numpy()[0]),"attn_entropy":float(out["attn_entropy"].cpu().numpy()[0]),"proto_entropy":float(out["proto_entropy"].cpu().numpy()[0]),"proto_peak":float(out["proto_peak"].cpu().numpy()[0]),
             "low_mass":float(out["low_mass"].cpu().numpy()[0]),"mid_mass":float(out["mid_mass"].cpu().numpy()[0]),"high_mass":float(out["high_mass"].cpu().numpy()[0]),"abs_match":float(out["abs_match"].cpu().numpy()[0]),"null_reject_rate":float(out["null_reject_rate"].cpu().numpy()[0]),"raw_sim_max":float(out["raw_sim_max"].cpu().numpy()[0]),"n_segments":len(v.segments)}
        for k,val in enumerate(out["temporal_stats"].cpu().numpy()[0]): row[f"temporal{k:02d}"]=float(val)
        hist=out["mean_hist"].cpu().numpy()[0]
        for k,val in enumerate(hist): row[f"proto{k:02d}_mean"]=float(val)
        rows.append(row)
    df=pd.DataFrame(rows); return df,metrics_np(df["gt"],df["pred"]),bin_metrics(df)


def paired_predictions(df):
    """Equal-average Freeform/Northwind predictions for each AVEC2014 pair."""
    rows=[]
    for pair_id,g in df.groupby("pair_id",sort=True):
        if g["gt"].nunique()!=1:
            raise ValueError(f"Different labels within pair {pair_id}: {g['gt'].tolist()}")
        tasks=set(g["task"].astype(str))
        if tasks!={"Freeform","Northwind"}:
            raise ValueError(f"Pair {pair_id} must contain Freeform and Northwind, got {sorted(tasks)}")
        y=float(g["gt"].iloc[0])
        rows.append({"split":str(g["split"].iloc[0]),"pair_id":str(pair_id),"gt":y,
                     "bin":bin_name(y),"pred":float(g["pred"].mean()),
                     "anchor_pred":float(g["anchor_pred"].mean()),
                     "freeform_pred":float(g.loc[g["task"]=="Freeform","pred"].iloc[0]),
                     "northwind_pred":float(g.loc[g["task"]=="Northwind","pred"].iloc[0])})
    return pd.DataFrame(rows)


def metrics_by_view(df):
    pair_counts=df.groupby("pair_id").size()
    pair_tasks=df.groupby("pair_id")["task"].apply(lambda x:set(map(str,x)))
    has_complete_pairs=(len(pair_counts)>0 and (pair_counts==2).all() and
                        pair_tasks.apply(lambda x:x=={"Freeform","Northwind"}).all())
    paired=paired_predictions(df) if has_complete_pairs else None
    views={"clips_pooled":df}
    if paired is not None: views={"paired_mean":paired,**views}
    for task in ("Freeform","Northwind"):
        g=df[df["task"]==task]
        if len(g): views[task]=g
    return {name:metrics_np(v["gt"],v["pred"]) for name,v in views.items()}, paired


def save_eval_bundle(df,save_dir,split):
    view_metrics,paired=metrics_by_view(df)
    df.to_csv(os.path.join(save_dir,f"{split}_clip_predictions.csv"),index=False)
    if paired is not None:
        paired.to_csv(os.path.join(save_dir,f"{split}_paired_predictions.csv"),index=False)
        bin_metrics(paired).to_csv(os.path.join(save_dir,f"{split}_paired_bin_metrics.csv"),index=False)
    else:
        bin_metrics(df).to_csv(os.path.join(save_dir,f"{split}_bin_metrics.csv"),index=False)
    pd.DataFrame([{"view":k,**v} for k,v in view_metrics.items()]).to_csv(
        os.path.join(save_dir,f"{split}_metrics_by_view.csv"),index=False)
    return view_metrics,paired


def select_score(m,bm,args):
    minimal=bm[bm["bin"]=="minimal_0_13"]; mild=bm[bm["bin"]=="mild_14_19"]; moderate=bm[bm["bin"]=="moderate_20_28"]
    mo=float(max(0,minimal["bias"].iloc[0])) if len(minimal) else 0
    mb=float(abs(mild["bias"].iloc[0])) if len(mild) else 0
    mu=float(max(0,-moderate["bias"].iloc[0])) if len(moderate) else 0
    return float(args.select_rmse_w*m["rmse"]-args.select_ccc_w*m["ccc"]-args.select_corr_w*m["corr"]+args.select_minimal_over_w*mo+args.select_mild_bias_w*mb+args.select_moderate_under_w*mu)


def plot_history(save_dir):
    hp=os.path.join(save_dir,"history.csv")
    if not os.path.exists(hp): return
    h=pd.read_csv(hp); plt.figure(figsize=(12,5))
    for c in ["train_loss","dev_rmse","dev_ccc","dev_corr","dev_score","dev_pred_std","train_residual_mean","train_residual_std","train_residual_gate","train_abs_match","res_loss","gate_loss","dir_loss","mod_under","severe_under","pair_loss","proto_kl","proto_score","attn_uniform"]:
        if c in h.columns: plt.plot(h["epoch"],h[c],label=c)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(save_dir,"history_metrics.png"),dpi=200); plt.close()


def plot_pred(df,save_dir,name):
    plt.figure(figsize=(6,6)); plt.scatter(df["gt"],df["anchor_pred"],alpha=.4,label="anchor"); plt.scatter(df["gt"],df["pred"],alpha=.8,label="anchor+residual")
    mn=min(df["gt"].min(),df["pred"].min(),df["anchor_pred"].min()); mx=max(df["gt"].max(),df["pred"].max(),df["anchor_pred"].max())
    plt.plot([mn,mx],[mn,mx],linestyle="--"); plt.xlabel("GT"); plt.ylabel("Prediction"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(save_dir,f"{name}_pred_vs_gt.png"),dpi=200); plt.close()


# -----------------------------
# Main
# -----------------------------

def main(args):
    set_seed(args.seed); os.makedirs(args.save_dir,exist_ok=True)
    print(f"[Device] {DEVICE}")
    print(f"[Model] AVEC2014 AEE V3.1 | task_mode={args.task_mode} | anchor_mode={args.anchor_mode}")
    print(f"[Window] seq_len={args.seq_len}, stride={args.stride}, overlap={args.seq_len-args.stride}")
    if args.diagnostic_train_test:
        print("[WARNING] TEST-ORACLE DIAGNOSTIC: test selects the epoch; never report this as the formal result.")
    au_cols=[c.strip() for c in args.au_columns.split(",") if c.strip()]
    tr_meta=read_txt(args.train_txt); te_meta=read_txt(args.test_txt)
    dev_meta=te_meta.copy() if args.diagnostic_train_test else read_txt(args.dev_txt)
    if args.task_mode!="shared":
        tr_meta=tr_meta[tr_meta["task"]==args.task_mode].reset_index(drop=True)
        dev_meta=dev_meta[dev_meta["task"]==args.task_mode].reset_index(drop=True)
        te_meta=te_meta[te_meta["task"]==args.task_mode].reset_index(drop=True)
        if args.selection_view!=args.task_mode:
            raise ValueError(f"task_mode={args.task_mode} requires --selection_view {args.task_mode}")
    elif args.selection_view not in ("paired_mean","clips_pooled","Freeform","Northwind"):
        raise ValueError(f"Invalid shared selection view: {args.selection_view}")
    split_pairs=[set(x["pair_id"].astype(str)) for x in (tr_meta,dev_meta,te_meta)]
    if split_pairs[0]&split_pairs[1] or split_pairs[0]&split_pairs[2]:
        raise ValueError("pair_id overlap between training and evaluation data")
    if not args.diagnostic_train_test and split_pairs[1]&split_pairs[2]:
        raise ValueError("pair_id overlap between dev and test")
    checked=(("train",tr_meta),("test",te_meta)) if args.diagnostic_train_test else (("train",tr_meta),("dev",dev_meta),("test",te_meta))
    for name,meta in checked:
        counts=meta.groupby("pair_id").size()
        required_count=2 if args.task_mode=="shared" else 1
        if not (counts==required_count).all():
            raise ValueError(f"{name} must have {required_count} clip(s) per pair: {counts[counts!=required_count].to_dict()}")
        task_sets=meta.groupby("pair_id")["task"].apply(set)
        required_tasks={"Freeform","Northwind"} if args.task_mode=="shared" else {args.task_mode}
        bad=task_sets[task_sets.apply(lambda x:x!=required_tasks)]
        if len(bad): raise ValueError(f"{name} has invalid task pairs: {bad.to_dict()}")

    def load_split(meta,split):
        arrays=[]; paths=[]
        for _,r in tqdm(meta.iterrows(),total=len(meta),desc=f"load {split}",ncols=100):
            p=find_au_file(r["raw_path"],args.data_root); paths.append(p)
            arrays.append(load_au(p,au_cols,not args.no_success_filter,args.allow_missing_au,
                                  args.confidence_threshold,args.resample_hz,args.max_interp_gap))
        return arrays,paths

    tr_arr,tr_paths=load_split(tr_meta,"train"); mean,std=fit_global_stats(tr_arr)
    if args.diagnostic_train_test:
        te_arr,te_paths=load_split(te_meta,"test-oracle"); dev_arr,dev_paths=te_arr,te_paths
    else:
        dev_arr,dev_paths=load_split(dev_meta,"dev"); te_arr,te_paths=load_split(te_meta,"test")

    Xtr_raw=np.stack([make_descriptor(a,mean,std,t) for a,t in zip(tr_arr,tr_meta["task"])]); Xdev_raw=np.stack([make_descriptor(a,mean,std,t) for a,t in zip(dev_arr,dev_meta["task"])]); Xte_raw=np.stack([make_descriptor(a,mean,std,t) for a,t in zip(te_arr,te_meta["task"])])
    (Xs,dmu,dsig)=standardize(Xtr_raw,Xdev_raw,Xte_raw); Xtr,Xdev,Xte=Xs
    ytr=tr_meta["score"].values.astype(np.float32); ydev=dev_meta["score"].values.astype(np.float32); yte=te_meta["score"].values.astype(np.float32)
    anchor=fit_anchor_mode(Xtr,ytr,Xdev,ydev,Xte,yte,tr_meta,dev_meta,te_meta,args)
    with open(os.path.join(args.save_dir,"anchor_summary.json"),"w",encoding="utf-8") as f: json.dump(to_jsonable(anchor["summary"]|{"cv_tables":anchor["cv_tables"]}),f,indent=2,ensure_ascii=False)
    print("[Anchor]"); print(json.dumps(to_jsonable(anchor["summary"]),indent=2,ensure_ascii=False))

    tr_v=build_videos(tr_meta,tr_arr,tr_paths,anchor["train_oof"],mean,std,dmu,dsig,args,"train")
    selection_name="test_oracle" if args.diagnostic_train_test else "dev"
    dev_v=build_videos(dev_meta,dev_arr,dev_paths,anchor["dev"],mean,std,dmu,dsig,args,selection_name)
    te_v=dev_v if args.diagnostic_train_test else build_videos(te_meta,te_arr,te_paths,anchor["test"],mean,std,dmu,dsig,args,"test")

    ds=ExprDataset(tr_v,train=True,args=args)
    sampler=(PairBalancedSampler(tr_v,args.batch_size,args.steps_per_epoch,args.seed)
             if args.task_mode=="shared" else BalancedSampler(tr_v,args.batch_size,args.steps_per_epoch,args.seed))
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=args.num_workers,pin_memory=True)
    model=AEE2014(tr_v[0].segments.shape[-1],tr_v[0].desc.shape[0],args.hidden_dim,args.n_proto,args.expr_score_cap,args.residual_max,args.dropout,args.proto_tau,args.attn_temp,args.score_max,args.minimal_cut,args.match_center,args.match_scale,args.frame_match_weight,args.segment_match_weight).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.min_lr)

    best_score=1e9; best_epoch=-1; patience=0; hist=[]; grad_rows=[]
    best_path=os.path.join(args.save_dir,"best_aee_2014_test_oracle.pth" if args.diagnostic_train_test else "best_aee_2014.pth")
    for ep in range(1,args.epochs+1):
        model.train(); rows=[]; pbar=tqdm(loader,desc=f"AEE2014 Ep{ep}",ncols=145)
        for bi,b in enumerate(pbar,1):
            x=b["x"].to(DEVICE); mask=b["mask"].to(DEVICE); bag=b["bag_mask"].to(DEVICE); desc=b["desc"].to(DEVICE); anch=b["anchor"].to(DEVICE); y=b["y"].to(DEVICE)
            opt.zero_grad(set_to_none=True); out=model(x,mask,bag,desc,anch); loss,st=loss_fn(out,y,anch,model,args,ep,b["pair_id"]); loss.backward()
            gn=0.0
            for p in model.parameters():
                if p.grad is not None: gn+=float((p.grad.detach()**2).sum().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step()
            st["epoch"]=ep; st["batch"]=bi; rows.append(st); grad_rows.append({"epoch":ep,"batch":bi,"grad_total":math.sqrt(gn)})
            pbar.set_postfix(loss=f"{st['loss']:.3f}",ps=f"{st['pred_std']:.2f}",rs=f"{st['residual_std']:.2f}",r=f"{st['ramp']:.2f}")
        sch.step(); ldf=pd.DataFrame(rows)
        dev_df,_,_=evaluate(model,dev_v,args,"dev")
        dev_views,dev_pair=metrics_by_view(dev_df); dev_m=dev_views[args.selection_view]; dev_bm=bin_metrics(dev_pair if args.selection_view=="paired_mean" else dev_df[dev_df["task"]==args.selection_view] if args.selection_view in ("Freeform","Northwind") else dev_df)
        score=select_score(dev_m,dev_bm,args)
        rec={"epoch":ep,"lr":float(opt.param_groups[0]["lr"]),"train_loss":float(ldf.loss.mean()),"train_pred_std":float(ldf.pred_std.mean()),"train_residual_mean":float(ldf.residual_mean.mean()),"train_residual_std":float(ldf.residual_std.mean()),"train_residual_gate":float(ldf.residual_gate.mean()),"train_abs_match":float(ldf.abs_match.mean()),"res_loss":float(ldf.res_loss.mean()),"gate_loss":float(ldf.gate_loss.mean()),"dir_loss":float(ldf.dir_loss.mean()),"mod_under":float(ldf.mod_under.mean()),"severe_under":float(ldf.severe_under.mean()),"pair_loss":float(ldf.pair_loss.mean()),"proto_kl":float(ldf.proto_kl.mean()),"proto_score":float(ldf.proto_score.mean()),"attn_uniform":float(ldf.attn_uniform.mean()),"dev_score":score,"dev_rmse":dev_m["rmse"],"dev_mae":dev_m["mae"],"dev_ccc":dev_m["ccc"],"dev_corr":dev_m["corr"],"dev_pred_mean":dev_m["pred_mean"],"dev_pred_std":dev_m["pred_std"]}
        hist.append(rec); pd.DataFrame(hist).to_csv(os.path.join(args.save_dir,"history.csv"),index=False); pd.DataFrame(grad_rows).to_csv(os.path.join(args.save_dir,"gradient_flow.csv"),index=False)
        selection_label="TEST-ORACLE" if args.diagnostic_train_test else "DEV"
        print(f"Ep{ep} | score={score:.4f} | {selection_label} RMSE={dev_m['rmse']:.4f} MAE={dev_m['mae']:.4f} CCC={dev_m['ccc']:.4f} corr={dev_m['corr']:.4f} std={dev_m['pred_std']:.3f}")
        if score<best_score:
            best_score=score; best_epoch=ep; patience=0
            protocol="avec2014_train_test_oracle_diagnostic" if args.diagnostic_train_test else "avec2014_train_dev_select_test_report"
            torch.save({"model_state":model.state_dict(),"args":vars(args),"au_columns":au_cols,
                        "input_columns":au_cols+[QUALITY_COLUMN],"raw_mean":mean,"raw_std":std,
                        "desc_mu":dmu,"desc_sig":dsig,"anchor_models":anchor["models"],
                        "best_epoch":best_epoch,
                        "best_selection_score":best_score,"proto_scores":model.proto_scores.detach().cpu().numpy(),
                        "protocol":protocol,"model_name":"AEE2014V3_1NullRejectEvidenceGatedLexicon"},best_path)
            prefix="best_test_oracle" if args.diagnostic_train_test else "best_dev"
            dev_df.to_csv(os.path.join(args.save_dir,f"{prefix}_clip_predictions.csv"),index=False)
            if dev_pair is not None: dev_pair.to_csv(os.path.join(args.save_dir,f"{prefix}_paired_predictions.csv"),index=False)
            dev_bm.to_csv(os.path.join(args.save_dir,f"{prefix}_bin_metrics.csv"),index=False)
        else: patience+=1
        if ep>=args.min_epochs and patience>=args.early_stop:
            print(f"Early stop at epoch {ep}."); break

    ckpt=load_checkpoint(best_path,map_location=DEVICE); model.load_state_dict(ckpt["model_state"],strict=True)
    tr_df,_,_=evaluate(model,tr_v,args,"train")
    if args.diagnostic_train_test:
        te_df,_,_=evaluate(model,te_v,args,"test"); dev_df=te_df
    else:
        dev_df,_,_=evaluate(model,dev_v,args,"dev"); te_df,_,_=evaluate(model,te_v,args,"test")
    tr_views,tr_pair=save_eval_bundle(tr_df,args.save_dir,"train")
    te_views,te_pair=save_eval_bundle(te_df,args.save_dir,"test")
    if args.diagnostic_train_test:
        dev_views,dev_pair=te_views,te_pair
    else:
        dev_views,dev_pair=save_eval_bundle(dev_df,args.save_dir,"dev")
    dev_m=dev_views[args.selection_view]
    dev_selected=dev_pair if args.selection_view=="paired_mean" else dev_df if args.selection_view=="clips_pooled" else dev_df[dev_df["task"]==args.selection_view]
    te_selected=te_pair if args.selection_view=="paired_mean" else te_df if args.selection_view=="clips_pooled" else te_df[te_df["task"]==args.selection_view]
    tr_selected=tr_pair if args.selection_view=="paired_mean" else tr_df if args.selection_view=="clips_pooled" else tr_df[tr_df["task"]==args.selection_view]
    dev_bm=bin_metrics(dev_selected); te_bm=bin_metrics(te_selected)
    proto_cols=[c for c in te_df.columns if c.startswith("proto") and c.endswith("_mean")]; pu=[]
    for c in proto_cols:
        k=int(c.replace("proto","").replace("_mean","")); row={"proto":k,"score_anchor":float(model.proto_scores[k].detach().cpu())}
        for b in SEVERITY_NAMES:
            g=te_df[te_df["bin"]==b]; row[b]=float(g[c].mean()) if len(g) else 0.0
        row["corr_gt"]=safe_corr(te_df[c],te_df["gt"]); pu.append(row)
    pu=pd.DataFrame(pu); pu.to_csv(os.path.join(args.save_dir,"prototype_usage_test.csv"),index=False)

    common={"score_definition":"BDI-II integer range 0..63","task_mode":args.task_mode,"selection_view":args.selection_view,"best_epoch":int(best_epoch),"best_selection_score":float(best_score),"anchor":anchor["summary"],"train":{"views":tr_views,"selection_bins":bin_metrics(tr_selected).to_dict(orient="records")},"test":{"views":te_views,"selection_bins":te_bm.to_dict(orient="records")},"prototype_usage_test":pu.to_dict(orient="records"),"recommend_checkpoint":best_path,"seed":args.seed,"model_config":{"n_proto":args.n_proto,"score_max":args.score_max,"expr_score_cap":args.expr_score_cap,"residual_max":args.residual_max,"anchor_mode":args.anchor_mode,"resample_hz":args.resample_hz,"confidence_threshold":args.confidence_threshold,"residual_target_scale":args.residual_target_scale,"residual_minimal_scale":args.residual_minimal_scale,"residual_moderate_scale":args.residual_moderate_scale,"seq_len":args.seq_len,"stride":args.stride,"overlap_frames":args.seq_len-args.stride,"input_mode":args.input_mode}}
    if args.diagnostic_train_test:
        summary={"protocol":"avec2014_train_test_oracle_diagnostic","diagnostic_only":True,"test_selects_checkpoint":True,**common,"notes":{"purpose":"estimate architecture ceiling before formal tuning; never use as paper result","training_data":"train only","pairing":"paired mean" if args.task_mode=="shared" else f"{args.task_mode}-only"}}
        summary_name="train_test_oracle_summary.json"
    else:
        summary={"protocol":"avec2014_train_dev_select_test_report","diagnostic_only":False,**common,"dev":{"views":dev_views,"selection_bins":dev_bm.to_dict(orient="records")},"notes":{"core":"same 2017 train-only shallow AU anchor + bounded temporal-MIL residual + compressed prototype lexicon","pairing":"paired mean" if args.task_mode=="shared" else f"{args.task_mode}-only","leakage_control":"anchor OOF folds are grouped by pair_id; test never selects checkpoint"}}
        summary_name="train_summary.json"
    with open(os.path.join(args.save_dir,summary_name),"w",encoding="utf-8") as f: json.dump(to_jsonable(summary),f,indent=2,ensure_ascii=False)
    plot_history(args.save_dir)
    if not args.diagnostic_train_test: plot_pred(dev_df,args.save_dir,"dev")
    plot_pred(te_df,args.save_dir,"test")
    print(json.dumps(to_jsonable(summary),indent=2,ensure_ascii=False)); print(f"\nFinal TEST {args.selection_view} bin metrics:"); print(te_bm.to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train_txt",default="manifests/train_2014.txt"); p.add_argument("--dev_txt",default="manifests/dev_2014.txt"); p.add_argument("--test_txt",default="manifests/test_2014.txt"); p.add_argument("--data_root",default=""); p.add_argument("--save_dir",required=True)
    p.add_argument("--score_max",type=float,default=63.0); p.add_argument("--minimal_cut",type=float,default=14.0); p.add_argument("--mild_cut",type=float,default=20.0); p.add_argument("--moderate_cut",type=float,default=29.0)
    p.add_argument("--task_mode",default="shared",choices=["shared","Freeform","Northwind"])
    p.add_argument("--selection_view",default="paired_mean",choices=["paired_mean","clips_pooled","Freeform","Northwind"])
    p.add_argument("--diagnostic_train_test",action="store_true",help="Select best epoch directly on test; diagnostic only, never a paper result")
    p.add_argument("--au_columns",default=",".join(DEFAULT_AU_COLUMNS)); p.add_argument("--no_success_filter",action="store_true"); p.add_argument("--allow_missing_au",action="store_true")
    p.add_argument("--confidence_threshold",type=float,default=0.55); p.add_argument("--resample_hz",type=float,default=30.0); p.add_argument("--max_interp_gap",type=float,default=0.25)
    p.add_argument("--input_mode",default="raw_rel_delta",choices=["raw","raw_rel","raw_rel_delta"]); p.add_argument("--au_clip",type=float,default=5.0); p.add_argument("--robust_eps",type=float,default=0.15); p.add_argument("--rel_clip",type=float,default=5.0); p.add_argument("--delta_clip",type=float,default=5.0)
    p.add_argument("--anchor_mode",default="task_specific",choices=["shared","task_specific"]); p.add_argument("--anchor_folds",type=int,default=5); p.add_argument("--anchor_alphas",default="3,10,30,100,300,1000,3000")
    p.add_argument("--seq_len",type=int,default=300); p.add_argument("--stride",type=int,default=150); p.add_argument("--max_segments",type=int,default=128); p.add_argument("--max_train_segments_per_video",type=int,default=48)
    p.add_argument("--hidden_dim",type=int,default=64); p.add_argument("--n_proto",type=int,default=10); p.add_argument("--expr_score_cap",type=float,default=1.0); p.add_argument("--residual_max",type=float,default=20.0); p.add_argument("--dropout",type=float,default=0.08); p.add_argument("--proto_tau",type=float,default=0.16); p.add_argument("--attn_temp",type=float,default=0.45)
    p.add_argument("--match_center",type=float,default=0.15); p.add_argument("--match_scale",type=float,default=0.10); p.add_argument("--frame_match_weight",type=float,default=0.35); p.add_argument("--segment_match_weight",type=float,default=0.25)
    p.add_argument("--epochs",type=int,default=120); p.add_argument("--min_epochs",type=int,default=20); p.add_argument("--early_stop",type=int,default=32); p.add_argument("--batch_size",type=int,default=10); p.add_argument("--steps_per_epoch",type=int,default=80); p.add_argument("--num_workers",type=int,default=0); p.add_argument("--lr",type=float,default=7e-5); p.add_argument("--min_lr",type=float,default=5e-6); p.add_argument("--weight_decay",type=float,default=9e-4); p.add_argument("--grad_clip",type=float,default=5.0); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--segment_keep_ratio",type=float,default=0.92); p.add_argument("--noise_std",type=float,default=0.005); p.add_argument("--feature_dropout",type=float,default=0.005); p.add_argument("--time_mask_prob",type=float,default=0.04); p.add_argument("--time_mask_ratio",type=float,default=0.04)
    p.add_argument("--huber_beta",type=float,default=2.5); p.add_argument("--loss_minimal_weight",type=float,default=1.0); p.add_argument("--loss_mild_weight",type=float,default=1.0); p.add_argument("--loss_moderate_weight",type=float,default=1.05); p.add_argument("--loss_severe_weight",type=float,default=1.15)
    p.add_argument("--w_huber",type=float,default=0.70); p.add_argument("--w_ccc",type=float,default=0.08); p.add_argument("--w_corr",type=float,default=0.03); p.add_argument("--w_rank",type=float,default=0.015); p.add_argument("--w_residual",type=float,default=0.35); p.add_argument("--w_res_l2",type=float,default=0.006); p.add_argument("--w_res_direction",type=float,default=0.08); p.add_argument("--w_gate",type=float,default=0.12); p.add_argument("--w_moderate_under",type=float,default=0.020); p.add_argument("--w_severe_under",type=float,default=0.030); p.add_argument("--severe_under_margin",type=float,default=4.0); p.add_argument("--w_pair",type=float,default=0.25); p.add_argument("--w_low_bce",type=float,default=0.04); p.add_argument("--w_low_over",type=float,default=0.035); p.add_argument("--w_aux",type=float,default=0.025); p.add_argument("--w_dist",type=float,default=0.020)
    p.add_argument("--library_warmup_epochs",type=int,default=1); p.add_argument("--ordinal_sigma",type=float,default=0.08); p.add_argument("--proto_mean_kl_weight",type=float,default=0.15); p.add_argument("--proto_attn_kl_weight",type=float,default=0.85); p.add_argument("--w_proto_ordinal",type=float,default=0.08); p.add_argument("--w_proto_score",type=float,default=0.06); p.add_argument("--w_proto_diversity",type=float,default=0.020); p.add_argument("--w_attn_focus",type=float,default=0.005); p.add_argument("--w_attn_uniform",type=float,default=0.05); p.add_argument("--attn_entropy_target",type=float,default=0.92); p.add_argument("--w_proto_entropy",type=float,default=0.006)
    p.add_argument("--rank_margin",type=float,default=1.2); p.add_argument("--rank_min_gap",type=float,default=10.0); p.add_argument("--low_over_margin",type=float,default=2.5); p.add_argument("--residual_beta",type=float,default=2.0); p.add_argument("--residual_target_scale",type=float,default=1.0); p.add_argument("--residual_minimal_scale",type=float,default=1.0); p.add_argument("--residual_mild_scale",type=float,default=1.0); p.add_argument("--residual_moderate_scale",type=float,default=1.0); p.add_argument("--residual_severe_scale",type=float,default=1.0); p.add_argument("--residual_minimal_weight",type=float,default=1.0); p.add_argument("--residual_mild_weight",type=float,default=1.0); p.add_argument("--residual_moderate_weight",type=float,default=1.10); p.add_argument("--residual_severe_weight",type=float,default=1.20); p.add_argument("--residual_magnitude_weight",type=float,default=0.25); p.add_argument("--residual_push_fraction",type=float,default=0.35); p.add_argument("--mild_push_weight",type=float,default=0.35); p.add_argument("--moderate_under_margin",type=float,default=6.0)
    p.add_argument("--select_rmse_w",type=float,default=1.0); p.add_argument("--select_ccc_w",type=float,default=0.90); p.add_argument("--select_corr_w",type=float,default=0.42); p.add_argument("--select_minimal_over_w",type=float,default=0.035); p.add_argument("--select_mild_bias_w",type=float,default=0.020); p.add_argument("--select_moderate_under_w",type=float,default=0.015)
    args=p.parse_args()
    if (args.score_max,args.minimal_cut,args.mild_cut,args.moderate_cut)!=(63.0,14.0,20.0,29.0):
        raise ValueError("This AVEC2014 implementation fixes BDI-II to max=63 and cuts 14/20/29.")
    if abs(args.expr_score_cap-1.0)>1e-8:
        raise ValueError("V3 prototype lexicon is normalized and requires --expr_score_cap 1.0")
    if not 0<=args.confidence_threshold<=1: raise ValueError("confidence_threshold must be in [0,1]")
    if args.resample_hz<=0 or args.max_interp_gap<0: raise ValueError("resample_hz must be >0 and max_interp_gap >=0")
    main(args)
