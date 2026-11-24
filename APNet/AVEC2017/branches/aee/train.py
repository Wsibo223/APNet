#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
branches/aee/train.py

V17: Residual Target Strengthening Expression Branch for AVEC2017.

核心思想：
1. 不再让深层 AU 表情分支从零预测 0~24。
2. 先用 train-only shallow AU descriptor ridge 得到稳定 anchor。
   - train anchor 使用 K-fold OOF，避免 train label 过拟合。
   - dev/test anchor 使用 train-only full ridge 推理。
3. 深层 temporal-MIL + compressed expression lexicon 只学习 bounded residual:
      pred = anchor + residual, residual in [-R, R]
4. 表情库压缩到 0~14，因为表情分支不负责 severe 的完整建模，severe 交给麻木分支。
5. 使用重叠滑窗：默认 seq_len=300, stride=100。

输出：
- anchor_summary.json
- train_summary.json
- final_dev_predictions.csv / final_test_predictions.csv
- final_dev_bin_metrics.csv / final_test_bin_metrics.csv
- prototype_usage_test.csv
"""

import os, re, glob, json, math, argparse, random
import time
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from checkpoint import load_checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_AU_COLUMNS = [
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r", "AU09_r",
    "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r", "AU20_r", "AU25_r",
]


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
    return 0 if y<6 else (1 if y<12 else (2 if y<18 else 3))


def bin_name(y):
    return ["minimal_0_5","mild_6_11","moderate_12_17","severe_18_24"][bin_id(y)]


def bin_metrics(df, pred_col="pred"):
    rows=[]; total=float(((df[pred_col]-df["gt"])**2).sum())+1e-8
    for b in ["minimal_0_5","mild_6_11","moderate_12_17","severe_18_24"]:
        g=df[df["bin"]==b]
        if len(g)==0: continue
        m=metrics_np(g["gt"], g[pred_col])
        row={"bin":b, "count":int(len(g)), **m, "sse_contribution":float(((g[pred_col]-g["gt"])**2).sum()/total)}
        for c in ["anchor_pred","anchor_error","residual","residual_target","residual_target_scaled","proto_score","seg_attn","top_score","low_mass","mid_mass","high_mass","p_low","attn_peak","attn_entropy","proto_entropy","proto_peak"]:
            if c in g.columns: row[c]=float(g[c].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Loading AU
# -----------------------------

def video_id_from_path(path):
    b=os.path.basename(path); b=re.sub(r"\.csv$","",b)
    for suf in ["_AUs","_aus","_AU","_pose","_gaze","_features","_openface"]:
        if b.endswith(suf): b=b[:-len(suf)]
    return b


def read_txt(txt):
    rows=[]
    with open(txt,"r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            p,y=line.strip().split()[:2]; y=float(y)
            rows.append({"raw_path":p, "score":y, "video_id":video_id_from_path(p), "bin":bin_name(y)})
    return pd.DataFrame(rows)


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


def load_au(path, au_cols, success_filter=True):
    df=pd.read_csv(path); df.columns=[c.strip() for c in df.columns]
    if success_filter and "success" in df.columns:
        try: df=df[df["success"].astype(float)>=0.5].copy()
        except Exception: pass
    if len(df)==0: raise ValueError(f"No valid rows in {path}")
    xs=[]
    for c in au_cols:
        if c in df.columns: v=pd.to_numeric(df[c], errors="coerce").fillna(0).values.astype(np.float32)
        else: v=np.zeros(len(df),dtype=np.float32)
        xs.append(v)
    return np.stack(xs,axis=1).astype(np.float32)


def fit_global_stats(arrays):
    x=np.concatenate(arrays,axis=0)
    return x.mean(0,keepdims=True).astype(np.float32), np.maximum(x.std(0,keepdims=True),1e-6).astype(np.float32)


# -----------------------------
# Descriptor anchor and sequence features
# -----------------------------

def make_descriptor(x_raw, mean, std):
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
    return np.clip(Xb@beta,0,24)


def fit_anchor(Xtr,ytr,Xdev,ydev,Xte,yte,args):
    alphas=[float(a) for a in args.anchor_alphas.split(",") if a.strip()]
    rng=np.random.default_rng(args.seed); idx=np.arange(len(ytr)); rng.shuffle(idx)
    folds=np.array_split(idx,min(args.anchor_folds,len(ytr)))
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
            "test":metrics_np(yte,te),
        }
    }


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
    split: str; video_id: str; raw_path: str; au_path: str; score: float; bin: str; anchor_pred: float
    segments: np.ndarray; masks: np.ndarray; desc: np.ndarray


def build_videos(meta, arrays, paths, anchors, mean, std, dmu, dsig, args, split):
    videos=[]
    for i,(_,r) in enumerate(tqdm(list(meta.iterrows()),desc=f"segment {split}",ncols=100)):
        feat=make_frame_features(arrays[i],mean,std,args)
        segs,masks=make_segments(feat,args.seq_len,args.stride,args.max_segments)
        desc=((make_descriptor(arrays[i],mean,std).reshape(-1)-dmu.reshape(-1))/dsig.reshape(-1)).astype(np.float32)
        videos.append(VideoItem(split,str(r.video_id),str(r.raw_path),str(paths[i]),float(r.score),str(r.bin),float(anchors[i]),segs,masks,desc))
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
            "y":torch.tensor(v.score,dtype=torch.float32), "video_id":v.video_id, "bin":v.bin
        }


def collate(batch):
    B=len(batch); S=max(b["x"].shape[0] for b in batch); T=batch[0]["x"].shape[1]; C=batch[0]["x"].shape[2]; D=batch[0]["desc"].reshape(-1).shape[0]
    x=torch.zeros(B,S,T,C); mask=torch.zeros(B,S,T,dtype=torch.bool); bag=torch.zeros(B,S,dtype=torch.bool)
    desc=torch.zeros(B,D); y=torch.zeros(B); anchor=torch.zeros(B); vids=[]; bins=[]
    for i,b in enumerate(batch):
        s=b["x"].shape[0]; x[i,:s]=b["x"]; mask[i,:s]=b["mask"]; bag[i,:s]=True
        desc[i]=b["desc"].reshape(-1); y[i]=b["y"]; anchor[i]=b["anchor"]; vids.append(b["video_id"]); bins.append(b["bin"])
    return {"x":x,"mask":mask,"bag_mask":bag,"desc":desc,"y":y,"anchor":anchor,"video_id":vids,"bin":bins}


class BalancedSampler(Sampler):
    def __init__(self,videos,batch_size,steps,seed):
        self.bs=batch_size; self.steps=steps; self.seed=seed; self.epoch=0
        self.by={0:[],1:[],2:[],3:[]}
        for i,v in enumerate(videos): self.by[bin_id(v.score)].append(i)
        print("[BatchSampler] counts:",{k:len(v) for k,v in self.by.items()})
    def __len__(self): return self.steps
    def __iter__(self):
        rng=np.random.default_rng(self.seed+self.epoch); self.epoch+=1
        quota=[7,5,3,1] if self.bs>=16 else [max(1,self.bs//2),max(1,self.bs//3),1,1]
        while sum(quota)<self.bs: quota[0]+=1
        while sum(quota)>self.bs: quota[0]-=1
        for _ in range(self.steps):
            batch=[]
            for k,q in enumerate(quota):
                pool=self.by[k]
                if pool: batch += rng.choice(pool,size=q,replace=len(pool)<q).tolist()
            rng.shuffle(batch); yield batch


# -----------------------------
# Model
# -----------------------------

class SegmentEncoder(nn.Module):
    def __init__(self,input_dim,hid=64,K=10,cap=14.0,drop=0.08,tau=0.16):
        super().__init__()
        self.proj=nn.Sequential(nn.Linear(input_dim,hid),nn.LayerNorm(hid),nn.GELU())
        self.conv=nn.Sequential(nn.Conv1d(hid,hid,3,padding=1),nn.BatchNorm1d(hid),nn.GELU(),nn.Dropout(drop))
        self.gru=nn.GRU(hid,hid//2,batch_first=True,bidirectional=True)
        p=torch.empty(K,hid); nn.init.orthogonal_(p); self.proto=nn.Parameter(p)
        self.register_buffer("scores",torch.linspace(0,cap,K)); self.tau=tau
        self.attn=nn.Sequential(nn.Linear(hid+K+2,hid//2),nn.Tanh(),nn.Linear(hid//2,1))
        self.offset=nn.Sequential(nn.Linear(hid+K+4,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1),nn.Tanh())
    def forward(self,x,mask):
        h=self.proj(x); h=self.conv(h.transpose(1,2)).transpose(1,2); h,_=self.gru(h)
        z=F.normalize(h,dim=-1); p=F.normalize(self.proto,dim=1)
        w=torch.softmax(torch.matmul(z,p.T)/self.tau,dim=-1)
        proto_frame=(w*self.scores.view(1,1,-1)).sum(-1)
        peak=w.max(-1).values
        ent=-(w.clamp_min(1e-8)*torch.log(w.clamp_min(1e-8))).sum(-1)/math.log(w.shape[-1])
        conf=peak*(1-0.5*ent).clamp_min(0.05)
        logits=self.attn(torch.cat([h,w,(proto_frame/self.scores.max()).unsqueeze(-1),conf.unsqueeze(-1)],-1)).squeeze(-1)
        logits=logits.masked_fill(~mask.bool(),-1e9)
        a=torch.softmax(logits,dim=1)
        emb=(h*a.unsqueeze(-1)).sum(1); hist=(w*a.unsqueeze(-1)).sum(1)
        valid=mask.float(); den=valid.sum(1).clamp_min(1)
        pscore=(hist*self.scores.view(1,-1)).sum(1)
        pc=(conf*valid).sum(1)/den; pe=(ent*valid).sum(1)/den; pp=(peak*valid).sum(1)/den
        off=self.offset(torch.cat([emb,hist,(pscore/self.scores.max()).unsqueeze(-1),pc.unsqueeze(-1),pe.unsqueeze(-1),pp.unsqueeze(-1)],1)).view(-1)*2.0
        seg=torch.clamp(pscore+off,0,float(self.scores.max()))
        return emb,hist,seg,pscore,pc,pe,pp


class V17(nn.Module):
    def __init__(self,input_dim,desc_dim,hid=64,K=10,cap=14.0,residual_max=3.5,drop=0.08,tau=0.16,attn_temp=0.85):
        super().__init__()
        self.enc=SegmentEncoder(input_dim,hid,K,cap,drop,tau); self.residual_max=residual_max; self.attn_temp=attn_temp
        self.register_buffer("proto_scores",torch.linspace(0,cap,K))
        self.desc=nn.Sequential(nn.Linear(desc_dim,hid),nn.LayerNorm(hid),nn.GELU(),nn.Dropout(drop))
        self.seg_attn=nn.Sequential(nn.Linear(hid+K+5,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1))
        vdim=hid*2+K*3+hid+19
        self.res_head=nn.Sequential(nn.Linear(vdim,hid),nn.LayerNorm(hid),nn.GELU(),nn.Dropout(drop),nn.Linear(hid,hid//2),nn.GELU(),nn.Linear(hid//2,1),nn.Tanh())
        self.low_head=nn.Sequential(nn.Linear(vdim,hid//2),nn.GELU(),nn.Dropout(drop),nn.Linear(hid//2,1))
        self.mil_res=nn.Sequential(nn.Linear(9,18),nn.GELU(),nn.Linear(18,1),nn.Tanh())
    def forward(self,x,mask,bag,desc,anchor):
        B,S,T,D=x.shape; cap=self.proto_scores.max().clamp_min(1)
        emb,hist,seg,pscore,pc,pe,pp=self.enc(x.view(B*S,T,D),mask.view(B*S,T))
        H=emb.shape[-1]; K=hist.shape[-1]
        emb=emb.view(B,S,H); hist=hist.view(B,S,K); seg=seg.view(B,S); pscore=pscore.view(B,S); pc=pc.view(B,S); pe=pe.view(B,S); pp=pp.view(B,S)
        valid=bag.float(); den=valid.sum(1).clamp_min(1)
        logits=self.seg_attn(torch.cat([emb,hist,(seg/cap).unsqueeze(-1),(pscore/cap).unsqueeze(-1),pc.unsqueeze(-1),pe.unsqueeze(-1),pp.unsqueeze(-1)],-1)).squeeze(-1)
        logits=logits+0.15*(seg/cap)+0.10*torch.log(pc.clamp_min(1e-4))
        logits=logits.masked_fill(~bag.bool(),-1e9)
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
        low_mask=(self.proto_scores<=4.5).float(); mid_mask=((self.proto_scores>4.5)&(self.proto_scores<9.5)).float(); high_mask=(self.proto_scores>=9.5).float()
        low_mass=(mean_hist*low_mask.view(1,-1)).sum(1); mid_mass=(mean_hist*mid_mask.view(1,-1)).sum(1); high_mass=(mean_hist*high_mask.view(1,-1)).sum(1)
        proto_score=(mean_hist*self.proto_scores.view(1,-1)).sum(1); proto_score_attn=(att_hist*self.proto_scores.view(1,-1)).sum(1)
        d=self.desc(desc); an=(anchor/24).clamp(0,1)
        mil=torch.stack([an,proto_score/cap,proto_score_attn/cap,seg_mean/cap,seg_attn/cap,top_score/cap,low_mass,mid_mass,high_mass],1)
        stats=torch.stack([an,proto_score/cap,proto_score_attn/cap,seg_mean/cap,seg_attn/cap,top_score/cap,seg_low/cap,seg_std/cap,low_mass,mid_mass,high_mass,proto_entropy,proto_peak,attn_peak,attn_entropy,den/float(S),torch.relu(anchor-6)/18,torch.relu(6-anchor)/6, (att_hist-mean_hist).abs().sum(1)/2],1)
        vf=torch.cat([att_emb,mean_emb,att_hist,mean_hist,top_hist,d,stats],1)
        r1=self.residual_max*self.res_head(vf).view(-1); r2=self.residual_max*self.mil_res(mil).view(-1)
        residual=0.75*r1+0.25*r2
        pred=torch.clamp(anchor+residual,0,24); p_low=torch.sigmoid(self.low_head(vf).view(-1))
        return {"pred":pred,"residual":residual,"residual_deep":r1,"residual_mil":r2,"p_low":p_low,"seg_mean":seg_mean,"seg_attn":seg_attn,"top_score":top_score,"seg_low":seg_low,"seg_std":seg_std,"attn":a,"attn_hist":att_hist,"mean_hist":mean_hist,"proto_entropy":proto_entropy,"proto_peak":proto_peak,"attn_peak":attn_peak,"attn_entropy":attn_entropy,"low_mass":low_mass,"mid_mass":mid_mass,"high_mass":high_mass,"proto_score":proto_score,"proto_score_attn":proto_score_attn}


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
    if y < 6:
        scale = args.residual_minimal_scale
    elif y < 12:
        scale = args.residual_mild_scale
    elif y < 18:
        scale = args.residual_moderate_scale
    else:
        scale = args.residual_severe_scale
    return float(np.clip(raw * scale, -args.residual_max, args.residual_max))

def loss_fn(out,y,anchor,model,args,epoch):
    pred=out["pred"]; y=y.float(); anchor=anchor.float()
    cap=float(model.proto_scores.max().detach().cpu()); y_expr=torch.clamp(y,0,cap)

    # V17: residual target is still derived from train-only anchor, but supervision is strengthened
    # in a controlled way. We do not force full gt-anchor correction; instead we use bin-wise
    # target scaling and directional push losses.
    raw_rt=torch.clamp(y-anchor,-args.residual_max,args.residual_max)
    scale=torch.ones_like(y)*args.residual_target_scale
    scale=torch.where(y<6, torch.ones_like(y)*args.residual_minimal_scale, scale)
    scale=torch.where((y>=6)&(y<12), torch.ones_like(y)*args.residual_mild_scale, scale)
    scale=torch.where((y>=12)&(y<18), torch.ones_like(y)*args.residual_moderate_scale, scale)
    scale=torch.where(y>=18, torch.ones_like(y)*args.residual_severe_scale, scale)
    rt=torch.clamp(raw_rt*scale,-args.residual_max,args.residual_max)

    ramp=min(1.0,max(0.0,(epoch-1)/max(1,args.library_warmup_epochs)))
    w=torch.ones_like(y)
    w=torch.where(y<6,w*args.loss_minimal_weight,w); w=torch.where((y>=6)&(y<12),w*args.loss_mild_weight,w)
    w=torch.where((y>=12)&(y<18),w*args.loss_moderate_weight,w); w=torch.where(y>=18,w*args.loss_severe_weight,w)
    w=w/w.mean().clamp_min(1e-6)
    huber=(F.smooth_l1_loss(pred,y,beta=args.huber_beta,reduction="none")*w).mean()
    cl=ccc_loss(pred,y); cr=corr_loss(pred,y); rl=rank_loss(pred,y,args.rank_margin,args.rank_min_gap)

    res_w=torch.ones_like(y)
    res_w=torch.where(y<6,res_w*args.residual_minimal_weight,res_w)
    res_w=torch.where((y>=6)&(y<12),res_w*args.residual_mild_weight,res_w)
    res_w=torch.where((y>=12)&(y<18),res_w*args.residual_moderate_weight,res_w)
    res_w=torch.where(y>=18,res_w*args.residual_severe_weight,res_w)
    res_w=res_w*(1.0+args.residual_magnitude_weight*(raw_rt.abs()/max(args.residual_max,1e-6)))
    res_w=res_w/res_w.mean().clamp_min(1e-6)
    res_loss=(F.smooth_l1_loss(out["residual"],rt,beta=args.residual_beta,reduction="none")*res_w).mean()
    res_l2=(out["residual"]**2).mean()

    # Directional push: make residual actually move in the expected direction.
    minimal=y<6; mild=(y>=6)&(y<12); moderate=(y>=12)&(y<18)
    push=torch.clamp(rt.abs()*args.residual_push_fraction, min=0.0, max=args.residual_max)
    dir_loss=pred.new_tensor(0.)
    if minimal.any():
        dir_loss=dir_loss+torch.relu(out["residual"][minimal]+push[minimal]).pow(2).mean()
    if mild.any():
        dir_loss=dir_loss+args.mild_push_weight*torch.relu(push[mild]-out["residual"][mild]).pow(2).mean()
    if moderate.any():
        dir_loss=dir_loss+torch.relu(push[moderate]-out["residual"][moderate]).pow(2).mean()

    # Directly address the observed moderate underestimation without using a separate boost head.
    mod_under=torch.relu(y[moderate]-pred[moderate]-args.moderate_under_margin).pow(2).mean() if moderate.any() else pred.new_tensor(0.)

    low_bce=F.binary_cross_entropy(out["p_low"],(y<6).float())
    low_over=torch.relu(pred[minimal]-y[minimal]-args.low_over_margin).pow(2).mean() if minimal.any() else pred.new_tensor(0.)
    target=ordinal_target(y_expr,model.proto_scores,args.ordinal_sigma)
    mh=out["mean_hist"].clamp_min(1e-8); ah=out["attn_hist"].clamp_min(1e-8)
    pkl=0.65*(target*(torch.log(target.clamp_min(1e-8))-torch.log(mh))).sum(1).mean()+0.35*(target*(torch.log(target.clamp_min(1e-8))-torch.log(ah))).sum(1).mean()
    pscore=F.smooth_l1_loss(out["proto_score"],y_expr,beta=1.5)+0.5*F.smooth_l1_loss(out["proto_score_attn"],y_expr,beta=1.5)
    aux=0.5*F.smooth_l1_loss(out["seg_attn"],y_expr,beta=2.0)+0.5*F.smooth_l1_loss(out["top_score"],y_expr,beta=2.0)
    div=proto_div_loss(model.enc.proto); attn=out["attn_entropy"].mean(); pent=out["proto_entropy"].mean()
    dist=pred.new_tensor(0.)
    if pred.numel()>1:
        dist=((pred.mean()-y.mean())/8).pow(2)+0.25*((pred.std(unbiased=False)-y.std(unbiased=False))/8).pow(2)
    loss=args.w_huber*huber+args.w_ccc*cl+args.w_corr*cr+args.w_rank*rl+args.w_residual*res_loss+args.w_res_l2*res_l2+args.w_res_direction*dir_loss+args.w_moderate_under*mod_under+args.w_low_bce*low_bce+args.w_low_over*low_over+args.w_aux*aux+ramp*(args.w_proto_ordinal*pkl+args.w_proto_score*pscore+args.w_proto_diversity*div+args.w_attn_focus*attn+args.w_proto_entropy*pent)+args.w_dist*dist
    return loss, {"loss":float(loss.detach().cpu()),"pred_std":float(pred.detach().cpu().std(unbiased=False)) if pred.numel()>1 else 0.0,"residual_std":float(out["residual"].detach().cpu().std(unbiased=False)) if pred.numel()>1 else 0.0,"residual_mean":float(out["residual"].detach().cpu().mean()),"res_loss":float(res_loss.detach().cpu()),"dir_loss":float(dir_loss.detach().cpu()),"mod_under":float(mod_under.detach().cpu()),"proto_kl":float(pkl.detach().cpu()),"proto_score":float(pscore.detach().cpu()),"ramp":float(ramp)}


@torch.no_grad()
def evaluate(model,videos,args,split):
    model.eval(); rows=[]
    for v in tqdm(videos,desc=f"eval {split}",ncols=100,leave=False):
        b=collate([ExprDataset([v],train=False,args=args)[0]])
        out=model(b["x"].to(DEVICE),b["mask"].to(DEVICE),b["bag_mask"].to(DEVICE),b["desc"].to(DEVICE),b["anchor"].to(DEVICE))
        y=float(v.score); a=float(v.anchor_pred)
        row={"split":split,"video_id":v.video_id,"gt":y,"bin":v.bin,"anchor_pred":a,"anchor_error":a-y,"residual_target":float(np.clip(y-a,-args.residual_max,args.residual_max)),"residual_target_scaled":residual_target_scaled_np(y,a,args),
             "pred":float(out["pred"].cpu().numpy()[0]),"residual":float(out["residual"].cpu().numpy()[0]),"residual_deep":float(out["residual_deep"].cpu().numpy()[0]),"residual_mil":float(out["residual_mil"].cpu().numpy()[0]),
             "p_low":float(out["p_low"].cpu().numpy()[0]),"proto_score":float(out["proto_score"].cpu().numpy()[0]),"proto_score_attn":float(out["proto_score_attn"].cpu().numpy()[0]),
             "seg_mean":float(out["seg_mean"].cpu().numpy()[0]),"seg_attn":float(out["seg_attn"].cpu().numpy()[0]),"top_score":float(out["top_score"].cpu().numpy()[0]),"seg_low":float(out["seg_low"].cpu().numpy()[0]),
             "attn_peak":float(out["attn_peak"].cpu().numpy()[0]),"attn_entropy":float(out["attn_entropy"].cpu().numpy()[0]),"proto_entropy":float(out["proto_entropy"].cpu().numpy()[0]),"proto_peak":float(out["proto_peak"].cpu().numpy()[0]),
             "low_mass":float(out["low_mass"].cpu().numpy()[0]),"mid_mass":float(out["mid_mass"].cpu().numpy()[0]),"high_mass":float(out["high_mass"].cpu().numpy()[0]),"n_segments":len(v.segments)}
        hist=out["mean_hist"].cpu().numpy()[0]
        for k,val in enumerate(hist): row[f"proto{k:02d}_mean"]=float(val)
        rows.append(row)
    df=pd.DataFrame(rows); return df,metrics_np(df["gt"],df["pred"]),bin_metrics(df)


def select_score(m,bm,args):
    minimal=bm[bm["bin"]=="minimal_0_5"]; mild=bm[bm["bin"]=="mild_6_11"]; moderate=bm[bm["bin"]=="moderate_12_17"]
    mo=float(max(0,minimal["bias"].iloc[0])) if len(minimal) else 0
    mb=float(abs(mild["bias"].iloc[0])) if len(mild) else 0
    mu=float(max(0,-moderate["bias"].iloc[0])) if len(moderate) else 0
    return float(args.select_rmse_w*m["rmse"]-args.select_ccc_w*m["ccc"]-args.select_corr_w*m["corr"]+args.select_minimal_over_w*mo+args.select_mild_bias_w*mb+args.select_moderate_under_w*mu)


def plot_history(save_dir):
    hp=os.path.join(save_dir,"history.csv")
    if not os.path.exists(hp): return
    h=pd.read_csv(hp); plt.figure(figsize=(12,5))
    for c in ["train_loss","dev_rmse","dev_ccc","dev_corr","dev_score","dev_pred_std","train_residual_mean","train_residual_std","res_loss","dir_loss","mod_under","proto_kl","proto_score"]:
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
    print("[Model] V17 Residual Target Strengthening Expression Branch")
    print(f"[Window] seq_len={args.seq_len}, stride={args.stride}, overlap={args.seq_len-args.stride}")
    au_cols=[c.strip() for c in args.au_columns.split(",") if c.strip()]
    tr_meta,dev_meta,te_meta=read_txt(args.train_txt),read_txt(args.dev_txt),read_txt(args.test_txt)

    def load_split(meta,split):
        arrays=[]; paths=[]
        for _,r in tqdm(meta.iterrows(),total=len(meta),desc=f"load {split}",ncols=100):
            p=find_au_file(r["raw_path"],args.data_root); paths.append(p); arrays.append(load_au(p,au_cols,not args.no_success_filter))
        return arrays,paths

    tr_arr,tr_paths=load_split(tr_meta,"train"); mean,std=fit_global_stats(tr_arr)
    dev_arr,dev_paths=load_split(dev_meta,"dev"); te_arr,te_paths=load_split(te_meta,"test")

    Xtr_raw=np.stack([make_descriptor(a,mean,std) for a in tr_arr]); Xdev_raw=np.stack([make_descriptor(a,mean,std) for a in dev_arr]); Xte_raw=np.stack([make_descriptor(a,mean,std) for a in te_arr])
    (Xs,dmu,dsig)=standardize(Xtr_raw,Xdev_raw,Xte_raw); Xtr,Xdev,Xte=Xs
    ytr=tr_meta["score"].values.astype(np.float32); ydev=dev_meta["score"].values.astype(np.float32); yte=te_meta["score"].values.astype(np.float32)
    anchor=fit_anchor(Xtr,ytr,Xdev,ydev,Xte,yte,args)
    with open(os.path.join(args.save_dir,"anchor_summary.json"),"w",encoding="utf-8") as f: json.dump(to_jsonable(anchor["summary"]|{"cv_table":anchor["cv_table"]}),f,indent=2,ensure_ascii=False)
    print("[Anchor]"); print(json.dumps(to_jsonable(anchor["summary"]),indent=2,ensure_ascii=False))

    tr_v=build_videos(tr_meta,tr_arr,tr_paths,anchor["train_oof"],mean,std,dmu,dsig,args,"train")
    dev_v=build_videos(dev_meta,dev_arr,dev_paths,anchor["dev"],mean,std,dmu,dsig,args,"dev")
    te_v=build_videos(te_meta,te_arr,te_paths,anchor["test"],mean,std,dmu,dsig,args,"test")

    ds=ExprDataset(tr_v,train=True,args=args); sampler=BalancedSampler(tr_v,args.batch_size,args.steps_per_epoch,args.seed)
    loader=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=args.num_workers,pin_memory=True)
    model=V17(tr_v[0].segments.shape[-1],tr_v[0].desc.shape[0],args.hidden_dim,args.n_proto,args.expr_score_cap,args.residual_max,args.dropout,args.proto_tau,args.attn_temp).to(DEVICE)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.min_lr)

    best_score=1e9; best_epoch=-1; patience=0; hist=[]; grad_rows=[]; best_path=os.path.join(args.save_dir,"aee_best.pth")
    progress_start = time.monotonic()
    for ep in range(1,args.epochs+1):
        model.train(); rows=[]; pbar=tqdm(loader,desc=f"ExprV17 Ep{ep}",ncols=145)
        for bi,b in enumerate(pbar,1):
            x=b["x"].to(DEVICE); mask=b["mask"].to(DEVICE); bag=b["bag_mask"].to(DEVICE); desc=b["desc"].to(DEVICE); anch=b["anchor"].to(DEVICE); y=b["y"].to(DEVICE)
            opt.zero_grad(set_to_none=True); out=model(x,mask,bag,desc,anch); loss,st=loss_fn(out,y,anch,model,args,ep); loss.backward()
            gn=0.0
            for p in model.parameters():
                if p.grad is not None: gn+=float((p.grad.detach()**2).sum().cpu())
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step()
            st["epoch"]=ep; st["batch"]=bi; rows.append(st); grad_rows.append({"epoch":ep,"batch":bi,"grad_total":math.sqrt(gn)})
            pbar.set_postfix(loss=f"{st['loss']:.3f}",ps=f"{st['pred_std']:.2f}",rs=f"{st['residual_std']:.2f}",r=f"{st['ramp']:.2f}")
        sch.step(); ldf=pd.DataFrame(rows)
        dev_df,dev_m,dev_bm=evaluate(model,dev_v,args,"dev"); score=select_score(dev_m,dev_bm,args)
        rec={"epoch":ep,"lr":float(opt.param_groups[0]["lr"]),"train_loss":float(ldf.loss.mean()),"train_pred_std":float(ldf.pred_std.mean()),"train_residual_mean":float(ldf.residual_mean.mean()),"train_residual_std":float(ldf.residual_std.mean()),"res_loss":float(ldf.res_loss.mean()),"dir_loss":float(ldf.dir_loss.mean()),"mod_under":float(ldf.mod_under.mean()),"proto_kl":float(ldf.proto_kl.mean()),"proto_score":float(ldf.proto_score.mean()),"dev_score":score,"dev_rmse":dev_m["rmse"],"dev_mae":dev_m["mae"],"dev_ccc":dev_m["ccc"],"dev_corr":dev_m["corr"],"dev_pred_mean":dev_m["pred_mean"],"dev_pred_std":dev_m["pred_std"]}
        hist.append(rec); pd.DataFrame(hist).to_csv(os.path.join(args.save_dir,"history.csv"),index=False); pd.DataFrame(grad_rows).to_csv(os.path.join(args.save_dir,"gradient_flow.csv"),index=False)
        print(f"Ep{ep} | score={score:.4f} | DEV RMSE={dev_m['rmse']:.4f} MAE={dev_m['mae']:.4f} CCC={dev_m['ccc']:.4f} corr={dev_m['corr']:.4f} std={dev_m['pred_std']:.3f}")
        eta = (time.monotonic() - progress_start) / ep * (args.epochs - ep)
        print(f"APNET_PROGRESS Epoch {ep}/{args.epochs} | train_loss={rec['train_loss']:.4f} | "
              f"lr={rec['lr']:.3g} | ETA={eta/60:.1f} min", flush=True)
        if score<best_score:
            best_score=score; best_epoch=ep; patience=0
            torch.save({"model_state":model.state_dict(),"args":vars(args),"au_columns":au_cols,"raw_mean":mean,"raw_std":std,"desc_mu":dmu,"desc_sig":dsig,"anchor_beta":anchor["beta"],"anchor_alpha":anchor["alpha"],"best_epoch":best_epoch,"best_dev_score":best_score,"proto_scores":model.proto_scores.detach().cpu().numpy(),"protocol":"train_only_anchor_residual_strengthened_dev_select_test_report","model_name":"ExprV17AnchorResidualCompressedLexicon"},best_path)
            dev_df.to_csv(os.path.join(args.save_dir,"best_dev_predictions.csv"),index=False); dev_bm.to_csv(os.path.join(args.save_dir,"best_dev_bin_metrics.csv"),index=False)
        else: patience+=1
        if ep>=args.min_epochs and patience>=args.early_stop:
            print(f"Early stop at epoch {ep}."); break

    ckpt=load_checkpoint(best_path,map_location=DEVICE); model.load_state_dict(ckpt["model_state"],strict=True)
    dev_df,dev_m,dev_bm=evaluate(model,dev_v,args,"dev"); te_df,te_m,te_bm=evaluate(model,te_v,args,"test")
    dev_df.to_csv(os.path.join(args.save_dir,"final_dev_predictions.csv"),index=False); te_df.to_csv(os.path.join(args.save_dir,"final_test_predictions.csv"),index=False)
    dev_bm.to_csv(os.path.join(args.save_dir,"final_dev_bin_metrics.csv"),index=False); te_bm.to_csv(os.path.join(args.save_dir,"final_test_bin_metrics.csv"),index=False)
    proto_cols=[c for c in te_df.columns if c.startswith("proto") and c.endswith("_mean")]; pu=[]
    for c in proto_cols:
        k=int(c.replace("proto","").replace("_mean","")); row={"proto":k,"score_anchor":float(model.proto_scores[k].detach().cpu())}
        for b in ["minimal_0_5","mild_6_11","moderate_12_17","severe_18_24"]:
            g=te_df[te_df["bin"]==b]; row[b]=float(g[c].mean()) if len(g) else 0.0
        row["corr_gt"]=safe_corr(te_df[c],te_df["gt"]); pu.append(row)
    pu=pd.DataFrame(pu); pu.to_csv(os.path.join(args.save_dir,"prototype_usage_test.csv"),index=False)

    summary={"protocol":"train_only_anchor_residual_strengthened_dev_select_test_report","best_epoch":int(best_epoch),"best_dev_score":float(best_score),"anchor":anchor["summary"],"dev":{"all":dev_m,"anchor_only":metrics_np(dev_df["gt"],dev_df["anchor_pred"]),"bins":dev_bm.to_dict(orient="records")},"test":{"all":te_m,"anchor_only":metrics_np(te_df["gt"],te_df["anchor_pred"]),"bins":te_bm.to_dict(orient="records")},"prototype_usage_test":pu.to_dict(orient="records"),"recommend_checkpoint":best_path,"seed":args.seed,"model_config":{"n_proto":args.n_proto,"expr_score_cap":args.expr_score_cap,"residual_max":args.residual_max,"residual_target_scale":args.residual_target_scale,"residual_minimal_scale":args.residual_minimal_scale,"residual_moderate_scale":args.residual_moderate_scale,"seq_len":args.seq_len,"stride":args.stride,"overlap_frames":args.seq_len-args.stride,"input_mode":args.input_mode},"notes":{"core":"train-only shallow AU anchor + strengthened bounded temporal-MIL residual","purpose":"make residual correction closer to the scaled residual target, especially negative residual for minimal and positive residual for moderate","decision":"V17 removes V16 boost and returns to V15 conservative backbone"}}
    with open(os.path.join(args.save_dir,"train_summary.json"),"w",encoding="utf-8") as f: json.dump(to_jsonable(summary),f,indent=2,ensure_ascii=False)
    plot_history(args.save_dir); plot_pred(dev_df,args.save_dir,"dev"); plot_pred(te_df,args.save_dir,"test")
    print(json.dumps(to_jsonable(summary),indent=2,ensure_ascii=False)); print("\nFinal TEST bin metrics:"); print(te_bm.to_string(index=False))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train_txt",default="train_2017.txt"); p.add_argument("--dev_txt",default="dev_2017.txt"); p.add_argument("--test_txt",default="test_2017.txt"); p.add_argument("--data_root",default=""); p.add_argument("--save_dir",required=True)
    p.add_argument("--au_columns",default=",".join(DEFAULT_AU_COLUMNS)); p.add_argument("--no_success_filter",action="store_true")
    p.add_argument("--input_mode",default="raw_rel_delta",choices=["raw","raw_rel","raw_rel_delta"]); p.add_argument("--au_clip",type=float,default=5.0); p.add_argument("--robust_eps",type=float,default=0.15); p.add_argument("--rel_clip",type=float,default=5.0); p.add_argument("--delta_clip",type=float,default=5.0)
    p.add_argument("--anchor_folds",type=int,default=5); p.add_argument("--anchor_alphas",default="3,10,30,100,300,1000,3000")
    p.add_argument("--seq_len",type=int,default=300); p.add_argument("--stride",type=int,default=100); p.add_argument("--max_segments",type=int,default=128); p.add_argument("--max_train_segments_per_video",type=int,default=40)
    p.add_argument("--hidden_dim",type=int,default=64); p.add_argument("--n_proto",type=int,default=10); p.add_argument("--expr_score_cap",type=float,default=14.0); p.add_argument("--residual_max",type=float,default=3.5); p.add_argument("--dropout",type=float,default=0.08); p.add_argument("--proto_tau",type=float,default=0.16); p.add_argument("--attn_temp",type=float,default=0.85)
    p.add_argument("--epochs",type=int,default=120); p.add_argument("--min_epochs",type=int,default=20); p.add_argument("--early_stop",type=int,default=32); p.add_argument("--batch_size",type=int,default=16); p.add_argument("--steps_per_epoch",type=int,default=64); p.add_argument("--num_workers",type=int,default=0); p.add_argument("--lr",type=float,default=7e-5); p.add_argument("--min_lr",type=float,default=5e-6); p.add_argument("--weight_decay",type=float,default=9e-4); p.add_argument("--grad_clip",type=float,default=5.0); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--segment_keep_ratio",type=float,default=0.92); p.add_argument("--noise_std",type=float,default=0.005); p.add_argument("--feature_dropout",type=float,default=0.005); p.add_argument("--time_mask_prob",type=float,default=0.04); p.add_argument("--time_mask_ratio",type=float,default=0.04)
    p.add_argument("--huber_beta",type=float,default=1.0); p.add_argument("--loss_minimal_weight",type=float,default=1.35); p.add_argument("--loss_mild_weight",type=float,default=1.20); p.add_argument("--loss_moderate_weight",type=float,default=0.95); p.add_argument("--loss_severe_weight",type=float,default=0.50)
    p.add_argument("--w_huber",type=float,default=0.48); p.add_argument("--w_ccc",type=float,default=0.24); p.add_argument("--w_corr",type=float,default=0.08); p.add_argument("--w_rank",type=float,default=0.025); p.add_argument("--w_residual",type=float,default=0.50); p.add_argument("--w_res_l2",type=float,default=0.008); p.add_argument("--w_res_direction",type=float,default=0.10); p.add_argument("--w_moderate_under",type=float,default=0.045); p.add_argument("--w_low_bce",type=float,default=0.05); p.add_argument("--w_low_over",type=float,default=0.045); p.add_argument("--w_aux",type=float,default=0.025); p.add_argument("--w_dist",type=float,default=0.020)
    p.add_argument("--library_warmup_epochs",type=int,default=8); p.add_argument("--ordinal_sigma",type=float,default=1.35); p.add_argument("--w_proto_ordinal",type=float,default=0.08); p.add_argument("--w_proto_score",type=float,default=0.06); p.add_argument("--w_proto_diversity",type=float,default=0.020); p.add_argument("--w_attn_focus",type=float,default=0.004); p.add_argument("--w_proto_entropy",type=float,default=0.006)
    p.add_argument("--rank_margin",type=float,default=0.45); p.add_argument("--rank_min_gap",type=float,default=4.0); p.add_argument("--low_over_margin",type=float,default=1.0); p.add_argument("--residual_beta",type=float,default=0.75); p.add_argument("--residual_target_scale",type=float,default=0.60); p.add_argument("--residual_minimal_scale",type=float,default=0.65); p.add_argument("--residual_mild_scale",type=float,default=0.55); p.add_argument("--residual_moderate_scale",type=float,default=0.65); p.add_argument("--residual_severe_scale",type=float,default=0.45); p.add_argument("--residual_minimal_weight",type=float,default=1.60); p.add_argument("--residual_mild_weight",type=float,default=1.20); p.add_argument("--residual_moderate_weight",type=float,default=1.45); p.add_argument("--residual_severe_weight",type=float,default=0.55); p.add_argument("--residual_magnitude_weight",type=float,default=0.35); p.add_argument("--residual_push_fraction",type=float,default=0.55); p.add_argument("--mild_push_weight",type=float,default=0.35); p.add_argument("--moderate_under_margin",type=float,default=5.5)
    p.add_argument("--select_rmse_w",type=float,default=1.0); p.add_argument("--select_ccc_w",type=float,default=0.35); p.add_argument("--select_corr_w",type=float,default=0.16); p.add_argument("--select_minimal_over_w",type=float,default=0.035); p.add_argument("--select_mild_bias_w",type=float,default=0.020); p.add_argument("--select_moderate_under_w",type=float,default=0.015)
    args=p.parse_args(); main(args)
