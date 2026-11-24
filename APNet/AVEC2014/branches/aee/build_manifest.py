#!/usr/bin/env python3
"""Scan OpenFace AVEC2014 CSVs and merge BDI-II labels.

Output text format consumed by train.py:
    absolute_csv_path score task pair_id split

The label table may be long (one row per clip) or compact (one row per pair).
Column names are detected from conservative aliases; explicit --*_col overrides are
available when the official/local label sheet uses different names.
"""

import argparse
import os
import re
from pathlib import Path
import pandas as pd

TASKS=("Freeform","Northwind")
SPLITS=("train","dev","test")


def choose_col(df, explicit, candidates, role, required=True):
    if explicit:
        if explicit not in df.columns: raise ValueError(f"{role} column {explicit!r} not found; columns={list(df.columns)}")
        return explicit
    lower={str(c).strip().lower():c for c in df.columns}
    for name in candidates:
        if name.lower() in lower: return lower[name.lower()]
    if required: raise ValueError(f"Cannot detect {role} column; use --{role}_col. columns={list(df.columns)}")
    return None


def normalize_task(x):
    s=str(x).strip().lower()
    if s in ("freeform","free_form","free-form"): return "Freeform"
    if s in ("northwind","north_wind","north-wind"): return "Northwind"
    return ""


def scan_features(root):
    root=Path(root).expanduser().resolve(); rows=[]
    pat=re.compile(r"^(?P<pair>.+?)_(?P<task>Freeform|Northwind)_video\.csv$",re.I)
    for split in SPLITS:
        for task in TASKS:
            folder=root/split/task
            if not folder.is_dir(): raise FileNotFoundError(f"Missing expected directory: {folder}")
            for path in sorted(folder.glob("*.csv")):
                m=pat.match(path.name)
                if not m: continue
                parsed_task="Freeform" if m.group("task").lower()=="freeform" else "Northwind"
                if parsed_task!=task: raise ValueError(f"Task folder/filename mismatch: {path}")
                rows.append({"path":str(path),"split":split,"task":task,"pair_id":m.group("pair"),
                             "video_id":path.stem.removesuffix("_video")})
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError(f"No *_Freeform_video.csv or *_Northwind_video.csv under {root}")
    return out


def read_labels(args):
    path=Path(args.labels_csv).expanduser()
    df=pd.read_csv(path,sep=None,engine="python")
    df.columns=[str(c).strip() for c in df.columns]
    score_col=choose_col(df,args.score_col,("score","bdi","bdi_ii","bdi-ii","label","gt","depression_score"),"score")
    split_col=choose_col(df,args.split_col,("split","partition","set"),"split",required=False)
    task_col=choose_col(df,args.task_col,("task","condition"),"task",required=False)
    key_col=choose_col(df,args.key_col,("pair_id","video_id","recording_id","participant_id","subject_id","id"),"key")
    rows=[]
    for _,r in df.iterrows():
        score=float(r[score_col])
        if not 0<=score<=63: raise ValueError(f"BDI-II score must be in [0,63], got {score}")
        split=str(r[split_col]).strip().lower() if split_col and pd.notna(r[split_col]) else ""
        task=normalize_task(r[task_col]) if task_col and pd.notna(r[task_col]) else ""
        key=str(r[key_col]).strip()
        if key.endswith(".0") and re.fullmatch(r"\d+\.0",key): key=key[:-2]
        rows.append({"label_key":key,"label_split":split,"label_task":task,"score":score})
    return pd.DataFrame(rows)


def candidates(feature_row):
    # Exact pair/video keys first. The subject prefix fallback is intentionally last.
    pair=str(feature_row.pair_id); video=str(feature_row.video_id)
    subject=pair.split("_",1)[0]
    return (video,pair,subject)


def attach_labels(features,labels):
    scores=[]
    for r in features.itertuples(index=False):
        match=None
        for key in candidates(r):
            cand=labels[labels["label_key"]==key]
            if len(cand)==0: continue
            narrowed=cand
            split_specific=narrowed[narrowed["label_split"]==r.split]
            if len(split_specific): narrowed=split_specific
            task_specific=narrowed[narrowed["label_task"]==r.task]
            if len(task_specific): narrowed=task_specific
            unique=narrowed["score"].drop_duplicates()
            if len(unique)==1:
                match=float(unique.iloc[0]); break
            raise ValueError(f"Ambiguous labels for {r.path}: {narrowed.to_dict(orient='records')}")
        if match is None: raise ValueError(f"No label for {r.path}; tried keys={candidates(r)}")
        scores.append(match)
    out=features.copy(); out["score"]=scores
    for (split,pair),g in out.groupby(["split","pair_id"]):
        if set(g["task"])!=set(TASKS): raise ValueError(f"Incomplete task pair: split={split}, pair_id={pair}")
        if g["score"].nunique()!=1: raise ValueError(f"Task labels disagree: split={split}, pair_id={pair}")
    return out


def write_outputs(df,out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    df.to_csv(out_dir/"all_2014.csv",index=False)
    for split in SPLITS:
        sub=df[df["split"]==split].sort_values(["pair_id","task"])
        with open(out_dir/f"{split}_2014.txt","w",encoding="utf-8") as f:
            for r in sub.itertuples(index=False):
                f.write(f"{r.path} {r.score:g} {r.task} {r.pair_id} {r.split}\n")
        print(f"[{split}] clips={len(sub)}, pairs={sub['pair_id'].nunique()}, scores={sub['score'].min():g}..{sub['score'].max():g}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data_root",default="data/AVEC2014")
    p.add_argument("--labels_csv",required=True)
    p.add_argument("--out_dir",default="manifests")
    p.add_argument("--score_col"); p.add_argument("--split_col"); p.add_argument("--task_col"); p.add_argument("--key_col")
    args=p.parse_args()
    features=scan_features(args.data_root); labels=read_labels(args)
    write_outputs(attach_labels(features,labels),args.out_dir)


if __name__=="__main__": main()
