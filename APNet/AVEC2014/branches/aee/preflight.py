#!/usr/bin/env python3
"""Fast AVEC2014 manifest/OpenFace validation without importing PyTorch."""
import argparse
from pathlib import Path
import pandas as pd

AU_CODES=("01","02","04","05","06","07","09","10","12","14","15","17","20","23","25","26","45")
REQ_AUS=tuple(f"AU{x}_r" for x in AU_CODES)+tuple(f"AU{x}_c" for x in AU_CODES)


def load(path,expected_split):
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        p,y,task,pair_id,split=line.split()[:5]
        y=float(y)
        if split!=expected_split: raise ValueError(f"Declared split mismatch in {path}: {line}")
        if task not in ("Freeform","Northwind"): raise ValueError(f"Unknown task: {task}")
        if not 0<=y<=63: raise ValueError(f"Score outside 0..63: {line}")
        csv=Path(p)
        if not csv.is_file(): raise FileNotFoundError(csv)
        header=[str(c).strip() for c in pd.read_csv(csv,nrows=0).columns]
        missing=[c for c in ("timestamp","confidence","success",*REQ_AUS) if c not in header]
        if missing: raise ValueError(f"{csv} missing columns: {missing}")
        rows.append({"path":p,"score":y,"task":task,"pair_id":pair_id,"split":split})
    df=pd.DataFrame(rows)
    if df.empty: raise ValueError(f"Empty manifest: {path}")
    for pair,g in df.groupby("pair_id"):
        if len(g)!=2 or set(g["task"])!={"Freeform","Northwind"}: raise ValueError(f"Bad task pair: {expected_split}/{pair}")
        if g["score"].nunique()!=1: raise ValueError(f"Label mismatch within pair: {expected_split}/{pair}")
    return df


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest_dir",default="manifests"); p.add_argument("--expected_pairs",type=int,default=50); args=p.parse_args()
    data={s:load(Path(args.manifest_dir)/f"{s}_2014.txt",s) for s in ("train","dev","test")}
    pair_sets={s:set(df["pair_id"]) for s,df in data.items()}
    for a,b in (("train","dev"),("train","test"),("dev","test")):
        overlap=pair_sets[a]&pair_sets[b]
        if overlap: raise ValueError(f"pair_id overlap {a}/{b}: {sorted(overlap)}")
    print("split  clips  pairs  min  max  mean")
    for s,df in data.items():
        labels=df.drop_duplicates("pair_id")["score"]
        if args.expected_pairs>0 and len(labels)!=args.expected_pairs:
            raise ValueError(f"{s} expected {args.expected_pairs} pairs, got {len(labels)}")
        print(f"{s:5s}  {len(df):5d}  {len(labels):5d}  {labels.min():3g}  {labels.max():3g}  {labels.mean():.2f}")
    print("PREFLIGHT OK: paths, labels, task pairs, timestamps, confidence and 17 AU r/c features are valid.")


if __name__=="__main__": main()
