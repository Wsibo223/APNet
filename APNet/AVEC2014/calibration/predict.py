#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from calibration_common import atomic_csv, load_tables, prediction_frame

def main(a):
    artifact = json.loads(Path(a.rule).read_text())
    config = artifact.get("config", artifact)
    frame = load_tables(a.p16_result, include_test=True)[a.split]
    out = prediction_frame(frame, config); atomic_csv(out, a.output_csv)
    print("DONE rows=%d output=%s" % (len(out), Path(a.output_csv).resolve()))

def parser():
    p=argparse.ArgumentParser(); p.add_argument("--rule",required=True)
    p.add_argument("--p16_result",required=True); p.add_argument("--split",choices=("train","dev","test"),default="test")
    p.add_argument("--output_csv",required=True); return p
if __name__=="__main__": main(parser().parse_args())
