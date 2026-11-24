#!/usr/bin/env python3
"""Build fresh AVEC2019 compatibility files from locally licensed OpenFace features."""
import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

SPLITS = ("train", "dev", "test")
META = ["frame", "timestamp", "confidence", "success"]
POSE = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz"]


def pose_delta(frame):
    aliases = {key: [key, f"pose_{key}"] for key in POSE}
    cols = {}
    for key, possible in aliases.items():
        found = next((n for n in possible if n in frame.columns), None)
        if found is None:
            raise ValueError(f"Required pose column not found: {key} or pose_{key}")
        v = pd.to_numeric(frame[found], errors="coerce")
        cols[key] = v.interpolate(limit_direction="both").ffill().bfill().fillna(0.0).to_numpy()
    x = np.stack([cols[k] for k in POSE], axis=1)
    diff = np.clip(np.diff(x, axis=0, prepend=x[:1]), -10, 10).astype(np.float32)
    meta = {"frame": np.arange(1, len(frame) + 1), "timestamp": np.arange(len(frame)) / 30,
            "confidence": np.ones(len(frame)), "success": np.ones(len(frame), dtype=int)}
    return pd.DataFrame({**{c: frame[c].to_numpy() if c in frame else meta[c] for c in META},
                         **{c: diff[:, i] for i, c in enumerate(POSE)}})


def read_labels(label_root, col_id, col_score, data_root):
    all_ids = set()
    rows = {}
    for split in SPLITS:
        table = pd.read_csv(label_root / f"{split}_split.csv", dtype=str)
        table.columns = table.columns.str.strip()
        if col_id not in table or col_score not in table:
            raise ValueError(f"{split}_split.csv: need {col_id!r} and {col_score!r}; columns are {list(table)}")
        rows[split] = []
        for raw_id, raw_score in zip(table[col_id], table[col_score]):
            m = re.fullmatch(r"(\d+)(?:\.0+)?", str(raw_id).strip())
            if not m or m[1] in all_ids:
                raise ValueError(f"{split}_split.csv: invalid or duplicate participant ID across splits")
            score = float(raw_score)
            if not math.isfinite(score) or not 0 <= score <= 24:
                raise ValueError(f"{split}_split.csv: invalid PHQ-8 score")
            full_csv = data_root / f"{m[1]}_OpenFace2.1.0_Pose_gaze_AUs.csv"
            if not full_csv.is_file():
                raise FileNotFoundError(full_csv)
            rows[split].append((m[1], score, full_csv))
            all_ids.add(m[1])
        if not rows[split]:
            raise ValueError(f"{split}_split.csv: no rows")
    return rows


def read_compat(root):
    rows = {}
    seen = set()
    for split in SPLITS:
        manifest = root / f"{split}_2019.txt"
        rows[split] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"{manifest.name}: invalid manifest entry")
            au_path = Path(parts[0])
            if not au_path.is_absolute():
                au_path = root / au_path
            if not au_path.is_file():
                raise FileNotFoundError(f"{manifest.name}: an AU reference is missing")
            score = float(parts[1])
            m = re.match(r"\d+", au_path.name)
            if not m or m[0] in seen or not math.isfinite(score) or not 0 <= score <= 24:
                raise ValueError(f"{manifest.name}: invalid ID, duplicate ID, or score")
            vid = m[0]
            rows[split].append((vid, score, au_path))
            seen.add(vid)
        if not rows[split]:
            raise ValueError(f"{manifest.name}: no rows")
    return rows


def historical_pose_source(au_path, vid, data_root):
    # Legacy delta builder first derived a separate pose CSV from the AEE-list input.
    candidates = []
    for suffix in ("_AUs.csv", "_OpenFace2.1.0_Pose_gaze_AUs.csv"):
        if au_path.name.endswith(suffix):
            candidates.append(au_path.with_name(au_path.name[:-len(suffix)] + "_pose.csv"))
    candidates.append(au_path.parent / f"{vid}_pose.csv")
    if data_root:
        candidates.append(data_root / f"{vid}_OpenFace2.1.0_Pose_gaze_AUs.csv")
    candidates.append(au_path)
    for candidate in dict.fromkeys(candidates):
        if candidate.is_file():
            header = set(pd.read_csv(candidate, nrows=0).columns.str.strip())
            if all(c in header or f"pose_{c}" in header for c in POSE):
                return candidate
    raise FileNotFoundError("No usable OpenFace pose source for an entry in the existing split manifests")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, default=Path.cwd())
    ap.add_argument("--data", type=Path, help="Raw combined OpenFace CSVs; optional fallback with --from-compat")
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--labels", type=Path)
    source.add_argument("--from-compat", type=Path, help="Keep the existing train/dev/test participant order and scores")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id-col", default="Participant_ID")
    ap.add_argument("--score-col", default="PHQ_Score")
    args = ap.parse_args()
    try:
        if not all(p is None or p.is_absolute() for p in (args.project, args.data, args.labels, args.from_compat, args.out)):
            raise ValueError("Use absolute paths")
        if args.labels and not args.data:
            raise ValueError("--labels requires --data")
        dest, project = args.out.resolve(), args.project.resolve()
        if not project.is_dir() or dest == project or not dest.is_relative_to(project) or any(c.isspace() for c in str(dest)):
            raise ValueError("Output must be a new directory with no spaces under the project")
        data_root = args.data.resolve() if args.data else None
        rows = (read_compat(args.from_compat.resolve()) if args.from_compat
                else read_labels(args.labels.resolve(), args.id_col, args.score_col, data_root))
        dest.mkdir(parents=True, exist_ok=False)
        originals, deltas = dest / "AVEC2019_compat", dest / "AVEC2019_delta_pose_compat"
        originals.mkdir()
        deltas.mkdir()
        for s, split_rows in rows.items():
            a, b = [], []
            for vid, score, full_csv in split_rows:
                pose_source = (historical_pose_source(full_csv, vid, data_root) if args.from_compat else full_csv)
                frame = pd.read_csv(pose_source)
                frame.columns = frame.columns.str.strip()
                au_cols = (pd.read_csv(full_csv, nrows=0).columns.str.strip()
                           if pose_source != full_csv else frame.columns)
                if frame.empty or not any(re.fullmatch(r"AU\d+_r", c) for c in au_cols):
                    raise ValueError("Empty pose source or no AU intensity columns")
                orig_au, delta_au = originals / f"{vid}_AUs.csv", deltas / f"{vid}_AUs.csv"
                delta_pose = deltas / f"{vid}_pose.csv"
                orig_au.symlink_to(full_csv)
                delta_au.symlink_to(full_csv)
                pose_delta(frame).to_csv(delta_pose, index=False)
                a.append(f"{orig_au} {score}\n")
                b.append(f"{delta_au} {score}\n")
            for d, lines in ((originals, a), (deltas, b)):
                (d / f"{s}_2019.txt").write_text("".join(lines), encoding="utf-8")
            print(f"Prepared {s}: {len(split_rows)} rows")
        print(f"Data prepared at {dest}")
    except (ValueError, OSError) as e:
        ap.exit(1, f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
