#!/usr/bin/env python3
"""Normalize existing AVEC2017 split lists while preserving their row order."""
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True,
                   help="Base path for relative paths in the original split lists.")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise FileExistsError(a.out)
    allpaths = set()
    tables = {}
    for split in ("train", "dev", "test"):
        rows = []
        for line in (a.splits / f"{split}_2017.txt").read_text().splitlines():
            if not line.strip():
                continue
            name, score = line.split()[:2]
            path = Path(name)
            if not path.is_absolute():
                path = a.data_root / path
            path = path.resolve(strict=True)
            if not path.is_file() or any(c.isspace() for c in str(path)):
                raise ValueError(f"Invalid OpenFace AU path: {path}")
            if path in allpaths:
                raise ValueError(f"Repeated feature file: {path}")
            if not 0 <= float(score) <= 24:
                raise ValueError("PHQ-8 score must be between 0 and 24")
            allpaths.add(path)
            rows.append(f"{path} {score}")
        if len(rows) != dict(train=107, dev=35, test=47)[split]:
            raise ValueError(f"Unexpected {split} size: {len(rows)}")
        tables[split] = rows
    a.out.mkdir(parents=True, exist_ok=False)
    for split, rows in tables.items():
        (a.out / f"{split}_2017.txt").write_text("\n".join(rows) + "\n")
        print(f"{split}: {len(rows)}")

if __name__ == "__main__":
    main()
