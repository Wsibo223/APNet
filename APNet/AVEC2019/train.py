#!/usr/bin/env python3
"""Reinitialize both networks, train the Ridge head, and evaluate the fixed recipe once."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from fusion import RECIPE, fit, score_test, write_json

HERE = Path(__file__).resolve().parent
RUNS = json.loads((HERE / "configs" / "train.json").read_text())
SPLITS = ("train", "dev", "test")
EPOCH = re.compile(r"^Epoch (\d+)/(\d+) \| seconds=([\d.]+)\b")
TRAIN_FIELDS = ("train_loss=", "lr=", "train_pred_std=", "residual_loss=")
# Child output includes multi-line Anchor and export metrics; only forward known
# non-metric progress messages. The original child output still goes into logs.
SETUP_MESSAGES = ("[Device] ", "[Model] ", "[Window] ", "[Feature] ", "[ARD-Retardation-FAST] ",
                  "Prepared ", "Data prepared at ", "Early stop at epoch ")
INPUT_PROGRESS = re.compile(r"^(?:load (?:train|dev|test)|segment (?:train|dev|test)|building/cache videos):")


def absolute(path):
    if not path.is_absolute():
        raise ValueError(f"Absolute path required: {path}")
    return path.resolve()


def split_paths(root):
    manifest = {s: root / f"{s}_2019.txt" for s in SPLITS}
    missing = [str(p) for p in manifest.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing split lists: " + ", ".join(missing))
    return manifest


def stage(script, argv, out, step, report, env):
    target = ["-m", script] if isinstance(script, str) else [str(script)]
    command = [sys.executable, "-B", "-u", *target, *map(str, argv)]
    log_path = out / (step + ".log")
    t0 = time.monotonic()
    print(f"\n[{step}] starting; log: {log_path}", flush=True)
    elapsed, count = 0.0, 0
    last_input_progress = t0 - 15.0
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=out, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                              bufsize=1) as proc:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                log.write(line + "\n")
                log.flush()
                m = EPOCH.match(line)
                if m:
                    epoch, maximum = int(m[1]), int(m[2])
                    elapsed, count = elapsed + float(m[3]), count + 1
                    remaining = (elapsed / count) * (maximum - epoch)
                    fields = [field.strip() for field in line.split("|")[2:]
                              if field.strip().startswith(TRAIN_FIELDS)]
                    display = " | ".join([f"Epoch {epoch}/{maximum}", *fields, f"{float(m[3]):.1f}s",
                                          f"ETA (epoch limit) {remaining/60:.1f} min"])
                    print(f"[{step}] {display}", flush=True)
                elif line.startswith(SETUP_MESSAGES):
                    print(line, flush=True)
                elif INPUT_PROGRESS.match(line):
                    now = time.monotonic()
                    if "100%|" in line or now - last_input_progress >= 15.0:
                        print(line, flush=True)
                        last_input_progress = now
            code = proc.wait()
    report["stages"][step] = {"exit": code, "seconds": round(time.monotonic() - t0, 3)}
    if code:
        raise RuntimeError(f"{step} exited {code}; see {log_path}")
    print(f"[{step}] done in {report['stages'][step]['seconds']:.1f}s", flush=True)


def file_hash(path):
    acc = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            acc.update(block)
    return acc.hexdigest()


def train_args(recipe, manifest, withheld, model_dir, cache_dir):
    # Do not pass the Test manifest to the training process.
    replacements = {"${TRAIN_TXT}": manifest["train"], "${DEV_TXT}": manifest["dev"],
                    "${TEST_TXT}": withheld, "${SAVE_DIR}": model_dir, "${CACHE_ROOT}": cache_dir}
    return [replacements.get(v, v) for v in recipe["args"]] + ["--train-dev-only"]


def export_args(key, path, manifest, out, cache, splits, withheld):
    argv = ["--ckpt", path] if key == "aee" else ["--run_dir", path.parent, "--cache_root", cache]
    for split in SPLITS:
        argv += [f"--{split}_txt", manifest[split] if split in splits else withheld]
    return argv + ["--out_dir", out, "--splits", *splits]


def save_reports(out, report):
    write_json(out / "report.json", report)
    lines = ["# AVEC2019 from-scratch training report", "",
             "- Source: `AVEC2019/train.py`, using `configs/train.json` and `configs/fusion.json`.",
             f"- Status: `{report['status']}`.",
             "- Purpose: random initialization of AEE and PAE, re-fit Ridge and evaluate the fixed recipe.",
             "- The recipe originated from a historical experiment on this split; this is not an independent benchmark.", ""]
    if "error" in report:
        lines.extend([f"- Failure stage: `{report['failed_at']}`", f"- Error: {report['error']}", ""])
    test_metrics = report.get("metrics", {}).get("test")
    if test_metrics is not None:
        m = test_metrics
        lines.extend(["| Split | n | MAE | RMSE | CCC | PCC |", "|---|---:|---:|---:|---:|---:|"])
        lines.append(f"| Test | {m['n']} | {m['mae']:.8f} | {m['rmse']:.8f} | {m['ccc']:.8f} | {m['corr']:.8f} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if test_metrics is not None:
        print("\n" + "\n".join(lines[-3:]), flush=True)
    print(f"Status: {report['status']}", flush=True)
    print(f"Report: {out / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--out", required=True, type=Path, help="Fresh directory below --project")
    parser.add_argument("--compat-root", type=Path, help="Existing original AEE split lists")
    parser.add_argument("--delta-root", type=Path, help="Existing delta-pose split lists")
    parser.add_argument("--data", type=Path, help="Combined OpenFace CSV fallback when --delta-root is absent")
    args = parser.parse_args()
    out = None
    step = "preflight"
    report = {"mode": "train_from_scratch", "initialization": "random_for_both_branches",
              "pretrained_weights_used": False, "old_predictions_used": False,
              "status": "failed", "stages": {}, "branches": {}}
    try:
        os.umask(0o077)
        project = absolute(args.project)
        dest = absolute(args.out)
        if not project.is_dir() or dest == project or not dest.is_relative_to(project):
            raise ValueError("--out must be a new directory inside an existing --project")
        if any(c.isspace() for c in str(dest)):
            raise ValueError("The training directory cannot have spaces because split lists are whitespace-delimited")
        dest.mkdir(parents=True, exist_ok=False)
        out = dest
        env = {**os.environ, "MPLCONFIGDIR": str(out / ".mpl"), "TMPDIR": str(out / ".tmp"),
               "PYTHONPATH": str(HERE) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env.setdefault(key, "4")
        for directory in (out / ".mpl", out / ".tmp"):
            directory.mkdir()
        print(f"New isolated run: {out}", flush=True)
        raw = absolute(args.compat_root or project / "AVEC2019_compat")
        split_paths(raw)
        if args.delta_root:
            inputs = {"aee": raw, "pae": absolute(args.delta_root)}
        else:
            step = "prepare_inputs"
            rebuilt = out / "prepared"
            stage(HERE / "prepare_data.py", ["--project", project, "--from-compat", raw,
                  "--data", absolute(args.data or project / "AU/data"), "--out", rebuilt],
                  out, step, report, env)
            inputs = {k: rebuilt / name for k, name in
                      (("aee", "AVEC2019_compat"), ("pae", "AVEC2019_delta_pose_compat"))}
        manifests = {k: split_paths(v) for k, v in inputs.items()}
        withheld = out / "withheld_manifest_not_created.txt"
        fresh = {}
        exporters = {"aee": "branches.aee.export", "pae": "branches.pae.export"}
        for key, recipe in RUNS.items():
            step = "train_" + key
            model_dir = out / (key + "_training")
            stage(recipe["module"],
                  train_args(recipe, manifests[key], withheld, model_dir, out / (key + "_training_cache")),
                  out, step, report, {**env, **recipe.get("env", {})})
            fresh[key] = model_dir / recipe["new_checkpoint_name"]
            if not fresh[key].is_file() or not fresh[key].resolve().is_relative_to(out):
                raise RuntimeError(f"{key}: did not produce a new checkpoint under this run")
            summary = json.loads((model_dir / "train_summary.json").read_text())
            if summary.get("protocol") != "train_from_random_initialization_dev_checkpoint_selection_no_test":
                raise RuntimeError(f"{key}: missing Train/Dev-only checkpoint evidence")
            if not 1 <= summary["best_epoch"] <= summary["epochs_completed"]:
                raise RuntimeError(f"{key}: invalid epoch record")
            report["branches"][key] = {**summary, "recommend_checkpoint": str(fresh[key].relative_to(out)),
                                        "checkpoint_sha256": file_hash(fresh[key])}
            print(f"[{step}] checkpoint saved (epoch {summary['best_epoch']} of {summary['epochs_completed']} completed)", flush=True)

        # Only checkpoints created by this process are opened here; historical weights are not packaged.
        export_env = {**env, "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}
        dev_exports = {k: out / (k + "_train_dev_exports") for k in fresh}
        test_exports = {k: out / (k + "_test_export") for k in fresh}
        for key in fresh:
            step = "export_" + key + "_train_dev"
            stage(exporters[key],
                  export_args(key, fresh[key], manifests[key], dev_exports[key], out / "export_cache",
                              ("train", "dev"), withheld), out, step, report, export_env)
        step = "fit_ridge"
        t0 = time.monotonic()
        print(f"\n[fit_ridge] Fitting Ridge | Train samples={RECIPE['n']['train']} | "
              f"features={len(RECIPE['feature_columns'])} | alpha={RECIPE['alpha']} | single fit", flush=True)
        model_path, participants, metrics = fit(dev_exports, out / "fusion")
        model_hash = file_hash(model_path)
        report["fusion"] = {"model": str(model_path.relative_to(out)), "sha256_before_test": model_hash,
                            "alpha": RECIPE["alpha"], "dev_bias_target": RECIPE["dev_bias_target"]}
        report["metrics"] = metrics
        report["stages"][step] = {"exit": 0, "seconds": round(time.monotonic() - t0, 3)}
        print(f"[fit_ridge] Fitted and frozen in {report['stages'][step]['seconds']:.2f}s; "
              f"model: {model_path.relative_to(out)}. Starting Test inference.", flush=True)
        for key in fresh:
            step = "export_" + key + "_test"
            stage(exporters[key],
                  export_args(key, fresh[key], manifests[key], test_exports[key], out / "export_cache",
                              ("test",), withheld), out, step, report, export_env)
        step = "score_test"
        t0 = time.monotonic()
        if file_hash(model_path) != model_hash:
            raise RuntimeError("Frozen fusion model changed before Test")
        metrics["test"] = score_test(test_exports, model_path, participants)
        report["stages"][step] = {"exit": 0, "seconds": round(time.monotonic() - t0, 3)}
        report["historical_test_reference"] = RECIPE["historical_test_reference"]
        report["test_delta_from_historical"] = {
            k: metrics["test"][k] - v for k, v in RECIPE["historical_test_reference"].items()}
        report["status"] = "completed_from_scratch"
    except (OSError, ValueError, TypeError, RuntimeError, KeyError) as exc:
        report.update(failed_at=step, error=str(exc))
        print(f"ERROR [{step}]: {exc}", file=sys.stderr, flush=True)
    finally:
        if out:
            save_reports(out, report)
    return 0 if report["status"] == "completed_from_scratch" else 1


if __name__ == "__main__":
    sys.exit(main())
