"""Terminal output for training: explicit progress records, complete stage logs."""
import re
import subprocess
import time

# Match only known training fields. Do not forward arbitrary lines or
# multiline evaluation summaries. The complete child output is saved.
NUMBER = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|nan|inf"
PROGRESS = re.compile(
    rf"APNET_PROGRESS ((?:Fold \d+/\d+ \| )?Epoch \d+/\d+ \| "
    rf"train_loss=(?:{NUMBER}) \| lr=(?:{NUMBER}) \| ETA=(?:{NUMBER}) min)"
)
INPUT_PROGRESS = re.compile(
    r"^(load (?:train|dev|test)|segment (?:train|dev|test)|building/cache videos|PRE token cache)"
    r":\s*(\d+)%"
)
STOP = re.compile(r"(?:Early stop at epoch (\d+)\.|fold=(\d+) early_stop epoch=(\d+))")

def visible_line(raw):
    line = raw.strip()
    m = PROGRESS.fullmatch(line)
    if m:
        return m.group(1)
    m = INPUT_PROGRESS.match(line)
    if m:
        return f"Preparing inputs: {m.group(2)}%"
    m = STOP.fullmatch(line)
    if m:
        return f"Early stop at epoch {m.group(1) or m.group(3)}"
    if line.startswith("[NGRC-REPLAY]"):
        return "Initializing the historical fusion training sequence"
    if line.startswith("[NGRC-TRAIN]"):
        return "Training fusion head"
    return None

def run_stage(name, command, out, env):
    log_path = out / f"{name}.log"
    started = time.monotonic()
    last_preparation = started - 15
    print(f"[{name}] starting; log: {log_path}", flush=True)
    with log_path.open("x", encoding="utf-8") as log:
        with subprocess.Popen(command, cwd=out, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                display = visible_line(line)
                if display is None:
                    continue
                if display.startswith("Preparing inputs:"):
                    now = time.monotonic()
                    if now - last_preparation < 15 and not display.endswith("100%"):
                        continue
                    last_preparation = now
                print(f"[{name}] {display}", flush=True)
            code = proc.wait()
    if code:
        raise RuntimeError(f"{name} failed (exit {code}); see {log_path}")
    print(f"[{name}] done in {time.monotonic()-started:.1f}s", flush=True)

def print_test_metrics(metrics):
    print("\n| Split | n | MAE | RMSE | CCC | PCC |")
    print("|---|---:|---:|---:|---:|---:|")
    print(f"| Test | {metrics['n']} | {metrics['mae']:.8f} | {metrics['rmse']:.8f} | "
          f"{metrics['ccc']:.8f} | {metrics['corr']:.8f} |", flush=True)
