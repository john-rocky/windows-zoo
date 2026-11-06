"""Check that your CPU EP outputs are the table's: same inputs, same model function, same numbers.

    python check_cpu.py --src <your root> [--model m ...]

For each model, reads your runs/first/<model>/cpu.json and out/<model>/cpu.npz (zoo_run.py --save-outputs) and the
committed results/<model>/cpu.json, then reports one of:
  identical  every output has the same sha256 as in the table
  close      not bit-identical, but at the 16 fixed positions the table recorded, every output is within 1e-3 x its
             standard deviation of the table's values, and its mean is within 1e-3 x its standard deviation
             (another CPU, another ONNX Runtime build, or an ONNX file rebuilt by tf2onnx)
  DIFFERENT  anything else (the line says which output and by how much)
  not run    ONNX Runtime's CPU EP did not run the file, neither in the table nor for you (yolact)
  inputs differ / missing   your input .npz is not the table's, or a file is not there
Exit 0 when no model is DIFFERENT, inputs differ or missing.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-3


def load_json(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def check(src, m):
    rec = load_json(os.path.join(HERE, "results", m, "cpu.json")) or {}
    table = rec.get("first_pass") or {}
    mine = load_json(os.path.join(src, "runs", "first", m, "cpu.json"))
    if rec.get("status") == "ERROR":
        if mine is not None and mine.get("status") == "OK":
            return "DIFFERENT", "your CPU EP ran this file; in the table it did not run"
        return "not run", "ONNX Runtime's CPU EP did not run this file in the table" + ("" if mine is None else " or for you")
    if not table.get("output_stats") or mine is None or mine.get("status") != "OK":
        return "missing", "no table record" if not table.get("output_stats") else "no OK run of yours"
    if mine.get("input_sha256") != table.get("input_sha256"):
        return "inputs differ", f"input sha256 {str(mine.get('input_sha256'))[:12]} vs table {str(table.get('input_sha256'))[:12]}"
    ts, ms = table["output_stats"], mine.get("output_stats") or []
    if [t["sha256"] for t in ts] == [o.get("sha256") for o in ms]:
        return "identical", f"{len(ts)} outputs"
    npz = os.path.join(src, "out", m, "cpu.npz")
    if not os.path.exists(npz):
        return "missing", "outputs differ in sha256 and out/<model>/cpu.npz is not there"
    worst = 0.0
    with np.load(npz) as z:
        outs = [z[f"o{i}"] for i in range(len(ts))]
    for i, (t, o) in enumerate(zip(ts, outs)):
        if list(o.shape) != t["shape"]:
            return "DIFFERENT", f"output {i} ({t['name']}): shape {list(o.shape)} vs {t['shape']}"
        flat = o.reshape(-1)
        scale = t["std"] if t["std"] > 0 else max(abs(t["mean"]), 1.0)
        d_sample = max((abs(float(flat[p]) - v) for p, v in zip(t["sample_pos"], t["sample"])), default=0.0) / scale
        d_mean = abs(float(flat.astype(np.float64).mean()) - t["mean"]) / scale
        worst = max(worst, d_sample, d_mean)
        if d_sample > TOL or d_mean > TOL:
            return "DIFFERENT", f"output {i} ({t['name']}): |delta| / std = {d_sample:.2e} at the 16 positions, {d_mean:.2e} in the mean"
    return "close", f"largest |delta| / std = {worst:.1e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="your root: the folder with runs/ and out/")
    ap.add_argument("--model", nargs="*")
    a = ap.parse_args()
    models = a.model or [l.strip() for l in open(os.path.join(HERE, "pc", "models.txt")) if l.strip()]
    count = {}
    for m in models:
        state, why = check(a.src, m)
        count[state] = count.get(state, 0) + 1
        print(f"{m}: {state} ({why})", flush=True)
    print(", ".join(f"{k} {v}" for k, v in sorted(count.items())))
    sys.exit(0 if set(count) <= {"identical", "close", "not run"} else 1)


if __name__ == "__main__":
    main()
