"""Rebuild the ONNX files from the zoo's .tflite files on Hugging Face (recipe.json: the 52 measured files, as
converted for the X2 Elite / Galaxy S26 columns, and the 13 <model>_fixed files converted with the FULLY_CONNECTED
bias fix of patch_tf2onnx.py).

    python convert_all.py --out <dir> [--model m ...]

Per model: download the .tflite at the pinned revision and check its sha256 -> for an fp16 file, fold the fp16
DEQUANTIZE weights into float32 (fold_deq.py, an identity transform) -> tf2onnx 1.17.0 --opset 17 (patched by
patch_tf2onnx.py; --continue_on_error where the recipe says so) -> compare the result with the file the table was
measured on. tf2onnx does not write the same bytes twice (initializer names, node order and the doc string change
from run to run), so the comparison is the graph hash of graph_signature.py (ops, attributes, wiring, weights; names
and order ignored), with the file sha256 as a second line. tf2onnx can also place a Transpose differently from run
to run (an equivalent graph with another hash): check_cpu.py then compares the CPU outputs with the table's.
Writes <out>/<name>.onnx, <out>/<name>.log and <out>/convert_report.json.
Run in the venv that has tensorflow 2.21.0, tf2onnx 1.17.0, onnx 1.23.2 and ai-edge-litert 2.2.0 (setup_venvs.ps1).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

from graph_signature import signature

HERE = os.path.dirname(os.path.abspath(__file__))
HF = "https://huggingface.co/{repo}/resolve/{rev}/{file}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dst):
    req = urllib.request.Request(url, headers={"User-Agent": "windows-zoo-convert"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst + ".tmp", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(dst + ".tmp", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", nargs="*")
    ap.add_argument("--summary", action="store_true", help="print the counts of <out>/convert_report.json, convert nothing")
    a = ap.parse_args()
    recipe = json.load(open(os.path.join(HERE, "recipe.json"), encoding="utf-8"))
    models = [m for m in recipe if not m.startswith("_")]
    if a.model:
        models = [m for m in models if m in a.model]
    os.makedirs(a.out, exist_ok=True)
    tfl_dir = os.path.join(a.out, "tflite")
    os.makedirs(tfl_dir, exist_ok=True)
    rep_path = os.path.join(a.out, "convert_report.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    for m in ([] if a.summary else models):
        r = recipe[m]
        t0 = time.time()
        row = {"recipe": {k: r[k] for k in ("repo", "revision", "file", "fold_fp16", "continue_on_error")}}
        src = os.path.join(tfl_dir, r["file"])
        if not os.path.exists(src) or sha256(src) != r["tflite_sha256"]:
            download(HF.format(repo=r["repo"], rev=r["revision"], file=r["file"]), src)
        row["tflite_sha256_ok"] = sha256(src) == r["tflite_sha256"]
        tfl = src
        if r["fold_fp16"]:
            tfl = os.path.join(tfl_dir, m + ".fp32.tflite")
            p = subprocess.run([sys.executable, os.path.join(HERE, "fold_deq.py"), src, tfl], capture_output=True, text=True)
            row["fold"] = (p.stdout.strip().splitlines() or [""])[-1][-160:]
            row["fold_rc"] = p.returncode
            if p.returncode != 0:
                row["error"] = p.stderr[-2000:]
                report[m] = row
                json.dump(report, open(rep_path, "w"), indent=1)
                print(f"{m}: fold failed", flush=True)
                continue
        dst = os.path.join(a.out, m + ".onnx")
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--tflite", tfl, "--output", dst, "--opset", "17"]
        if r["continue_on_error"]:
            cmd.append("--continue_on_error")
        # patch_tf2onnx.py's FULLY_CONNECTED bias fix is on only for the <model>_fixed entries; off rebuilds the measured files
        env = dict(os.environ, ZOO_FC_BIAS_FIX="1" if r.get("fc_bias_fix") else "0")
        row["recipe"]["fc_bias_fix"] = bool(r.get("fc_bias_fix"))
        with open(os.path.join(a.out, m + ".log"), "w", encoding="utf-8") as lf:
            row["tf2onnx_rc"] = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        row["seconds"] = round(time.time() - t0, 1)
        if os.path.exists(dst):
            row["onnx_sha256"] = sha256(dst)
            row["onnx_sha256_same_as_table"] = row["onnx_sha256"] == r["onnx_sha256"]
            row["onnx_bytes"] = os.path.getsize(dst)
            row["graph_hash"] = signature(dst)["graph_hash"]
            row["graph_same_as_table"] = row["graph_hash"] == r["graph_hash"]
        report[m] = row
        json.dump(report, open(rep_path, "w"), indent=1)
        print(f"{m}: tflite ok={row['tflite_sha256_ok']} fold={r['fold_fp16']} rc={row['tf2onnx_rc']} "
              f"graph_same={row.get('graph_same_as_table')} bytes_same={row.get('onnx_sha256_same_as_table')} "
              f"{row['seconds']} s", flush=True)
    if a.summary or not a.model:
        models = [m for m in recipe if not m.startswith("_") and m in report]
    graph = sum(1 for m in models if report.get(m, {}).get("graph_same_as_table"))
    same = sum(1 for m in models if report.get(m, {}).get("onnx_sha256_same_as_table"))
    print(f"{graph}/{len(models)} with the measured file's graph hash, {same}/{len(models)} byte-identical -> {rep_path}; "
          f"check the others by their CPU outputs (check_cpu.py)", flush=True)


if __name__ == "__main__":
    main()
