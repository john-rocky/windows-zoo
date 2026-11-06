"""Pull the PC's run / parity JSON and write one file per model x cell: results/<model>/<cell>.json.

    python3 merge.py [--no-pull]          on the Mac: pull from the PC over ssh into pc_pull/, then merge
    python merge.py --src .               on the PC, in the repo root: read runs/, parity/ and logs/ in place

PC layout (C:\\src\\zoo): runs/<pass>/<model>/<cell>.json (zoo_run.py), parity/<pass>/<model>/<cell>.json (compare.py),
logs/<pass>/<model>/<cell>.err (stderr of the unit). The pull goes to pc_pull/ (raw copy, not committed).
results/<model>/<cell>.json = {first_pass, compile (ovnpu_pre), quiet, parity, status, table}.
"""

import argparse
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PULL = os.path.join(HERE, "pc_pull")
CELLS = ["cpu", "cuda32", "cuda16", "ovgpu", "ovnpu", "ovnpu_pre", "ovcpu"]  # ovcpu = fallback reference (yolact)
MODELS = [l.strip() for l in open(os.path.join(HERE, "pc", "models52.txt")) if l.strip()]


def pull():
    os.makedirs(PULL, exist_ok=True)
    # one tar on the PC (tar.exe ships with Windows 11), one scp
    subprocess.run(["ssh", "winpc", "tar.exe -C C:/src/zoo -cf C:/src/zoo/pull.tar runs parity logs progress.txt"], check=False)
    subprocess.run(["scp", "-q", "winpc:C:/src/zoo/pull.tar", os.path.join(PULL, "pull.tar")], check=True)
    subprocess.run(["tar", "-C", PULL, "-xf", os.path.join(PULL, "pull.tar")], check=True)


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def fallback_lines(src, model, cell, pass_):
    """Lines of the unit's stderr that speak of a fallback or a device change (ONNX Runtime / OpenVINO EP warnings)."""
    p = os.path.join(src, "logs", pass_, model, cell + ".err")
    if not os.path.exists(p):
        return []
    text = open(p, encoding="utf-8", errors="replace").read().replace("\x00", "")
    return [l.strip()[:300] for l in text.splitlines() if re.search(r"fall ?back|falling back|not supported|unsupported", l, re.I)][:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--src", default=None, help="a folder with runs/, parity/, logs/ (default: pc_pull/ after the pull)")
    a = ap.parse_args()
    if a.src is None and not a.no_pull:
        pull()
    src = a.src or PULL
    n = 0
    for m in MODELS:
        for c in CELLS:
            first = load(os.path.join(src, "runs", "first", m, c + ".json"))
            comp = load(os.path.join(src, "runs", "first", m, c + "_compile.json")) if c == "ovnpu_pre" else None
            quiet = load(os.path.join(src, "runs", "quiet", m, c + ".json"))
            par = load(os.path.join(src, "parity", "first", m, c + ".json")) if c != "cpu" else None
            if first is None and comp is None:
                continue
            status = "ERROR"
            if comp is not None and comp.get("status") != "OK":
                status = "ERROR"
            elif first is not None and first.get("status") == "OK":
                status = "PASS" if c == "cpu" else (par or {}).get("status", "UNCOMPARED")
                if c == "ovcpu" and par is None:
                    status = "REFERENCE"  # the parity reference of a model ONNX Runtime cannot load
            q_ok = quiet is not None and quiet.get("status") == "OK"
            table = {
                "ms": (quiet or {}).get("ms", {}).get("median") if q_ok else None,
                "ms_first_pass": (first or {}).get("ms", {}).get("median") if first and first.get("status") == "OK" else None,
                "load_ms": (quiet or {}).get("load_ms") if q_ok else (first or {}).get("load_ms"),
                "load_ms_first_pass": (first or {}).get("load_ms"),
                "compile_ms": (comp or {}).get("compile_ms"),
                "gate": (quiet or {}).get("gate", {}).get("state") if q_ok else None,
            }
            for k in ("ms_all",):
                for r in (first, quiet):
                    if r and k in r:
                        r[k] = [round(x, 4) for x in r[k]]
            rec = {"model": m, "cell": c, "status": status, "table": table, "first_pass": first, "compile": comp, "quiet": quiet,
                   "parity": par, "stderr_fallback_lines": fallback_lines(src, m, c, "first")}
            od = os.path.join(HERE, "results", m)
            os.makedirs(od, exist_ok=True)
            with open(os.path.join(od, c + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            n += 1
    print(f"wrote {n} files under results/")


if __name__ == "__main__":
    main()
