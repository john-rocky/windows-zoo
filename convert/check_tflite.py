"""Check each ONNX file against the .tflite it was converted from: same inputs, ONNX Runtime CPU outputs vs LiteRT.

    python convert/check_tflite.py [--root <root>] [--model m ...]      (in .venv-tf, after the first pass)

Per model: the zoo's original .tflite (the file on Hugging Face at the pinned revision, fp16 weights included, not
the folded copy; downloaded into <root>/onnx/tflite/ when missing, sha256 checked) runs once in the LiteRT
interpreter (ai-edge-litert) on the CPU with <root>/inputs/<model>.npz. Its outputs are the reference for the ONNX
file's CPU EP outputs saved by the first pass (<root>/out/<model>/cpu.npz; ovcpu.npz for a file ONNX Runtime cannot
load), with the pass line of pc/compare.py. Outputs are paired by shape (by name order among equal shapes).
Writes <root>/parity/tflite/<model>.json and prints one line per model.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pc"))
sys.path.insert(0, HERE)
from compare import compare_cell  # noqa: E402
from convert_all import HF, download, sha256  # noqa: E402


def interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter  # noqa: PLC0415
        import ai_edge_litert  # noqa: PLC0415

        return Interpreter(model_path=path), f"ai-edge-litert {getattr(ai_edge_litert, '__version__', '?')}"
    except ImportError:
        import tensorflow as tf  # noqa: PLC0415

        return tf.lite.Interpreter(model_path=path), f"tensorflow {tf.__version__} tf.lite"


def pair_by_shape(want, have):
    """For each wanted shape, the index of an unused available array of that shape (first come); None if none."""
    used, out = set(), []
    for w in want:
        j = next((k for k, h in enumerate(have) if k not in used and list(h) == list(w)), None)
        if j is None:
            j = next((k for k, h in enumerate(have) if k not in used and int(np.prod(h)) == int(np.prod(w))), None)
        if j is not None:
            used.add(j)
        out.append(j)
    return out


def check(root, m, r, spec):
    tdir = os.path.join(root, "onnx", "tflite")
    os.makedirs(tdir, exist_ok=True)
    src = os.path.join(tdir, r["file"])
    if not os.path.exists(src) or sha256(src) != r["tflite_sha256"]:
        download(HF.format(repo=r["repo"], rev=r["revision"], file=r["file"]), src)
    rec = {"model": m, "tflite": f"{r['repo']}@{r['revision'][:12]}/{r['file']}", "tflite_sha256_ok": sha256(src) == r["tflite_sha256"]}
    test_cell = "cpu" if os.path.exists(os.path.join(root, "out", m, "cpu.npz")) else "ovcpu"
    tp = os.path.join(root, "out", m, test_cell + ".npz")
    if not os.path.exists(tp):
        return {**rec, "status": "ERROR", "error": "no saved CPU output (out/<model>/cpu.npz or ovcpu.npz)"}
    rec["onnx_cell"] = test_cell
    test = dict(np.load(tp))
    names = [str(n) for n in test["_names"]]
    feeds = dict(np.load(os.path.join(root, "inputs", m + ".npz")))
    it, rec["interpreter"] = interpreter(src)
    it.allocate_tensors()
    ins = it.get_input_details()
    keys = list(feeds)
    by_name = [next((k for k in keys if k == d["name"] or k.split(":")[0] == d["name"].split(":")[0]), None) for d in ins]
    if None in by_name:
        idx = pair_by_shape([d["shape"] for d in ins], [feeds[k].shape for k in keys])
        by_name = [keys[j] if j is not None else None for j in idx]
        rec["inputs_paired"] = "by shape"
    else:
        rec["inputs_paired"] = "by name"
    if None in by_name:
        return {**rec, "status": "ERROR", "error": f"cannot pair inputs: tflite {[(d['name'], list(d['shape'])) for d in ins]} vs npz {[(k, list(feeds[k].shape)) for k in keys]}"}
    for d, k in zip(ins, by_name):
        it.set_tensor(d["index"], feeds[k].astype(d["dtype"]).reshape(d["shape"]))
    it.invoke()
    outs = [it.get_tensor(d["index"]) for d in it.get_output_details()]
    onames = [d["name"] for d in it.get_output_details()]
    idx = pair_by_shape([test[f"o{i}"].shape for i in range(len(names))], [o.shape for o in outs])
    rec["outputs_paired"] = [{"onnx": n, "tflite": (onames[j] if j is not None else None)} for n, j in zip(names, idx)]
    if None in idx:
        return {**rec, "status": "ERROR", "error": f"cannot pair outputs: onnx {[list(test[f'o{i}'].shape) for i in range(len(names))]} vs tflite {[list(o.shape) for o in outs]}"}
    ref = {"_names": np.array(names), **{f"o{i}": outs[j].astype(np.float32) for i, j in enumerate(idx)}}
    res = compare_cell(ref, test, spec, "tflite")
    return {**rec, **res, "reference": "the source .tflite in the LiteRT interpreter on the CPU", "line": {"cos": 0.99, "argmax": 0.99}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(HERE))
    ap.add_argument("--model", nargs="*")
    a = ap.parse_args()
    recipe = json.load(open(os.path.join(HERE, "recipe.json"), encoding="utf-8"))
    spec_all = json.load(open(os.path.join(a.root, "pc", "inputs_spec.json"), encoding="utf-8"))
    models = a.model or [m for m in recipe if not m.startswith("_")]
    od = os.path.join(a.root, "parity", "tflite")
    os.makedirs(od, exist_ok=True)
    count = {}
    for m in models:
        spec = (spec_all.get(m) or spec_all.get(m[: -len("_fixed")] if m.endswith("_fixed") else m, {})).get("argmax_outputs", [])
        try:
            res = check(a.root, m, recipe[m], spec)
        except Exception as e:  # noqa: BLE001 - one model's failure must not stop the others
            res = {"model": m, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:2000]}
        with open(os.path.join(od, m + ".json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
        count[res["status"]] = count.get(res["status"], 0) + 1
        cosmin = min((o["cos"] for o in res.get("outputs", []) if "cos" in o), default=None)
        print(f"{m}: {res['status']}" + (f" min cos {cosmin:.6f}" if cosmin is not None else "")
              + (f" :: {'; '.join(res['fails'])}" if res.get("fails") else "") + (f" :: {res['error'][:300]}" if res.get("error") else ""), flush=True)
    print(", ".join(f"{k} {v}" for k, v in sorted(count.items())))


if __name__ == "__main__":
    main()
