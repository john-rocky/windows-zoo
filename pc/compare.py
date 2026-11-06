"""Compare every saved cell output of a pass with the CPU EP fp32 reference; one JSON per model x cell.

    python compare.py --pass first [--root C:\\src\\zoo] [--spec C:\\src\\zoo\\pc\\inputs_spec.json] [--model m ...]

Line (METHOD.md section 3): PASS = every output has cos >= 0.99 against the reference, no NaN / Inf, and every class-score
output (spec "argmax_outputs") keeps >= 0.99 of its argmax positions (positions where the reference top-1 and top-2
are exactly equal are not counted; a single-position output must match). Otherwise FAIL with the broken outputs and a
reason candidate. Writes <root>/parity/<pass>/<model>/<cell>.json.
"""

import argparse
import glob
import json
import os

import numpy as np

COS_LINE = 0.99
ARGMAX_LINE = 0.99
FP16_CELLS = {"cuda16", "ovgpu", "ovnpu", "ovnpu_pre"}  # cells that compute in fp16 by default


def cos(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def argmax_agreement(ref, test, axis, start=0, ssd_anchors=None):
    """Top-1 agreement along `axis` (classes from index `start`). ssd_anchors = [A, K]: the channel axis holds A anchors
    x K classes and is split first. Positions where the reference top-1 and top-2 are exactly equal are not counted."""
    r, t = ref.astype(np.float64), test.astype(np.float64)
    if ssd_anchors:
        a, k = ssd_anchors
        r = r.reshape(r.shape[:axis] + (a, k) + r.shape[axis + 1:])
        t = t.reshape(t.shape[:axis] + (a, k) + t.shape[axis + 1:])
        axis += 1
    r = np.moveaxis(r, axis, -1)[..., start:]
    t = np.moveaxis(t, axis, -1)[..., start:]
    r = r.reshape(-1, r.shape[-1])
    t = t.reshape(-1, t.shape[-1])
    srt = np.sort(r, axis=1)
    tie = srt[:, -1] == srt[:, -2] if r.shape[1] > 1 else np.zeros(len(r), bool)
    ra, ta = r.argmax(1), t.argmax(1)
    keep = ~tie
    n = int(keep.sum())
    agree = int((ra[keep] == ta[keep]).sum())
    margin = (srt[:, -1] - srt[:, -2]) if r.shape[1] > 1 else np.zeros(len(r))
    flipped = keep & (ra != ta)
    return {"positions": int(len(r)), "counted": n, "agree": agree, "rate": (agree / n) if n else None,
            "ties_skipped": int(tie.sum()), "classes_from": start,
            "flipped_ref_margin_median": float(np.median(margin[flipped])) if flipped.any() else None,
            "ref_margin_median": float(np.median(margin)) if len(margin) else None}


def argmax_specs_for(shape, specs):
    """The spec entries that apply to an output of this shape (matched by shape, or by channel count for ssdlite)."""
    out = []
    for a in specs:
        if "shape" in a and list(a["shape"]) == list(shape):
            out.append(a)
        elif "channels" in a and len(shape) > a["axis"] and shape[a["axis"]] == a["channels"]:
            out.append(a)
    return out


def compare_cell(ref, test, argmax_spec, cell):
    rnames = list(ref["_names"])
    tnames = list(test["_names"])
    outs = []
    fails = []
    for i, name in enumerate(rnames):
        r = ref[f"o{i}"]
        j = tnames.index(name) if name in tnames else i
        if f"o{j}" not in test:
            fails.append(f"{name}: missing in the cell output")
            outs.append({"name": str(name), "missing": True})
            continue
        t = test[f"o{j}"]
        row = {"name": str(name), "shape": list(r.shape)}
        if t.shape != r.shape:
            if t.size == r.size:
                t = t.reshape(r.shape)
                row["reshaped_from"] = list(test[f"o{j}"].shape)
            else:
                row["shape_mismatch"] = list(t.shape)
                fails.append(f"{name}: shape {list(t.shape)} vs {list(r.shape)}")
                outs.append(row)
                continue
        d = np.abs(t.astype(np.float64) - r.astype(np.float64))
        rng = float(np.max(r) - np.min(r)) if r.size else 0.0
        row.update({"max_abs": float(np.nanmax(d)) if d.size else 0.0, "mean_abs": float(np.nanmean(d)) if d.size else 0.0,
                    "cos": cos(np.nan_to_num(t), r), "nan": int(np.isnan(t).sum()), "inf": int(np.isinf(t).sum()),
                    "ref_min": float(np.min(r)), "ref_max": float(np.max(r)), "ref_range": rng})
        row["max_abs_over_range"] = row["max_abs"] / rng if rng else None
        if row["nan"] or row["inf"]:
            fails.append(f"{name}: {row['nan']} NaN, {row['inf']} Inf")
        if row["cos"] < COS_LINE:
            fails.append(f"{name}: cos {row['cos']:.4f} < {COS_LINE}")
        for a in argmax_specs_for(r.shape, argmax_spec):
            ag = argmax_agreement(r, t, a["axis"], a.get("start", 0), a.get("ssd_anchors"))
            row["argmax"] = ag
            if ag["rate"] is not None and (ag["rate"] < ARGMAX_LINE or (ag["counted"] == 1 and ag["agree"] != 1)):
                fails.append(f"{name}: argmax agreement {ag['agree']}/{ag['counted']}")
        outs.append(row)
    reason = None
    if fails:
        cands = []
        if any(o.get("nan") or o.get("inf") for o in outs) and cell in FP16_CELLS:
            cands.append("NaN / Inf in a cell that computes in fp16 (overflow candidate)")
        if any(max(abs(o.get("ref_min", 0)), abs(o.get("ref_max", 0))) > 65504 for o in outs) and cell in FP16_CELLS:
            cands.append("the reference exceeds the fp16 range (65,504)")
        if any("argmax" in f for f in fails) and all(o.get("cos", 1) >= COS_LINE for o in outs):
            cands.append("argmax flips where the reference margin between top-1 and top-2 is small (see flipped_ref_margin_median)")
        if not cands:
            cands.append("unknown")
        reason = "; ".join(cands)
    return {"status": "FAIL" if fails else "PASS", "fails": fails, "reason_candidate": reason, "outputs": outs}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pass", dest="pass_", default="first")
    p.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--spec", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs_spec.json"))
    p.add_argument("--model", nargs="*")
    args = p.parse_args()
    spec = json.load(open(args.spec, encoding="utf-8"))
    models = args.model or sorted(os.listdir(os.path.join(args.root, "out")))
    for m in models:
        d = os.path.join(args.root, "out", m)
        refp, ref_name = os.path.join(d, "cpu.npz"), "cpu"
        if not os.path.exists(refp) and os.path.exists(os.path.join(d, "ovcpu.npz")):
            # ONNX Runtime could not load the model (yolact): OpenVINO on the CPU in fp32 is the reference
            refp, ref_name = os.path.join(d, "ovcpu.npz"), "ovcpu"
        if not os.path.exists(refp):
            print(f"{m}: no reference")
            continue
        ref = np.load(refp, allow_pickle=False)
        amx = (spec.get(m) or spec.get(m[: -len("_fixed")] if m.endswith("_fixed") else m, {})).get("argmax_outputs", [])
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            cell = os.path.basename(f)[:-4]
            if cell == ref_name:
                continue
            test = np.load(f, allow_pickle=False)
            res = compare_cell(ref, test, amx, cell)
            res.update({"model": m, "cell": cell, "pass": args.pass_, "reference": ref_name, "line": {"cos": COS_LINE, "argmax": ARGMAX_LINE}})
            od = os.path.join(args.root, "parity", args.pass_, m)
            os.makedirs(od, exist_ok=True)
            with open(os.path.join(od, cell + ".json"), "w", encoding="utf-8") as fo:
                json.dump(res, fo, indent=1)
            worst = min((o.get("cos", 1.0) for o in res["outputs"]), default=None)
            print(f"{m} {cell} {res['status']} min_cos={worst} {'; '.join(res['fails'])[:160]}")


if __name__ == "__main__":
    main()
