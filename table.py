"""Render RESULTS_WINDOWS.md from results/<model>/<cell>.json and the X2E / Galaxy S26 rows of the earlier AI Hub lane.

    python3 table.py [--x2e hub/results_wide.json]

hub/results_wide.json = the Qualcomm AI Hub records of the same 52 files on a Snapdragon X2 Elite and a Galaxy S26
(one profile job per model x device x form, 2026-10-30; a byte copy of the earlier lane's results_wide.json).

Speed = the quiet-pass median (PLAN.md section 6); a cell without a quiet number shows the first-pass median with *.
The conclusion lines are written by hand into CONCLUSION.md and copied in (only measured facts).
"""

import argparse
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [l.strip() for l in open(os.path.join(HERE, "pc", "models52.txt")) if l.strip()]
CELLS = ["ovnpu", "ovnpu_pre", "ovgpu", "cuda16", "cuda32", "cpu"]
CODE = {"ovnpu": "a", "ovnpu_pre": "p", "ovgpu": "g", "cuda16": "h", "cuda32": "f"}


def load(m, c):
    p = os.path.join(HERE, "results", m, c + ".json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def ms_text(r):
    if r is None:
        return "-", None
    if r["status"] == "ERROR":
        return "ERR", None
    t = r["table"]
    if t.get("ms") is not None:
        return f"{t['ms']:.2f}", t["ms"]
    if t.get("ms_first_pass") is not None:
        return f"{t['ms_first_pass']:.2f}*", t["ms_first_pass"]
    return "-", None


def cpu_nodes(r):
    fp = (r or {}).get("first_pass") or {}
    return ((fp.get("placement") or {}).get("nodes_by_provider") or {}).get("CPUExecutionProvider", 0)


def ran_on_device(r):
    """Did the process's GPU Engine counters show time on the cell's own adapter (the LUID the record names)?"""
    for part in ("quiet", "first_pass"):
        rec = (r or {}).get(part) or {}
        luid = (rec.get("device") or {}).get("counter_luid")
        e = rec.get("engine_running_ms") or {}
        if luid and any(k.startswith(luid + "/") and v > 0 for k, v in e.items()):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2e", default=os.path.join(HERE, "hub", "results_wide.json"))
    a = ap.parse_args()
    X = json.load(open(a.x2e))

    def hub(m, dev, form):
        r = X.get(f"{m}|{dev}|{form}|npu")
        if not r or r.get("status") != "SUCCESS":
            return "-", None, None
        cpu = (r.get("units") or {}).get("CPU", 0)
        return f"{r['infer_us'] / 1000:.2f}" + (f" (CPU {cpu})" if cpu else ""), r["infer_us"] / 1000, r.get("first_load_us")

    rows = []
    for m in MODELS:
        R = {c: load(m, c) for c in CELLS}
        s26, _, _ = hub(m, "S26", "pqo")
        x2e, _, _ = hub(m, "X2E", "pqo")
        _, _, x2e_asis_load = hub(m, "X2E", "onnx")
        cols = {}
        for c in ("ovnpu", "ovnpu_pre", "ovgpu", "cuda16", "cpu"):
            txt, _ = ms_text(R[c])
            if c == "ovnpu" and R[c] and R[c]["status"] != "ERROR":
                n = cpu_nodes(R[c])
                if n:
                    txt += f" (CPU {n})"
            cols[c] = txt
        loads = []
        for c in ("ovnpu", "ovnpu_pre"):
            r = R[c]
            v = (r or {}).get("table", {}).get("load_ms") if r and r["status"] != "ERROR" else None
            loads.append(f"{v / 1000:.2f}" if v is not None else "-")
        loads.append(f"{x2e_asis_load / 1e6:.2f}" if x2e_asis_load else "-")
        par = " ".join(f"{CODE[c]}:{(R[c] or {}).get('status', '-')[0] if R[c] else '-'}" for c in ("ovnpu", "ovnpu_pre", "ovgpu", "cuda16", "cuda32"))
        notes = []
        for c in ("ovnpu", "ovnpu_pre", "ovgpu", "cuda16", "cuda32", "cpu"):
            r = R[c]
            if r and r["status"] == "ERROR":
                err = ((r.get("compile") or {}).get("error") or (r.get("first_pass") or {}).get("error") or "")
                last = [l for l in err.strip().splitlines() if l.strip()][-1:] or ["?"]
                notes.append(f"{c} ERROR: {last[0][:90]}")
            elif r and r["status"] == "FAIL":
                notes.append(f"{c} FAIL: {'; '.join((r.get('parity') or {}).get('fails', []))[:90]}")
        for c in ("ovnpu", "ovnpu_pre", "ovgpu"):
            if R[c] and R[c]["status"] != "ERROR" and not ran_on_device(R[c]):
                notes.append(f"{c}: no engine time on its device in the process counters")
        rows.append(f"| {m} | {s26} | {x2e} | {cols['ovnpu']} | {cols['ovnpu_pre']} | {cols['ovgpu']} | {cols['cuda16']} | {cols['cpu']} | "
                    f"{' / '.join(loads)} | {par} | {'; '.join(notes)} |")

    concl = open(os.path.join(HERE, "CONCLUSION.md"), encoding="utf-8").read().strip() if os.path.exists(os.path.join(HERE, "CONCLUSION.md")) else "(conclusion pending)"
    head = [
        "# Vision zoo on one Windows laptop: Intel NPU, Intel GPU, RTX, CPU — next to Snapdragon X2 Elite and Galaxy S26\n",
        "52 single-graph ONNX files of the LiteRT zoo (the same files as the X2 Elite / Galaxy S26 columns), run on an MSI "
        "Stealth 16 AI+ (Core Ultra 9 386H with Intel AI Boost NPU and Intel Graphics, RTX 5080 Laptop GPU, Windows 11). "
        "Times are medians of 30 runs from host memory to host memory (copies included), taken one cell at a time on a quiet "
        "machine; * = first-pass number (no quiet re-run). The X2E / S26 columns are Qualcomm AI Hub's "
        "`estimated_inference_time` of the precompiled QNN form (one profile job per cell, 2026-10-30, "
        "`hub/results_wide.json`): AI Hub's own timer on the device, not this repo's host-to-host timer.\n",
        concl + "\n",
        "| model | Galaxy S26 NPU ms | X2E NPU ms | Intel NPU as-is ms | Intel NPU precompiled ms | Intel GPU ms | RTX fp16 ms | CPU ms | "
        "load s: Intel NPU as-is / Intel NPU precompiled / X2E as-is | parity | note |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    legend = [
        "",
        "Columns: Intel NPU as-is = ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1 on the NPU, the plain ONNX compiled at load; "
        "(CPU n) = n graph nodes left on the CPU EP. Intel NPU precompiled = OpenVINO 2026.4.1 `compile_model` → "
        "`export_model` blob, loaded with `import_model`. Intel GPU = OpenVINO EP on the Intel iGPU (default precision f16). "
        "RTX fp16 = ONNX Runtime CUDA EP on an fp16 copy of the model (`convert_float_to_float16`, I/O kept fp32). CPU = "
        "ONNX Runtime CPU EP fp32 = the parity reference.",
        "Parity (against the CPU EP fp32 outputs, PLAN.md section 3): a = Intel NPU as-is, p = precompiled, g = Intel GPU, "
        "h = RTX fp16, f = RTX fp32; P = PASS (every output cos ≥ 0.99, no NaN / Inf, class argmax kept ≥ 0.99), "
        "F = FAIL, E = ERROR (did not run; error in the note).",
    ]
    with open(os.path.join(HERE, "RESULTS_WINDOWS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(head + rows + legend) + "\n")
    print(f"{len(rows)} rows")


if __name__ == "__main__":
    main()
