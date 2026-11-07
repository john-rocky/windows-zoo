"""Write the Hugging Face card of each staged repo: hub/<repo>/README.md, next to the files pc/stage_hub.py staged.

    python3 cards.py [--only D-FINE-S-OpenVINO ...]

One repo per source model (convert/source_repos.json; a two-graph model keeps both graphs, as its LiteRT repo does).
Read: hub/<repo>/manifest.json (files, sha256, the blob's compile versions), results/<model>/<cell>.json (merge.py:
the quiet-pass median and load time, the parity against the CPU fp32 output), results/<model>/tflite.json (the ONNX
file against its source .tflite) and aihub/results_wide.json (the Snapdragon rows). No first-pass time goes into a
card: a cell without a quiet-pass number shows "-".
"""

import argparse
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
LISTED = [l.strip() for l in open(os.path.join(HERE, "pc", "models.txt")) if l.strip()]
GITHUB = "https://github.com/john-rocky/windows-zoo"
HUB_ORG = "mlboydaisuke"  # the proposed Hub owner (decided with the upload)
EP = "ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1"
RTX = "RTX 5080 Laptop (95 W limit)"  # nvidia-smi -q -d POWER: current power limit 95 W (read 2026-11-07)
ROWS = [  # cell, processor, runtime, file
    ("ovnpu_pre", "Intel AI Boost NPU", "OpenVINO 2026.4.1", "`npu/` blob (`import_model`)"),
    ("ovnpu", "Intel AI Boost NPU", EP, "`.onnx`, compiled when the session loads"),
    ("ovgpu", "Intel Graphics (iGPU)", EP, "`.onnx`"),
    ("cuda16", RTX, "ONNX Runtime 1.31.1, CUDA EP", "fp16 copy of the `.onnx` (not in this repo)"),
    ("cuda32", RTX, "ONNX Runtime 1.31.1, CUDA EP", "`.onnx`"),
    ("cpu", "Core Ultra 9 386H CPU", "ONNX Runtime 1.31.1, CPU EP", "`.onnx`"),
    ("ovcpu", "Core Ultra 9 386H CPU", "OpenVINO 2026.4.1", "`.onnx`, f32"),
]
HUB_ROWS = [  # key form, device, label, runtime, file
    ("X2E", "pqo", "Snapdragon X2 Elite NPU", "Qualcomm AI Hub, precompiled QNN ONNX", "`.onnx` compiled by AI Hub"),
    ("X2E", "onnx", "Snapdragon X2 Elite NPU", "Qualcomm AI Hub, ONNX Runtime QNN EP", "`.onnx`, compiled when it loads"),
    ("S26", "pqo", "Galaxy S26 NPU", "Qualcomm AI Hub, precompiled QNN ONNX", "`.onnx` compiled by AI Hub"),
]


def load(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def floor4(c):
    """c rounded down to 4 decimals, so that "≥ c" stays true when printed (0.99996 prints as 0.9999, not 1.0000)."""
    return math.floor(c * 1e4) / 1e4


def repo_folder(repo):  # same rule as pc/stage_hub.py
    name = repo.split("/", 1)[1]
    if name.endswith("-LiteRT"):
        return name[: -len("-LiteRT")] + "-OpenVINO"
    if name.lower().endswith("-litert"):
        return name[: -len("-litert")] + "-openvino"
    return name + "-OpenVINO"


def min_cos(p):
    return min((o["cos"] for o in (p or {}).get("outputs", []) if "cos" in o), default=None)


def check_text(r):
    if r is None:
        return None
    if r["status"] == "ERROR":
        return "did not run"
    if r["cell"] in ("cpu", "ovcpu") and r["status"] in ("PASS", "REFERENCE") and not r.get("parity"):
        return "reference"
    p = r.get("parity")
    if not p:
        return "not compared"
    c = min_cos(p)
    if p["status"] == "PASS":
        return f"PASS (cos ≥ {floor4(c):.4f})" if c is not None else "PASS"
    return "FAIL: " + "; ".join(f.replace("serving_default_", "") for f in p.get("fails", []))


def timing(r):
    """Quiet-pass median (ms) and load (s), or None when the cell has no quiet-pass number."""
    if r is None or r["status"] == "ERROR":
        return None
    t = r.get("table") or {}
    if t.get("ms") is None:
        return None
    q = r.get("quiet") or {}
    return {"ms": t["ms"], "load_s": (t["load_ms"] / 1000) if t.get("load_ms") is not None else None,
            "runs": len(q.get("ms_all") or []), "gate": t.get("gate"),
            "cpu_nodes": ((q.get("placement") or (r.get("first_pass") or {}).get("placement") or {}).get("nodes_by_provider") or {}).get("CPUExecutionProvider", 0)}


def graph_rows(m, hubx, with_hub):
    """The table rows of one graph, its notes, the days its quiet-pass rows were measured, and the Snapdragon row count."""
    R = {c: load(os.path.join(HERE, "results", m, c + ".json")) for c, *_ in ROWS}
    rows, notes, days, n_hub = [], set(), set(), 0
    for c, proc, rt, form in ROWS:
        r = R[c]
        if r is None:
            continue
        if c == "ovcpu" and R.get("cpu") and R["cpu"]["status"] != "ERROR":
            continue  # OpenVINO CPU is shown only where it is the reference (ONNX Runtime cannot load the file)
        t = timing(r)
        if t is not None and (r.get("quiet") or {}).get("started"):
            days.add(r["quiet"]["started"][:10])
        ms = "-" if t is None else f"{t['ms']:.2f}"
        if t and t["gate"] == "timeout":
            ms += " †"
            notes.add("† the machine did not go idle within 10 minutes before this run")
        if t and c == "ovnpu" and t["cpu_nodes"]:
            ms += f" ({t['cpu_nodes']} nodes on the CPU)"
        load_s = "-" if not t or t["load_s"] is None else f"{t['load_s']:.2f}"
        chk = check_text(r) or "-"
        if c == "ovcpu":
            chk = "reference (ONNX Runtime 1.31.1 cannot load this file)"
        rows.append(f"| {proc} | {rt} | {form} | {ms} | {load_s} | {chk} |")
    for dev, form, label, rt, f in (HUB_ROWS if with_hub else []):  # AI Hub measured the 52 files, not a _fixed one
        h = hubx.get(f"{m}|{dev}|{form}|npu")
        if not h or h.get("status") != "SUCCESS":
            continue
        cpu = (h.get("units") or {}).get("CPU", 0)
        ms = f"{h['infer_us'] / 1000:.2f}" + (f" ({cpu} layers on the CPU)" if cpu else "")
        load_s = f"{h['first_load_us'] / 1e6:.2f}" if h.get("first_load_us") else "-"
        rows.append(f"| {label} | {rt} | {f} | {ms} | {load_s} | not compared (AI Hub profile job, 2026-10-30) |")
        n_hub += 1
    return R, rows, notes, days, n_hub


def snippet_ov(m, has_blob):
    lines = ["```python", "import numpy as np", "import openvino as ov", "", "core = ov.Core()",
             'if "NPU" not in core.available_devices:',
             '    raise SystemExit(f"no Intel NPU visible to OpenVINO: {core.available_devices}")']
    if has_blob:
        lines += [f'with open("npu/{m}.blob", "rb") as f:',
                  '    compiled = core.import_model(f.read(), "NPU")  # precompiled: no compile at load',
                  f'# or compile the IR on this machine: compiled = core.compile_model("openvino/{m}.xml", "NPU")']
    else:
        lines += [f'compiled = core.compile_model("openvino/{m}.xml", "NPU")']
    lines += ["# replace the zeros with your preprocessed input (float32, the shapes printed by compiled.inputs)",
              "feeds = {i.get_any_name(): np.zeros(list(i.shape), np.float32) for i in compiled.inputs}",
              "outputs = compiled(feeds)",
              "print({o.get_any_name(): outputs[o].shape for o in compiled.outputs})", "```"]
    return "\n".join(lines)


def snippet_ort(m):
    return "\n".join([
        "```python", "import numpy as np", "import onnxruntime as ort", "import onnxruntime_ep_openvino as ov_ep", "",
        "ort.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())",
        "npu = [d for d in ort.get_ep_devices()",
        '       if d.ep_name == ov_ep.get_ep_name() and d.ep_metadata.get("ov_device") == "NPU"]',
        "if not npu:",
        '    raise SystemExit("no Intel NPU visible to the OpenVINO EP")',
        "so = ort.SessionOptions()", "so.add_provider_for_devices(npu, {})",
        f'sess = ort.InferenceSession("{m}.onnx", sess_options=so)  # compiles for the NPU here',
        "# replace the zeros with your preprocessed input (float32)",
        "feeds = {i.name: np.zeros(i.shape, np.float32) for i in sess.get_inputs()}",
        "outputs = sess.run(None, feeds)", "print([o.shape for o in outputs])", "```"])


def fmt_versions(bc):
    ov = (bc.get("openvino") or "").split("-")[0]
    return (f"OpenVINO {ov}, Intel NPU driver 32.0.100.4841 (OpenVINO reports NPU_DRIVER_VERSION "
            f"{bc.get('NPU_DRIVER_VERSION')}, NPU_COMPILER_VERSION {bc.get('NPU_COMPILER_VERSION')})")


def card(repo, info, hubx):
    folder = repo_folder(repo)
    out = os.path.join(HERE, "hub", folder)
    man = load(os.path.join(out, "manifest.json"))
    if man is None:
        return None, "no manifest.json (not staged)"
    models = info["models"]
    graphs = man.get("graphs", {})
    if sorted(graphs) != sorted(models):
        return None, f"staged graphs {sorted(graphs)} != {sorted(models)}"
    srcs = {m: (m + "_fixed" if m + "_fixed" in LISTED else m) for m in models}  # the file this repo ships
    tfl = {m: load(os.path.join(HERE, "results", srcs[m], "tflite.json")) for m in models}
    bad = [m for m in models if (tfl[m] or {}).get("status") != "PASS"]
    if bad:
        return None, "the ONNX file does not match its .tflite: " + ", ".join(f"{m} {(tfl[m] or {}).get('status', 'unchecked')}" for m in bad)
    name = folder.rsplit("-", 1)[0]
    rev = info["revision"]
    fm = ["---", f"license: {info['license']}", f"base_model: {repo}", "base_model_relation: quantized", "library_name: openvino"]
    if info.get("pipeline_tag"):
        fm.append(f"pipeline_tag: {info['pipeline_tag']}")
    fm += ["tags:", "- openvino", "- onnx", "- intel-npu", "---", ""]
    graphs_txt = models[0] if len(models) == 1 else " and ".join(models)
    blob_ms = []
    sections, notes_all, Rm, days, n_hub = [], set(), {}, set(), 0
    for m in models:
        R, rows, notes, d, h = graph_rows(srcs[m], hubx, with_hub=srcs[m] == m)
        Rm[m] = R
        notes_all |= notes
        days |= d
        n_hub += h
        t = timing(R.get("ovnpu_pre"))
        if t:
            blob_ms.append(f"{t['ms']:.2f} ms" + (f" ({m})" if len(models) > 1 else ""))
        tc = min_cos(tfl[m])
        sec = [f"## {m}" if len(models) > 1 else "## Speed and output check", ""]
        sec += ["| processor | runtime | file | ms | load s | output check |", "|---|---|---|---|---|---|"] + rows
        sec += ["", f"The ONNX file against the source `.tflite` (LiteRT interpreter on the CPU, the same input): "
                    f"PASS, cosine ≥ {floor4(tc):.4f} for every output."]
        sections.append("\n".join(sec))
    fixed = [m for m in models if srcs[m] != m]
    lead = (f"{name} for Intel NPUs, GPUs and CPUs: {graphs_txt} from "
            f"[{repo}](https://huggingface.co/{repo}/tree/{rev}) as ONNX (tf2onnx 1.17.0, opset 17), OpenVINO IR "
            "with fp32 weights, and " + ("precompiled Intel NPU blobs" if len(models) > 1 else "a precompiled Intel NPU blob") + ".")
    if fixed:
        recipe = load(os.path.join(HERE, "convert", "recipe.json")) or {}
        n = sum((recipe.get(m + "_fixed") or {}).get("fc_biases_restored", 0) for m in fixed)
        lead += (f" tf2onnx 1.17.0 drops the bias of {n} FULLY_CONNECTED layer{'s' if n != 1 else ''} of "
                 f"{' and '.join(fixed)}; these files were converted with a one-line fix of that handler "
                 "(`convert/patch_tf2onnx.py` in the GitHub repo), and their outputs match the `.tflite`.")
    if blob_ms:
        lead += (f" On the NPU of a Core Ultra 9 386H the blob runs in {' and '.join(blob_ms)} per run, numpy input to "
                 "numpy output.")
    body = [f"# {folder}", "", lead, "",
            f"Scripts, raw records, and the same table for all 52 graphs of LiteRT-Models: {GITHUB}", "",
            "## Files", "", "| file | bytes | sha256 |", "|---|---|---|"]
    for m in models:
        for rel, v in graphs[m]["files"].items():
            body.append(f"| `{rel}` | {v['bytes']:,} | `{v['sha256']}` |")
    bcs = {json.dumps(g.get("blob_compiled_with"), sort_keys=True) for g in graphs.values() if g.get("blob_compiled_with")}
    if len(bcs) == 1:
        body += ["", f"The NPU blob{'s were' if len(models) > 1 else ' was'} compiled with "
                     f"{fmt_versions(json.loads(next(iter(bcs))))}. If `import_model` refuses a blob on another driver "
                     "or OpenVINO version, compile the IR instead (the commented line in the snippet)."]
    days = sorted(days)
    when = "-" if not days else days[0] if len(days) == 1 else f"{days[0]} to {days[-1]}"
    ov_ref = [m for m in models if ((Rm[m].get("cpu") or {}).get("status") == "ERROR" and Rm[m].get("ovcpu"))]
    if not ov_ref:
        ref = "ONNX Runtime's CPU EP"
    elif ov_ref == models:
        ref = "OpenVINO on the CPU, as ONNX Runtime 1.31.1 cannot load this file"
    else:
        ref = f"ONNX Runtime's CPU EP; OpenVINO on the CPU for {', '.join(ov_ref)}, which ONNX Runtime 1.31.1 cannot load"
    how = (f"MSI Stealth 16 AI+ (Core Ultra 9 386H, RTX 5080 Laptop GPU at a 95 W power limit, Windows 11 build "
           f"26200.9711, on AC power), {when}. On the laptop, each "
           "row is one process: load, one run, 5 warmup runs, then the median of 30 timed runs (2 warmup + 10 timed "
           "runs when one run takes more than 3 s), numpy input to numpy output, every runtime at its default "
           "settings, one process at a time, each started after the CPU, the RTX and the Intel GPU had gone idle (the "
           f"thresholds are in METHOD.md of the GitHub repo). Output check: each output against the CPU output in "
           f"fp32 on the same input ({ref}); PASS = cosine ≥ 0.99, no NaN or Inf, and top-1 agreement ≥ 0.99 for "
           "class scores.")
    if n_hub:
        how += " The Snapdragon rows are Qualcomm AI Hub profile jobs of the same ONNX files (their own timer)."
    if fixed:
        how += ((" No Snapdragon rows: " if len(fixed) == len(models) else f" No Snapdragon rows for {' and '.join(fixed)}: ")
                + "Qualcomm AI Hub measured the file before the bias fix (its numbers are in the GitHub table).")
    body += ["", "## How it was measured", "", how, ""]
    body += sections
    if notes_all:
        body += [""] + sorted(notes_all)
    def ort_error(m):  # the last line of the error when the ONNX Runtime + OpenVINO EP NPU session did not load
        r = (Rm.get(m) or {}).get("ovnpu")
        if r is None or r["status"] != "ERROR":
            return None
        err = (r.get("first_pass") or {}).get("error") or ""
        lines = [l.strip() for l in err.strip().splitlines() if l.strip()]
        key = [l for l in lines if re.search(r"Failed Pass|Pass failed|ShapeInferenceError|Incompatible dimensions", l)]
        pick = key[0] if key else (lines[-1] if lines else "no error text recorded")
        pick = re.sub(r"^\[ERROR\] [0-9:.]+ ", "", pick)  # drop the log line's level and clock time
        return pick.replace("`", "'")[:160]

    ort_ok = [m for m in models if ort_error(m) is None]
    m0 = models[0]
    body += ["", "## Run it on the Intel NPU", "",
             "Download this repo, then run the snippets from its folder:", "",
             "```sh", "pip install huggingface_hub", f"hf download {HUB_ORG}/{folder} --local-dir {folder}", f"cd {folder}", "```", "",
             "OpenVINO (`pip install openvino==2026.4.1`):", "", snippet_ov(m0, f"npu/{m0}.blob" in graphs[m0]["files"]), ""]
    if m0 in ort_ok:
        body += ["ONNX Runtime with the OpenVINO EP (`pip install onnxruntime==1.31.1 onnxruntime-ep-openvino==1.8.1`):", "",
                 snippet_ort(m0), ""]
    if len(models) > 1:
        rest = models[1:]
        ort_rest = [m for m in rest if m in ort_ok] if m0 in ort_ok else []
        if ort_rest == rest:
            body += [f"The same lines run {', '.join(rest)} with its own file names.", ""]
        else:
            body += [f"The OpenVINO lines run {', '.join(rest)} with its own file names"
                     + (f"; the ONNX Runtime lines run {', '.join(ort_rest)}" if ort_rest else "") + ".", ""]
    for m in models:
        if m not in ort_ok:
            ov_runs = ((Rm.get(m) or {}).get("ovnpu_pre") or {}).get("status") not in (None, "ERROR")
            body += [f"ONNX Runtime 1.31.1 with the OpenVINO EP 1.8.1 does not run `{m}.onnx` on the NPU: the session "
                     f"fails to load (`{ort_error(m)}`)." + (" The OpenVINO lines above run it." if ov_runs else ""), ""]
    up = f" The upstream model: [{info['base_model']}](https://huggingface.co/{info['base_model']})." if info.get("base_model") else ""
    lic = {"mit": "MIT", "apache-2.0": "Apache-2.0", "bsd-3-clause": "BSD-3-Clause", "bsd-2-clause": "BSD-2-Clause"}.get(info["license"], info["license"])
    body += ["## Licence", "", f"{lic}, the licence of [{repo}](https://huggingface.co/{repo}).{up}", ""]
    return "\n".join(fm + body), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    src = json.load(open(os.path.join(HERE, "convert", "source_repos.json"), encoding="utf-8"))
    hubx = json.load(open(os.path.join(HERE, "aihub", "results_wide.json"), encoding="utf-8"))
    done, held = 0, []
    for repo, info in src.items():
        if repo.startswith("_") or (a.only and repo_folder(repo) not in a.only):
            continue
        text, why = card(repo, info, hubx)
        if text is None:
            held.append(f"{repo_folder(repo)}: {why}")
            continue
        with open(os.path.join(HERE, "hub", repo_folder(repo), "README.md"), "w", encoding="utf-8") as f:
            f.write(text)
        done += 1
    print(f"{done} cards written, {len(held)} held")
    for h in held:
        print("  held: " + h)


if __name__ == "__main__":
    main()
