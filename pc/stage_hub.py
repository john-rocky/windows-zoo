"""Stage one model's Hugging Face repo files under <root>/hub/<model>/ (run in .venv-ov, after the first pass).

    python stage_hub.py --model m [--root <root>]

<root>/hub/<model>/
  <model>.onnx                 the fp32 ONNX file the table was measured on (a hard link to onnx/<model>.onnx)
  openvino/<model>.xml, .bin   OpenVINO IR: read_model(onnx) -> save_model(compress_to_fp16=False), weights kept fp32
  npu/<model>.blob             the NPU blob of the ovnpu_pre cell (compile_model(NPU) -> export_model), when it exists
  manifest.json                sha256 and bytes of every file, the versions the blob was compiled with, the source
The card (README.md) is written later from the results (cards.py), next to these files.
"""

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def link_or_copy(src, dst):
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    import openvino as ov  # noqa: PLC0415

    m, root = a.model, a.root
    out = os.path.join(root, "hub", m)
    os.makedirs(os.path.join(out, "openvino"), exist_ok=True)
    rec = {"model": m, "staged": datetime.now(timezone.utc).astimezone().isoformat(), "openvino": ov.__version__, "files": {}}
    onnx = os.path.join(root, "onnx", m + ".onnx")
    rec["onnx_from"] = link_or_copy(onnx, os.path.join(out, m + ".onnx"))
    core = ov.Core()
    model = core.read_model(onnx)
    ov.save_model(model, os.path.join(out, "openvino", m + ".xml"), compress_to_fp16=False)
    blob = os.path.join(root, "blobs", m + ".blob")
    comp = os.path.join(root, "runs", "first", m, "ovnpu_pre_compile.json")
    if os.path.exists(blob):
        os.makedirs(os.path.join(out, "npu"), exist_ok=True)
        rec["blob_from"] = link_or_copy(blob, os.path.join(out, "npu", m + ".blob"))
        if os.path.exists(comp):
            with open(comp, encoding="utf-8-sig") as f:
                c = json.load(f)
            rec["blob_compiled_with"] = {"openvino": (c.get("versions") or {}).get("openvino"),
                                         **{k: (c.get("device") or {}).get(k) for k in
                                            ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION",
                                             "NPU_COMPILER_TYPE", "INFERENCE_PRECISION_HINT")},
                                         "compile_ms": c.get("compile_ms"), "ov_config": c.get("ov_config")}
    for dirpath, _, files in os.walk(out):
        for fn in files:
            if fn == "manifest.json":
                continue
            p = os.path.join(dirpath, fn)
            rel = os.path.relpath(p, out).replace(os.sep, "/")
            rec["files"][rel] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(f"{m}: {len(rec['files'])} files, blob={'npu/' + m + '.blob' in rec['files']}", flush=True)


if __name__ == "__main__":
    main()
