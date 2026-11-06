"""Stage one graph's files into its Hugging Face repo folder (run in .venv-ov, after the first pass).

    python stage_hub.py --model m [--root <root>]

One repo per source model (convert/source_repos.json: 45 repos; a two-graph model keeps both graphs in one repo, as
its LiteRT repo does). Folder name = the source repo name with "-LiteRT" replaced by "-OpenVINO".

<root>/hub/<repo>/
  <model>.onnx                 the fp32 ONNX file the table was measured on (a hard link to onnx/<model>.onnx)
  openvino/<model>.xml, .bin   OpenVINO IR: read_model(onnx) -> save_model(compress_to_fp16=False), weights kept fp32
  npu/<model>.blob             the NPU blob of the ovnpu_pre cell (compile_model(NPU) -> export_model), when it exists
  manifest.json                per graph: bytes and sha256 of its files, the versions the blob was compiled with
The card (README.md) is written later from the results (cards.py), next to these files.

A model with a <model>_fixed file in pc/models.txt (the FULLY_CONNECTED bias fix of convert/patch_tf2onnx.py) is staged
from that file, under the base name; the base model itself is skipped (its ONNX file drops biases).
"""

import argparse
import hashlib
import json
import os
import shutil


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
    except OSError:
        shutil.copy2(src, dst)


def repo_folder(repo):
    name = repo.split("/", 1)[1]
    if name.endswith("-LiteRT"):
        return name[: -len("-LiteRT")] + "-OpenVINO"
    if name.lower().endswith("-litert"):
        return name[: -len("-litert")] + "-openvino"
    return name + "-OpenVINO"


def source_of(root, m):
    with open(os.path.join(root, "convert", "source_repos.json"), encoding="utf-8") as f:
        src = json.load(f)
    for repo, info in src.items():
        if not repo.startswith("_") and m in info["models"]:
            return repo, info
    raise SystemExit(f"{m}: not in convert/source_repos.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    a = ap.parse_args()
    import openvino as ov  # noqa: PLC0415

    src_m, root = a.model, a.root
    listed = [l.strip() for l in open(os.path.join(root, "pc", "models.txt")) if l.strip()]
    if src_m + "_fixed" in listed:
        print(f"{src_m}: skipped (staged from {src_m}_fixed)", flush=True)
        return
    m = src_m[: -len("_fixed")] if src_m.endswith("_fixed") else src_m  # the name inside the repo
    repo, info = source_of(root, m)
    out = os.path.join(root, "hub", repo_folder(repo))
    os.makedirs(os.path.join(out, "openvino"), exist_ok=True)
    onnx = os.path.join(root, "onnx", src_m + ".onnx")
    link_or_copy(onnx, os.path.join(out, m + ".onnx"))
    ov.save_model(ov.Core().read_model(onnx), os.path.join(out, "openvino", m + ".xml"), compress_to_fp16=False)
    rels = [m + ".onnx", f"openvino/{m}.xml", f"openvino/{m}.bin"]
    g = {"from": src_m, "ir": {"openvino": ov.__version__, "compress_to_fp16": False}}
    blob = os.path.join(root, "blobs", src_m + ".blob")
    comp = os.path.join(root, "runs", "first", src_m, "ovnpu_pre_compile.json")
    if os.path.exists(blob):
        os.makedirs(os.path.join(out, "npu"), exist_ok=True)
        link_or_copy(blob, os.path.join(out, "npu", m + ".blob"))
        rels.append(f"npu/{m}.blob")
        if os.path.exists(comp):
            with open(comp, encoding="utf-8-sig") as f:
                c = json.load(f)
            g["blob_compiled_with"] = {"openvino": (c.get("versions") or {}).get("openvino"),
                                       **{k: (c.get("device") or {}).get(k) for k in
                                          ("FULL_DEVICE_NAME", "NPU_DRIVER_VERSION", "NPU_COMPILER_VERSION",
                                           "NPU_COMPILER_TYPE", "INFERENCE_PRECISION_HINT")}}
    g["files"] = {r: {"bytes": os.path.getsize(os.path.join(out, r)), "sha256": sha256(os.path.join(out, r))} for r in rels}
    mp = os.path.join(out, "manifest.json")
    man = {"source_repo": repo, "source_revision": info["revision"], "graphs": {}}
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            man["graphs"] = json.load(f).get("graphs", {})
    man["graphs"][m] = g
    man["graphs"] = dict(sorted(man["graphs"].items()))
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1)
    print(f"{m}: {repo_folder(repo)}, {len(rels)} files, blob={'npu/' + m + '.blob' in rels}", flush=True)


if __name__ == "__main__":
    main()
