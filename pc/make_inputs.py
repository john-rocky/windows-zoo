"""Build the fixed test input of each zoo model (inputs_spec.json) into <root>/inputs/<model>.npz; every cell reads it.

    python make_inputs.py [--root C:\\src\\zoo] [--spec inputs_spec.json] [--model m ...]

Images and the apps' host-side files come from the public zoo repo at a pinned commit, sam2_prompt.bin from its HF
repo at a pinned revision; they are cached in <root>/samples/. A model whose input is another graph's output
(DETR graph B, w2v2 head, DAC decoder, SAM2 decoder) runs that graph on the ONNX Runtime CPU EP first and builds
its input the way the zoo's Android app does (numpy ports of the Kotlin glue; paths in the spec).
Writes <root>/inputs/manifest.json (downloaded files with sha256, per-model input stats).
"""

import argparse
import hashlib
import json
import math
import os
import urllib.request

import numpy as np
from PIL import Image

ZOO_RAW = "https://raw.githubusercontent.com/john-rocky/LiteRT-Models/{commit}/{path}"
HF_RAW = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
SAMPLE_FILES = {  # zoo dir -> asset file name (the Android apps' bundled test inputs)
    "dfine": "test_image.jpg", "dinov2": "sample.jpg", "gaze": "test_image.jpg", "hsemotion": "sample.jpg",
    "lowlight": "test_image.jpg", "metric3d": "test_image.jpg", "migan": "test_image.jpg", "mlsd": "test_image.jpg",
    "nafnet": "test_image.jpg", "neuralstyle": "test_image.jpg", "nima": "test_image.jpg", "places365": "test_image.jpg",
    "ppocr": "test_image.png", "rtdetr": "test_image.jpg", "rtmanimal": "test_image.jpg", "rtmface": "test_image.jpg",
    "rtmhand": "test_image.jpg", "rtmpose": "test_image.jpg", "rtmw": "test_image.jpg", "saliency": "test_image.jpg",
    "tddfa": "test_image.jpg", "vrwkv": "sample.jpg", "yunet": "test_image.jpg",
}
NORM = {
    "imagenet": (1 / 255, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "unit": (1 / 255, [0, 0, 0], [1, 1, 1]),
    "minus1to1": (1 / 127.5, [1, 1, 1], [1, 1, 1]),
    "minus_half": (1 / 255, [0.5, 0.5, 0.5], [1, 1, 1]),
    "raw255": (1.0, [0, 0, 0], [1, 1, 1]),
    "mmpose255": (1.0, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]),
    "yolact_bgr": (1.0, [103.94, 116.78, 123.68], [57.38, 57.12, 58.40]),  # means / stds in B, G, R order
}


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


class Fetch:
    def __init__(self, root, commit):
        self.dir = os.path.join(root, "samples")
        os.makedirs(self.dir, exist_ok=True)
        self.commit = commit
        self.files = {}

    def get(self, ref):
        """ref = zoo path ("dfine/app/...") or "hf:<repo>@<rev>/<path>". Returns the local path (cached)."""
        if ref.startswith("hf:"):
            repo_rev, path = ref[3:].split("/", 2)[0:2], ref[3:].split("/", 2)[2]
            repo = repo_rev[0] + "/" + repo_rev[1].split("@")[0]
            rev = repo_rev[1].split("@")[1]
            url = HF_RAW.format(repo=repo, rev=rev, path=path)
        else:
            url = ZOO_RAW.format(commit=self.commit, path=ref)
        local = os.path.join(self.dir, ref.replace(":", "_").replace("/", "__").replace("@", "_"))
        if not os.path.exists(local):
            req = urllib.request.Request(url, headers={"User-Agent": "windows-zoo-inputs"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            with open(local + ".tmp", "wb") as f:
                f.write(data)
            os.replace(local + ".tmp", local)
        with open(local, "rb") as f:
            self.files[ref] = {"url": url, "bytes": os.path.getsize(local), "sha256": sha256_bytes(f.read())}
        return local


def load_image(fetch, sample):
    path = fetch.get(f"{sample}/app/src/main/assets/{SAMPLE_FILES[sample]}")
    return Image.open(path).convert("RGB")


def geometry(im, kind, W, H, inp):
    w, h = im.size
    if kind == "stretch":
        return np.asarray(im.resize((W, H), Image.BILINEAR), np.float32)
    if kind == "center_square":
        s = min(w, h)
        x, y = (w - s) // 2, (h - s) // 2
        return np.asarray(im.crop((x, y, x + s, y + s)).resize((W, H), Image.BILINEAR), np.float32)
    if kind == "center_3x4":  # crop to the model's aspect W:H, then resize
        tgt = W / H
        if w / h > tgt:
            cw = int(round(h * tgt)); x = (w - cw) // 2
            box = (x, 0, x + cw, h)
        else:
            ch = int(round(w / tgt)); y = (h - ch) // 2
            box = (0, y, w, y + ch)
        return np.asarray(im.crop(box).resize((W, H), Image.BILINEAR), np.float32)
    if kind in ("letterbox_white", "letterbox_black"):
        s = min(W / w, H / h)
        nw, nh = max(1, int(w * s)), max(1, int(h * s))
        canvas = Image.new("RGB", (W, H), (255, 255, 255) if kind.endswith("white") else (0, 0, 0))
        canvas.paste(im.resize((nw, nh), Image.BILINEAR), ((W - nw) // 2, (H - nh) // 2))
        return np.asarray(canvas, np.float32)
    if kind == "short256_center224":
        s = 256 / min(w, h)
        im2 = im.resize((max(1, int(round(w * s))), max(1, int(round(h * s)))), Image.BILINEAR)
        w2, h2 = im2.size
        x, y = (w2 - W) // 2, (h2 - H) // 2
        return np.asarray(im2.crop((x, y, x + W, y + H)), np.float32)
    if kind == "line_crop48":
        x0, y0, x1, y1 = inp["crop"]
        crop = im.crop((x0, y0, x1, y1))
        bw, bh = crop.size
        nw = max(1, min(int(H * bw / bh), W))
        out = np.zeros((H, W, 3), np.float32)  # raw 0 beyond the text, as the app
        out[:, :nw] = np.asarray(crop.resize((nw, H), Image.BILINEAR), np.float32)
        return out
    raise ValueError(kind)


def normalize(x, norm, order):
    if order == "BGR":
        x = x[..., ::-1]
    scale, mean, std = NORM[norm]
    return ((x * scale - np.array(mean, np.float32)) / np.array(std, np.float32)).astype(np.float32)


def chw(x):
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]).astype(np.float32)


def onnx_inputs(root, model):
    import onnxruntime as ort  # noqa: PLC0415
    s = ort.InferenceSession(os.path.join(root, "onnx", model + ".onnx"), providers=["CPUExecutionProvider"])
    return s, [(i.name, i.shape) for i in s.get_inputs()]


def graph_inputs(root, model):
    """Input names and shapes read from the ONNX graph, not from an ONNX Runtime session: ONNX Runtime 1.31.1 refuses
    to load yolact.onnx (shape inference fails on an FPN Add), and the OpenVINO cells still need its input."""
    import onnx  # noqa: PLC0415
    g = onnx.load(os.path.join(root, "onnx", model + ".onnx")).graph
    init = {t.name for t in g.initializer}
    return [(i.name, [d.dim_value if d.HasField("dim_value") else d.dim_param for d in i.type.tensor_type.shape.dim])
            for i in g.input if i.name not in init]


def by_shape(arrays, shape):
    hits = [a for a in arrays if list(a.shape) == list(shape)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} arrays of shape {shape}: {[a.shape for a in arrays]}")
    return hits[0]


def run_cpu(root, model, feeds):
    s, _ = onnx_inputs(root, model)
    return s.run(None, feeds)


def detr_b(fetch, root, inp, built):
    """D-FINE / RT-DETRv2 graph B inputs from graph A on the host, as DFine.kt / RtDetr.kt (fp32)."""
    HID, NPROP, NCLS, NQ = 256, 8400, 80, 300
    outs = run_cpu(root, inp["upstream"], built[inp["upstream"]])
    enc_class = by_shape(outs, [1, NPROP, NCLS])[0]
    mem = by_shape(outs, [1, NPROP, HID])[0] * np.float32(0.5)  # graph A emits memory_raw * 2
    raw = np.fromfile(fetch.get(inp["host_params"]), dtype="<f4")
    o = 0

    def take(n):
        nonlocal o
        a = raw[o:o + n]; o += n
        return a
    eoW = take(HID * HID).reshape(HID, HID); eoB = take(HID); eoG = take(HID); eoBeta = take(HID)
    bb0W = take(HID * HID).reshape(HID, HID); bb0B = take(HID)
    bb1W = take(HID * HID).reshape(HID, HID); bb1B = take(HID)
    bb2W = take(4 * HID).reshape(4, HID); bb2B = take(4)
    valid = take(NPROP); anchors = take(NPROP * 4).reshape(NPROP, 4)
    assert o == raw.size, (o, raw.size)
    order = np.argsort(-enc_class.max(axis=1), kind="stable")[:NQ]
    x = valid[order, None] * mem[order]
    h = x @ eoW.T + eoB
    mu = h.mean(axis=1, keepdims=True); var = ((h - mu) ** 2).mean(axis=1, keepdims=True)
    target = ((h - mu) / np.sqrt(var + np.float32(1e-5)) * eoG + eoBeta).astype(np.float32)
    h1 = np.maximum(target @ bb0W.T + bb0B, 0)
    h2 = np.maximum(h1 @ bb1W.T + bb1B, 0)
    ref = (h2 @ bb2W.T + bb2B + anchors[order]).astype(np.float32)
    return [mem[None].astype(np.float32), target[None], ref[None]]


def rfdetr_b(root, inp, built):
    NPROP, NCLS, NQ = 576, 91, 300
    outs = run_cpu(root, inp["upstream"], built[inp["upstream"]])
    enc_class = by_shape(outs, [1, NPROP, NCLS])[0]
    coord = by_shape(outs, [1, NPROP, 4])[0]
    memory = by_shape(outs, [1, NPROP, 256])
    order = np.argsort(-enc_class.max(axis=1), kind="stable")[:NQ]
    return [memory.astype(np.float32), coord[order][None].astype(np.float32)]


def dac_zq(fetch, root, inp, built):
    """z_q = RVQ.decode(RVQ.encode(encoder output)), as DacRVQ.kt (12 codebooks, dim 8, size 1024, hidden 1024)."""
    NQ, DIM, SIZE, HID = 12, 8, 1024, 1024
    z = run_cpu(root, inp["upstream"], built[inp["upstream"]])[0][0].astype(np.float32)  # [1024, T]
    raw = np.fromfile(fetch.get(inp["rvq"]), dtype="<f4")
    per = SIZE * DIM + DIM * HID + DIM + HID * DIM + HID
    assert raw.size == NQ * per, raw.size
    residual = z.copy()
    zq = np.zeros_like(z)
    for i in range(NQ):
        b = raw[i * per:(i + 1) * per]
        o = 0
        cb = b[o:o + SIZE * DIM].reshape(SIZE, DIM); o += SIZE * DIM
        wIn = b[o:o + DIM * HID].reshape(DIM, HID); o += DIM * HID
        bIn = b[o:o + DIM]; o += DIM
        wOut = b[o:o + HID * DIM].reshape(HID, DIM); o += HID * DIM
        bOut = b[o:o + HID]
        cbn = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
        proj = wIn @ residual + bIn[:, None]
        proj = proj / np.maximum(np.linalg.norm(proj, axis=0, keepdims=True), 1e-12)
        codes = np.argmax(cbn @ proj, axis=0)
        q = wOut @ cb[codes].T + bOut[:, None]
        residual -= q
        zq += q
    return [zq[None].astype(np.float32)]


def sam2_dec(fetch, root, inp, built):
    IE, F0, F1 = 256 * 64 * 64, 32 * 256 * 256, 64 * 128 * 128
    flat = run_cpu(root, inp["upstream"], built[inp["upstream"]])[0].reshape(-1).astype(np.float32)
    assert flat.size == IE + F0 + F1, flat.size
    p = np.fromfile(fetch.get(inp["prompt_bin"]), dtype="<f4")
    assert p.size == 768, p.size
    g, pe1, nap = p[0:256], p[256:512], p[512:768]
    x, y = inp["point"]
    ccx = 2 * ((x + 0.5) / 1024) - 1
    ccy = 2 * ((y + 0.5) / 1024) - 1
    proj = (2 * math.pi * (ccx * g[:128].astype(np.float64) + ccy * g[128:].astype(np.float64))).astype(np.float32)
    sparse = np.concatenate([np.sin(proj) + pe1[:128], np.cos(proj) + pe1[128:], nap]).astype(np.float32)
    return [np.concatenate([flat[:IE], sparse, flat[IE:IE + F0], flat[IE + F0:]])[None]]


def build(fetch, root, model, spec, built):
    ins = graph_inputs(root, model)
    sp = spec[model]
    kinds = [i["kind"] for i in sp["inputs"]]
    feeds = {}
    if kinds[0] in ("detr_b", "rfdetr_b", "dac_zq", "sam2_dec", "upstream"):
        inp = sp["inputs"][0]
        if inp["kind"] == "detr_b":
            arrays = detr_b(fetch, root, inp, built)
        elif inp["kind"] == "rfdetr_b":
            arrays = rfdetr_b(root, inp, built)
        elif inp["kind"] == "dac_zq":
            arrays = dac_zq(fetch, root, inp, built)
        elif inp["kind"] == "sam2_dec":
            arrays = sam2_dec(fetch, root, inp, built)
        else:
            arrays = [run_cpu(root, inp["upstream"], built[inp["upstream"]])[inp["output"]].astype(np.float32)]
        for name, shape in ins:
            feeds[name] = by_shape(arrays, shape)
        return feeds
    for k, (name, shape) in enumerate(ins):
        suffix = name.rsplit("_", 1)[-1] if not name.endswith("_output") else ""
        idx = int(suffix) if suffix.isdigit() else k  # tf2onnx names the inputs serving_default_args_<i>
        inp = next((i for i in sp["inputs"] if i["index"] == idx), None)
        if inp is None:  # movinet: the state inputs, given as index ranges in the spec
            kind = "zeros" if 1 <= idx <= 44 else "ones"
            feeds[name] = (np.zeros if kind == "zeros" else np.ones)(shape, np.float32)
            continue
        _, C, H, W = shape if len(shape) == 4 else (None, None, None, None)
        if inp["kind"] == "image":
            x = geometry(load_image(fetch, sp["sample"]), inp["geometry"], W, H, inp)
            feeds[name] = chw(normalize(x, inp["norm"], inp["order"]))
        elif inp["kind"] == "mlsd4":
            x = geometry(load_image(fetch, sp["sample"]), inp["geometry"], W, H, inp)
            rgb = x / np.float32(127.5) - 1
            ones = np.full(x.shape[:2] + (1,), 1 / 127.5 - 1, np.float32)
            feeds[name] = chw(np.concatenate([rgb, ones], axis=2))
        elif inp["kind"] == "migan4":
            x = geometry(load_image(fetch, sp["sample"]), inp["geometry"], W, H, inp)
            mask = np.ones(x.shape[:2], np.float32)
            s = H // 4
            mask[(H - s) // 2:(H + s) // 2, (W - s) // 2:(W + s) // 2] = 0
            rgb = (x / np.float32(127.5) - 1) * mask[..., None]
            feeds[name] = chw(np.concatenate([(mask - 0.5)[..., None], rgb], axis=2))
        elif inp["kind"] == "vrwkv_dist":
            t = np.arange(shape[-1], dtype=np.float32)
            feeds[name] = np.abs(t[:, None] - t[None, :]).reshape(shape).astype(np.float32)
        elif inp["kind"] == "audio_bin":
            a = np.fromfile(fetch.get(sp["sample"]), dtype="<f4")
            n = int(np.prod(shape))
            a = np.pad(a[:n], (0, max(0, n - a.size)))
            feeds[name] = a.reshape(shape).astype(np.float32)
        elif inp["kind"] in ("zeros", "ones"):
            feeds[name] = (np.zeros if inp["kind"] == "zeros" else np.ones)(shape, np.float32)
        else:
            raise ValueError(f"{model}: input kind {inp['kind']}")
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--spec", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs_spec.json"))
    ap.add_argument("--model", nargs="*")
    a = ap.parse_args()
    spec = json.load(open(a.spec, encoding="utf-8"))
    fetch = Fetch(a.root, spec["_about"]["zoo_commit"])
    models = [m for m in spec if not m.startswith("_")]
    if a.model:
        need = set(a.model)
        for m in a.model:  # upstreams first
            for i in spec[m]["inputs"]:
                if i.get("upstream"):
                    need.add(i["upstream"])
        models = [m for m in models if m in need]
    ups = [m for m in models if not any(i.get("upstream") for i in spec[m]["inputs"])]
    downs = [m for m in models if m not in ups]
    os.makedirs(os.path.join(a.root, "inputs"), exist_ok=True)
    built, stats = {}, {}
    for m in ups + downs:
        feeds = build(fetch, a.root, m, spec, built)
        built[m] = feeds
        path = os.path.join(a.root, "inputs", m + ".npz")
        np.savez(path, **feeds)
        with open(path, "rb") as f:
            digest = sha256_bytes(f.read())
        stats[m] = {"sha256": digest, "inputs": {k: {"shape": list(v.shape), "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())} for k, v in feeds.items()}}
        first = next(iter(feeds.values()))
        print(f"{m}: {len(feeds)} input(s), first {list(first.shape)} [{first.min():.3f}, {first.max():.3f}]", flush=True)
    man_path = os.path.join(a.root, "inputs", "manifest.json")
    old = json.load(open(man_path)) if os.path.exists(man_path) else {"files": {}, "models": {}}
    old["files"].update(fetch.files)
    old["models"].update(stats)
    old["zoo_commit"] = spec["_about"]["zoo_commit"]
    json.dump(old, open(man_path, "w"), indent=1)


if __name__ == "__main__":
    main()
