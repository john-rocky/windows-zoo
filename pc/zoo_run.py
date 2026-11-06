"""Run one zoo model on one runtime x device ("cell") and write one JSON: load time, run times, node placement, outputs.

    python zoo_run.py --model places_fp16 --cell ovnpu --pass first [--root C:\\src\\zoo] [--profile] [--save-outputs]

Cells (each needs the venv that has its runtime):
    cpu        ONNX Runtime CPU EP, fp32 ONNX (the reference)                       .venv-ort
    ovgpu      ONNX Runtime + OpenVINO EP plugin, Intel iGPU (ov_device GPU.0)      .venv-ort
    ovnpu      ONNX Runtime + OpenVINO EP plugin, NPU, ONNX as is (compile at load) .venv-ort
    cuda32     ONNX Runtime CUDA EP, fp32 ONNX                                      .venv-cuda
    cuda16     ONNX Runtime CUDA EP, fp16 ONNX (converted here on first use)       .venv-cuda
    ovnpu_pre  OpenVINO without ONNX Runtime, NPU: --phase compile writes the blob (compile_model + export_model),
               --phase run imports it (import_model) in a new process and runs it  .venv-ov

Inputs come from <root>/inputs/<model>.npz (make_inputs.py); the same file for every cell.
Timing: time.perf_counter() around one run with host numpy in and out (host<->device copies included), first run
apart, then warmup 5 + 30 runs (warmup 2 + 10 when the first run takes more than 3 s). The JSON is written on
failure too, with the full error text.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np

ORT_CELLS = {"cpu", "ovgpu", "ovnpu", "cuda32", "cuda16"}
OV_DEVICE = {"ovgpu": "GPU.0", "ovnpu": "NPU"}


def luid_key(device_luid):
    """OpenVINO's DEVICE_LUID (8 bytes, little-endian hex) -> the 'luid_0xHIGH_0xLOW' form of the Windows counters."""
    try:
        b = bytes.fromhex(device_luid)
        return f"0x{int.from_bytes(b[4:8], 'little'):08x}_0x{int.from_bytes(b[0:4], 'little'):08x}"
    except Exception:  # noqa: BLE001
        return None


def engine_time_ms(pid):
    """Cumulative GPU Engine 'Running Time' of this process per adapter and engine type (ms), from the Windows counters."""
    import re  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    cmd = (f"(Get-Counter '\\GPU Engine(pid_{pid}_*)\\Running Time' -ErrorAction SilentlyContinue).CounterSamples | "
           "ForEach-Object { $_.InstanceName + ' ' + $_.RawValue }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=60)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}
    out = {}
    for line in r.stdout.splitlines():
        m = re.match(r"pid_\d+_luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_\d+_eng_\d+_engtype_(\S+) (\d+)", line.strip())
        if not m:
            continue
        raw = int(m.group(3))
        if raw >= 1 << 63:
            continue
        key = f"{m.group(1)}/{m.group(2)}"  # adapter LUID / engine type (the LUIDs change with a reboot)
        out[key] = out.get(key, 0.0) + raw / 1e4  # 100 ns -> ms
    return {k: round(v, 3) for k, v in out.items() if v > 0}


def now():
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stats(times_ms):
    s = sorted(times_ms)
    n = len(s)
    return {"median": statistics.median(s), "p10": s[int(round(0.1 * (n - 1)))], "p90": s[int(round(0.9 * (n - 1)))],
            "min": s[0], "max": s[-1], "n": n}


def fingerprint(o):
    """Mean, std and 16 values at fixed positions of one output: enough to tell, without the array, whether another run
    (another machine, a rebuilt ONNX file) computed the same output (check_cpu.py)."""
    flat = np.ascontiguousarray(o).reshape(-1)
    s = s2 = 0.0
    for i in range(0, flat.size, 1 << 22):  # float64 sums in chunks: no float64 copy of a large output
        c = flat[i:i + (1 << 22)].astype(np.float64)
        s += float(c.sum())
        s2 += float((c * c).sum())
    n = max(flat.size, 1)
    mean = s / n
    pos = sorted({(k * 2654435761) % flat.size for k in range(1, 17)}) if flat.size else []
    return {"mean": mean, "std": max(s2 / n - mean * mean, 0.0) ** 0.5, "sample_pos": pos,
            "sample": [float(flat[p]) for p in pos]}


def bench(run_once, rec):
    """First run (its outputs are the ones saved), then warmup and timed runs. Returns the first outputs."""
    t = time.perf_counter()
    first = run_once()
    rec["first_run_ms"] = (time.perf_counter() - t) * 1000
    warm, n = (2, 10) if rec["first_run_ms"] > 3000 else (5, 30)
    rec["warmup"] = warm
    for _ in range(warm):
        run_once()
    times = []
    last = None
    for _ in range(n):
        t = time.perf_counter()
        last = run_once()
        times.append((time.perf_counter() - t) * 1000)
    rec["ms"] = stats(times)
    rec["ms_all"] = times
    rec["first_vs_last_max_abs"] = max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
                                       for a, b in zip(first, last))
    return first


def placement_from_profile(path):
    """Count graph nodes per execution provider from an ONNX Runtime profile (one run)."""
    with open(path, encoding="utf-8") as f:
        events = json.load(f)
    nodes = {}
    for e in events:
        if e.get("cat") != "Node":
            continue
        args = e.get("args", {})
        prov = args.get("provider")
        name = e.get("name", "")
        if not prov or not name.endswith("_kernel_time"):
            continue
        nodes.setdefault(prov, set()).add(name[: -len("_kernel_time")])
    out = {p: len(v) for p, v in nodes.items()}
    cpu_ops = {}
    for e in events:
        if e.get("cat") == "Node" and e.get("args", {}).get("provider") == "CPUExecutionProvider" and e.get("name", "").endswith("_kernel_time"):
            op = e["args"].get("op_name", "?")
            cpu_ops[op] = cpu_ops.get(op, 0) + 1
    return {"nodes_by_provider": out, "cpu_ops": cpu_ops}


def onnx_graph_nodes(path):
    import onnx  # noqa: PLC0415
    m = onnx.load(path, load_external_data=False)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    return len(m.graph.node), ops


def fp16_model(root, model, src, rec):
    dst = os.path.join(root, "fp16", model + ".onnx")
    if os.path.exists(dst):
        rec["fp16_convert"] = {"cached": True, "path": dst}
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    import onnx  # noqa: PLC0415
    from onnxruntime.transformers.float16 import convert_float_to_float16  # noqa: PLC0415
    t = time.perf_counter()
    m = onnx.load(src)
    m16 = convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, dst + ".tmp")
    os.replace(dst + ".tmp", dst)
    rec["fp16_convert"] = {"cached": False, "path": dst, "seconds": time.perf_counter() - t, "bytes": os.path.getsize(dst),
                           "tool": "onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)"}
    return dst


def run_ort(args, rec, feeds):
    import onnxruntime as ort  # noqa: PLC0415
    rec["versions"]["onnxruntime"] = ort.__version__
    path = os.path.join(args.root, "onnx", args.model + ".onnx")
    so = ort.SessionOptions()
    prof_prefix = None
    if args.profile:
        os.makedirs(os.path.join(args.root, "prof"), exist_ok=True)
        prof_prefix = os.path.join(args.root, "prof", f"{args.model}_{args.cell}_{args.pass_}")
        so.enable_profiling = True
        so.profile_file_prefix = prof_prefix
    providers = None
    if args.cell == "cpu":
        providers = ["CPUExecutionProvider"]
    elif args.cell in ("cuda32", "cuda16"):
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()  # the CUDA / cuDNN DLLs of the nvidia pip wheels
        rec["versions"]["cuda_wheels"] = cuda_wheels()
        if args.cell == "cuda16":
            path = fp16_model(args.root, args.model, path, rec)
        providers = [("CUDAExecutionProvider", {}), "CPUExecutionProvider"]
    else:
        import onnxruntime_ep_openvino as ov_ep  # noqa: PLC0415
        rec["versions"]["onnxruntime_ep_openvino"] = getattr(ov_ep, "__version__", None)
        ort.register_execution_provider_library(ov_ep.get_ep_name(), ov_ep.get_library_path())
        want = OV_DEVICE[args.cell]
        devices = [d for d in ort.get_ep_devices() if d.ep_name == ov_ep.get_ep_name() and d.ep_metadata.get("ov_device") == want]
        if not devices:
            raise RuntimeError(f"no OpenVINO EP device {want}: {[d.ep_metadata.get('ov_device') for d in ort.get_ep_devices()]}")
        meta = dict(devices[0].ep_metadata)
        rec["device"] = {k: meta.get(k) for k in ("FULL_DEVICE_NAME", "ov_device", "version", "os_driver_version", "NPU_COMPILER_VERSION", "DEVICE_ARCHITECTURE", "DEVICE_LUID")}
        rec["device"]["counter_luid"] = luid_key(meta.get("DEVICE_LUID") or "")
        rec["versions"]["openvino_ep_plugin"] = meta.get("version")
        so.add_provider_for_devices(devices, dict(args.ep_option))
        rec["ep_options"] = dict(args.ep_option)
    rec["model_path"] = path
    rec["model_sha256"] = sha256(path)
    t = time.perf_counter()
    sess = ort.InferenceSession(path, sess_options=so, providers=providers) if providers else ort.InferenceSession(path, sess_options=so)
    rec["load_ms"] = (time.perf_counter() - t) * 1000
    rec["providers"] = sess.get_providers()
    try:
        rec["provider_options"] = {k: dict(v) for k, v in sess.get_provider_options().items()}
    except Exception as e:  # noqa: BLE001
        rec["provider_options"] = f"<{type(e).__name__}: {e}>"
    if args.cell in ("cuda32", "cuda16") and "CUDAExecutionProvider" not in rec["providers"]:
        raise RuntimeError(f"CUDA EP not in the session providers: {rec['providers']}")
    names = [o.name for o in sess.get_outputs()]
    rec["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in sess.get_outputs()]
    in_names = [i.name for i in sess.get_inputs()]
    feed = {k: feeds[k] for k in in_names}

    def once():
        return sess.run(None, feed)

    if args.profile:
        t = time.perf_counter()
        first = once()
        rec["profiled_run_ms"] = (time.perf_counter() - t) * 1000
        prof = sess.end_profiling()
        rec["placement"] = placement_from_profile(prof)
        rec["placement"]["profile_bytes"] = os.path.getsize(prof)
        os.remove(prof)
        total, ops = onnx_graph_nodes(path)
        rec["placement"]["graph_nodes"] = total
    out = bench(once, rec)
    return names, out


def cuda_wheels():
    try:
        from importlib import metadata  # noqa: PLC0415
        return {d.metadata["Name"]: d.version for d in metadata.distributions() if d.metadata["Name"] and d.metadata["Name"].lower().startswith("nvidia")}
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def run_ov(args, rec, feeds):
    import openvino as ov  # noqa: PLC0415
    rec["versions"]["openvino"] = ov.__version__
    core = ov.Core()
    rec["device"] = {"FULL_DEVICE_NAME": core.get_property("NPU", "FULL_DEVICE_NAME"),
                     "DEVICE_LUID": str(core.get_property("NPU", "DEVICE_LUID")),
                     "counter_luid": luid_key(str(core.get_property("NPU", "DEVICE_LUID"))),
                     "NPU_DRIVER_VERSION": str(core.get_property("NPU", "NPU_DRIVER_VERSION")),
                     "NPU_COMPILER_VERSION": str(core.get_property("NPU", "NPU_COMPILER_VERSION")),
                     "NPU_COMPILER_TYPE": str(core.get_property("NPU", "NPU_COMPILER_TYPE")),
                     "INFERENCE_PRECISION_HINT": str(core.get_property("NPU", "INFERENCE_PRECISION_HINT"))}
    onnx_path = os.path.join(args.root, "onnx", args.model + ".onnx")
    blob = os.path.join(args.root, "blobs", args.model + ".blob")
    config = dict(args.ep_option)
    rec["ov_config"] = config
    if args.phase == "compile":
        rec["model_path"] = onnx_path
        rec["model_sha256"] = sha256(onnx_path)
        t = time.perf_counter()
        model = core.read_model(onnx_path)
        rec["read_ms"] = (time.perf_counter() - t) * 1000
        ops = [op for op in model.get_ordered_ops() if op.get_type_name() not in ("Parameter", "Result", "Constant")]
        t = time.perf_counter()
        supported = core.query_model(model, "NPU", config)
        rec["query_ms"] = (time.perf_counter() - t) * 1000
        unsupported = {}
        for op in ops:
            if op.get_friendly_name() not in supported:
                unsupported[op.get_type_name()] = unsupported.get(op.get_type_name(), 0) + 1
        rec["query_model"] = {"ops": len(ops), "supported": sum(1 for op in ops if op.get_friendly_name() in supported),
                              "unsupported_by_type": unsupported}
        t = time.perf_counter()
        compiled = core.compile_model(model, "NPU", config)
        rec["compile_ms"] = (time.perf_counter() - t) * 1000
        rec["execution_devices"] = str(compiled.get_property("EXECUTION_DEVICES"))
        t = time.perf_counter()
        data = compiled.export_model()
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        with open(blob + ".tmp", "wb") as f:
            f.write(data)
        os.replace(blob + ".tmp", blob)
        rec["export_ms"] = (time.perf_counter() - t) * 1000
        rec["blob"] = {"path": blob, "bytes": os.path.getsize(blob)}
        return None, None
    # phase run: import the blob in this (new) process
    rec["blob"] = {"path": blob, "bytes": os.path.getsize(blob), "sha256": sha256(blob)}
    t = time.perf_counter()
    with open(blob, "rb") as f:
        data = f.read()
    compiled = core.import_model(data, "NPU", config)
    rec["load_ms"] = (time.perf_counter() - t) * 1000
    rec["execution_devices"] = str(compiled.get_property("EXECUTION_DEVICES"))
    request = compiled.create_infer_request()
    names = [o.get_any_name() for o in compiled.outputs]
    rec["outputs"] = [{"name": o.get_any_name(), "shape": list(o.get_shape()), "type": str(o.get_element_type())} for o in compiled.outputs]
    in_names = [i.get_any_name() for i in compiled.inputs]
    feed = {k: feeds[k] for k in in_names}

    def once():
        request.infer(feed)
        return [request.get_output_tensor(i).data.copy() for i in range(len(names))]

    out = bench(once, rec)
    return names, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--cell", required=True, choices=sorted(ORT_CELLS | {"ovnpu_pre"}))
    p.add_argument("--pass", dest="pass_", default="first")
    p.add_argument("--phase", default="run", choices=["compile", "run"], help="ovnpu_pre only")
    p.add_argument("--root", default=os.environ.get("ZOO_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--profile", action="store_true", help="ORT cells: count nodes per provider from one profiled run")
    p.add_argument("--save-outputs", action="store_true")
    p.add_argument("--ep-option", action="append", default=[], type=lambda s: tuple(s.split("=", 1)),
                   help="OpenVINO EP option or OpenVINO property K=V, repeatable")
    p.add_argument("--gate", default="", help="JSON file with the quiet-gate readings of the chain")
    p.add_argument("--tag", default="", help="suffix of the JSON name (a second load, a variant)")
    p.add_argument("--no-engine-time", action="store_true", help="skip reading the GPU Engine counters of this process")
    args = p.parse_args()
    cell_name = args.cell + ("_compile" if args.cell == "ovnpu_pre" and args.phase == "compile" else "") + (f"_{args.tag}" if args.tag else "")
    out_json = os.path.join(args.root, "runs", args.pass_, args.model, cell_name + ".json")
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    rec = {"model": args.model, "cell": args.cell, "phase": args.phase, "pass": args.pass_, "tag": args.tag,
           "started": now(), "host": platform.node(), "python": sys.version.split()[0], "versions": {"numpy": np.__version__},
           "argv": sys.argv[1:], "status": "ERROR"}
    if args.gate and os.path.exists(args.gate):
        with open(args.gate, encoding="utf-8-sig") as f:
            rec["gate"] = json.load(f)
    try:
        inp = os.path.join(args.root, "inputs", args.model + ".npz")
        rec["input_sha256"] = sha256(inp)
        with np.load(inp) as z:
            feeds = {k: z[k] for k in z.files}
        rec["inputs"] = {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in feeds.items()}
        if args.cell == "ovnpu_pre":
            names, outs = run_ov(args, rec, feeds)
        else:
            names, outs = run_ort(args, rec, feeds)
        if outs is not None:
            rec["output_stats"] = [{"name": n, "min": float(np.min(o)), "max": float(np.max(o)), "nan": int(np.isnan(o).sum()),
                                    "inf": int(np.isinf(o).sum()), "dtype": str(o.dtype), "shape": list(o.shape),
                                    "sha256": hashlib.sha256(np.ascontiguousarray(o).tobytes()).hexdigest(), **fingerprint(o)}
                                   for n, o in zip(names, outs)]
            if args.save_outputs:
                od = os.path.join(args.root, "out", args.model)
                os.makedirs(od, exist_ok=True)
                np.savez(os.path.join(od, cell_name + ".npz"), **{f"o{i}": o for i, o in enumerate(outs)}, _names=np.array(names))
        if outs is not None and not args.no_engine_time:
            rec["engine_running_ms"] = engine_time_ms(os.getpid())
        rec["status"] = "OK"
    except BaseException as e:  # noqa: BLE001
        rec["error"] = traceback.format_exc()
        rec["error_type"] = type(e).__name__
    rec["ended"] = now()
    with open(out_json + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=str)
    os.replace(out_json + ".tmp", out_json)
    ms = rec.get("ms", {}).get("median")
    print(f"{args.model} {cell_name} {rec['status']} load_ms={rec.get('load_ms')} compile_ms={rec.get('compile_ms')} median_ms={ms}", flush=True)
    sys.exit(0 if rec["status"] == "OK" else 3)


if __name__ == "__main__":
    main()
