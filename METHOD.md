# Method and pass line

Written on 2026-11-06 before the first measurement; changes are listed at the end with their time and reason.

## 1. Cells (one model × one runtime × one processor)

| cell | runtime | processor | model form | load time |
|---|---|---|---|---|
| `cpu` | ONNX Runtime 1.31.1, CPU EP | CPU | the fp32 ONNX file | wall time of `InferenceSession()` |
| `cuda32` | onnxruntime-gpu 1.31.1, CUDA EP (CUDA 13 pip wheels) | RTX 5080 Laptop | fp32 ONNX | same |
| `cuda16` | same | same | fp16 copy (`onnxruntime.transformers.float16.convert_float_to_float16(keep_io_types=True)`, default op block list, nothing per model) | same (conversion time kept apart) |
| `ovgpu` | ONNX Runtime 1.31.1 + `onnxruntime-ep-openvino` 1.8.1 | Intel GPU (`ov_device == "GPU.0"`) | fp32 ONNX, the EP's default precision | same, no `cache_dir` |
| `ovnpu` | same | NPU (`ov_device == "NPU"`) | fp32 ONNX as is, compiled when the session loads (default precision f16) | same; no `cache_dir`, so the load is a compile that hits no cache |
| `ovnpu_pre` | OpenVINO 2026.4.1 (pip, no ONNX Runtime) | NPU | `read_model(onnx)` → `compile_model(NPU)` → `export_model` blob = precompiled | `compile_model` time (`compile_ms`; export apart), and in a new process reading the blob + `import_model` (`load_ms`) |

The reference is the `cpu` output. No other NVIDIA EP is used. Precision, threads and streams stay at each
runtime's defaults. The JSON records a default where the runtime reports it (`INFERENCE_PRECISION_HINT` in the
`ovnpu_pre` cell); ONNX Runtime's thread count and the OpenVINO EP's precision are not recorded.

## 2. Inputs (the same file for every cell)

- `pc/make_inputs.py` writes `inputs/<model>.npz` and their sha256.
- Images: the sample files of the zoo's app folders in `john-rocky/LiteRT-Models` at commit
  `96e0bc1f7965bedc17b36ab1ea5403e79e513cf1` (`<dir>/app/src/main/assets/`). A model whose folder has no sample
  borrows the sample of a folder with similar content (written in `pc/inputs_spec.json`). Resize and normalisation
  follow the zoo app's code (`pc/inputs_spec.json`, with the source path and line).
- Audio: the app's bundled `test_audio.bin`.
- Inputs that take another graph's output (the DETR graph B files, w2v2_head, the DAC decoder, sam2_decoder) are
  built from the upstream model's `cpu` output, joined the way the app joins them (numpy ports of the app's top-300
  selection and host MLP, the RVQ, the SAM2 point embedding). Recurrent states (movinet) start at the app's initial
  values.
- An input whose normalisation neither the code nor the card fixes would get uniform [0, 1) noise with a seed per
  model, marked "unverified normalisation" in the notes; none of the 52 needed it.

## 3. Pass line (per output, against the `cpu` output)

- Values: max|Δ|, mean|Δ|, cosine similarity (float64 over all elements), NaN / Inf counts. Class-score outputs
  (`argmax_outputs` in `pc/inputs_spec.json`) also get the top-1 agreement per position; positions where the
  reference's top-1 and top-2 are exactly equal are not counted.
- **PASS** = every output has cos ≥ 0.99, no NaN / Inf, and every class-score output keeps ≥ 0.99 of its top-1
  positions (a single-position classifier must match).
- **FAIL** = any of these broken. The JSON names the broken outputs, their values and a candidate reason (fp16
  overflow when the reference exceeds 65,504, top-1 flips at small reference margins, and so on; a candidate is
  written as a candidate).
- **ERROR** = the cell did not run (conversion, compile, session, run, or the 30-minute timeout), with the full
  error text and the versions.
- The same line compares each ONNX file's CPU output with its source `.tflite` in the LiteRT interpreter
  (`convert/check_tflite.py`, `results/<model>/tflite.json`): a check of the conversion, not of a runtime.

## 4. Timing

- One unit = one process: load → first run (its outputs are saved, `first_run_ms`) → 5 warmup runs → 30 timed
  runs. The table shows the median of the 30 (p10 / p90 / min / max are recorded). A unit whose first run takes more
  than 3 s does 2 warmup + 10 timed runs (written in the JSON).
- `time.perf_counter()` around `session.run()` (ONNX Runtime) or `infer()` (OpenVINO), with host numpy inputs and
  host outputs: the copies between host and device are included (the time an app sees). No IO binding.
- Determinism: the max|Δ| between the first and the last run's outputs is recorded.

## 5. Nodes left on the CPU

- `ovnpu` / `ovgpu`: one profiled run (ONNX Runtime `enable_profiling`) in the same session, counting graph nodes
  per provider (the OpenVINO EP's subgraphs and the nodes left on `CPUExecutionProvider`). Profiling stops with
  `end_profiling()` before the timed runs.
- `ovnpu_pre`: the ops that `Core.query_model(model, "NPU")` does not claim. The whole model is compiled for the
  `NPU` device (no HETERO or AUTO device), so the compile passes or fails; a failure is recorded with its text.

## 6. Two passes

- **First pass**: the NPU stream (`ovnpu`, then `ovnpu_pre`) and the RTX stream (`cuda32`, then `cuda16`) run at the
  same time as two detached jobs, then the Intel GPU stream (`ovgpu`) and the CPU stream (`cpu`). Parity and the
  first timing come from this pass (`first_pass` in the JSON; not the table's timing).
- **Quiet pass**: serial, only the cells that ran in the first pass. Before each unit: whole-CPU load < 15 % (max of
  3 one-second reads), RTX `utilization.gpu` = 0 % and memory.used < 800 MiB, Intel GPU engine utilisation sum < 5 %.
  After 10 minutes without quiet the unit runs anyway and its row carries `gate=timeout` and the readings (such a
  unit is measured once more at the end). The marker file that the jobs on this machine share (last bullet) is
  created only if no other job holds it.
- The table's speed = the quiet-pass median.
- Windows build: Windows Update restarted the machine on 2026-11-06 between 17:29 and 17:35 JST (26200.6899 →
  26200.9711; the NPU, Intel GPU and RTX drivers stayed the same). The first pass ran on 26200.6899 except the
  last 31 units of the 13 fixed files (26200.9711); the quiet pass, the table's times, ran on 26200.9711. Times from
  the two builds are not compared: the first-pass times stay in the JSON records only.
- The RTX runs under a 95 W software power cap; AC power, Windows balanced power mode (unchanged). `nvidia-smi`
  pstate, clocks, power draw and the software power cap flag are written to `logs\chain_<stream>_<pass>.log` before
  and after each unit (not to the JSON).
- The NPU as-is load is called "a compile that hits no cache" only after loading the same model a second and a
  third time in new processes and seeing no speed-up.
- A free-RAM floor of 4 GB before each unit; an optional marker file lets the scripts wait for other jobs that share
  the machine (`pc/local.ps1`, not in the repo).

## 7. Cache and disk

OpenVINO's `CACHE_DIR` and the EP's `cache_dir` are not used for measuring. Blobs go to `blobs/`. The working folder
plus the NPU driver's cache folder stay under 50 GB.

## 8. Records

`runs/<pass>/<model>/<cell>.json` per unit (merged into `results/<model>/<cell>.json`): model, cell, pass, status,
full error text, versions (Python, ONNX Runtime, the OpenVINO EP, OpenVINO, the CUDA / cuDNN wheels, the NPU
driver), device name, `load_ms`, `compile_ms`, `first_run_ms`, timing statistics and run count, node placement,
`query_model`, the quiet-gate readings (CPU %, RTX utilisation, memory and pstate, Intel GPU %), start and end times,
the input sha256, output names and shapes, an output fingerprint (sha256, mean, std, 16 fixed sample values) and the
parity values.

## 9. The table

Columns: model | Galaxy S26 NPU | X2E NPU | Intel NPU as-is | Intel NPU precompiled | Intel GPU | RTX fp16 | CPU | load
(Intel NPU as-is / Intel NPU precompiled / X2E as-is) | parity per cell | note. RTX fp32 is in the JSON and the
parity column. The conclusion is 3 to 5 lines of measured facts: for each of the three findings of the X2 Elite
and Galaxy S26 run (the two chips within a few percent of each other; plain ONNX compiled at load against the
precompiled form, in load time and layers left on the CPU; two DETR graph-B files slower precompiled), whether it
holds on the Intel NPU, looking for the counter-example first.

## Changes

- 2026-11-06 10:5x JST, before any measurement: image geometry changed from "resize only" to the zoo app's own
  geometry per model (stretch, centre square crop, 3:4 crop, white or black letterbox, short side 256 then centre
  224, text-line crop), because the app code (with path and line) became available; the inputs fed by another graph
  follow the app's joining code; audio = the app's `test_audio.bin`.
- Same time: class-score outputs are picked by output shape, not by output index (tf2onnx does not keep the
  original output order). Models whose class 0 is background or unused (yolact, rfdetr, ssdlite) are counted from
  class 1; ssdlite's class output [1, 6×91, H, W] is split into 6 anchors × 91 classes.
- 2026-11-06 12:3x JST, after the first end-to-end run (an error, not a number, prompted it): the fp16 copy for
  `cuda16` is cleaned after `convert_float_to_float16`. For rtdetr_graphB_fp16 the converter inserted the same Cast
  (same input, same output name) once per consumer of an fp16 constant, so one tensor had several definitions and ONNX
  Runtime refused the copy as an invalid model. The cleaning drops a node that repeats an earlier node exactly,
  renames nodes whose name still repeats, and puts the nodes in dependency order (the converter appends the input
  Casts at the end). None of this changes the computation. The counts are in the `cuda16` record (`fp16_convert`) of
  every model except places_fp16 and dewarp: their fp16 copies were made in that first run, before this change, and
  kept (both load as they are; their records say `cached`).
- 2026-11-06 14:2x JST, after looking at first-pass numbers (so this change is marked as made after seeing numbers):
  the `ovnpu_pre` runner copied every output twice per run (`infer()` already returns copies, and the runner copied
  them again), which inflates the time of models with large outputs (faceparsing: 20 MB). It now takes the copies
  `infer()` returns, one host copy as in the other cells. The first-pass `ovnpu_pre` times of the first 9 models in
  the list (6drepnet to dmcount) were taken before the fix (placed on the machine at 14:26 JST); every quiet-pass time
  (the table) is after it.
- 2026-11-06 15:3x JST, before the quiet pass: the quiet gate waits at most 10 minutes (was 60), the cap used by every
  timing job on this shared machine; a unit that hits it is measured once more at the end. The marker is now taken
  only when no one holds it (create-new), never overwritten.
- 2026-11-06 15:3x JST, after the CPU cell of yolact failed (an error, not a number): ONNX Runtime 1.31.1 cannot load
  yolact.onnx at all (`[ShapeInferenceError] Incompatible dimensions` on an FPN Add), so its CPU reference does not
  exist. For such a model only, the reference is OpenVINO on the CPU in fp32 (`ovcpu`: the same ONNX file,
  `INFERENCE_PRECISION_HINT` f32), and the table row says so.
- 2026-11-06 16:3x JST, after the first pass and before the quiet pass (a missing check prompted it, not a number):
  each ONNX file is also run against the `.tflite` it was converted from (`convert/check_tflite.py`: the zoo's
  original file in the LiteRT interpreter on the CPU, the same input, the line of section 3). 13 of the 52 files
  compute something else: tf2onnx 1.17.0 drops the bias of every FULLY_CONNECTED layer whose `keep_num_dims` is false
  and whose input rank is not 2 (1 to 24 layers per file; 8 of the 13 fail the line). The table stays on the 52 files
  (the X2E / S26 columns measured them), with a note on those 13 rows. The same 13 `.tflite` files are converted
  again with a one-line fix of that handler (`convert/patch_tf2onnx.py`; `<model>_fixed` in `convert/recipe.json`).
  They are converted without `--continue_on_error`, except metric3d: there the gemm rewriter's IndexError remains,
  and it is raised before that rewriter changes the graph. The 13 are measured in every cell and both passes, in a
  second table; they are the files of the Hugging Face repos. A `_fixed` file takes its base model's input; an input
  built from another graph comes from that graph's `_fixed` file where there is one (the DETR graph B files).
- 2026-11-06 19:1x JST, during the quiet pass (a Windows Update restart prompted it, not a number): the first-pass
  times are no longer compared with the quiet pass, because the two passes ran on different Windows builds
  (section 6). The quiet pass was also paused once between two units to let another job on the machine take the
  marker, and resumed with the units already measured kept (`pc/quiet_pass.ps1 -ResumeSince`). The NPU cells of the
  first 8 models in the list (6drepnet to dis) are measured once more after the pass, and the table uses those: 15 of
  their 16 units ran between 18:19 and 18:36 JST, and each passed the gate, but the gate readings between them showed
  15-30 % whole-CPU load from an unknown source.
- 2026-11-07 01:1x JST, during the quiet pass (the other jobs on the machine asked for it, not a number): while
  another job waits for the machine, the marker is held for 15 minutes and dropped for 15 minutes (before: dropped for
  90 s every 20 minutes, still the rule when no job waits), and after it is taken back the first unit waits 60 s more
  before the gate (`pc/chain.ps1`). The gate before each unit is unchanged. The pass was restarted between two units
  for this, with the units already measured kept.
