# windows-zoo

52 single-graph ONNX files of the [LiteRT zoo](https://github.com/john-rocky/LiteRT-Models)'s vision and audio models
(a few models are two graphs, such as a detector's encoder and decoder), on every processor of one Windows laptop: the
Intel NPU (two ways), the Intel GPU, the NVIDIA RTX GPU and the CPU. Each file runs on each processor through ONNX
Runtime or OpenVINO, its outputs are compared with the CPU's fp32 outputs, and the times sit next to the same 52
files on a Snapdragon X2 Elite and a Galaxy S26 (Qualcomm AI Hub, measured earlier). Each ONNX file is also checked
against the `.tflite` it was converted from: 13 of the 52 lose the bias of some FULLY_CONNECTED layers in tf2onnx
1.17.0, so those 13 are converted again with a one-line fix (`convert/patch_tf2onnx.py`) and measured as
`<model>_fixed`.

(results paragraph: written after the quiet pass)

- The table: [RESULTS_WINDOWS.md](RESULTS_WINDOWS.md)
- One JSON per model and cell: [results/](results/) (`results/<model>/<cell>.json`: load time, the run times, node
  placement, the parity numbers, the error text of a cell that did not run, and the versions;
  `results/<model>/tflite.json`: the ONNX file against its `.tflite`)
- How each input is built: [pc/inputs_spec.json](pc/inputs_spec.json) (the zoo's sample images at a pinned commit, the
  app's resize and normalisation)
- How it was measured and the pass line, written before measuring: [METHOD.md](METHOD.md)

## The machine

MSI Stealth 16 AI+ (B3WI): Intel Core Ultra 9 386H with Intel AI Boost NPU (driver 32.0.100.4841) and Intel
Graphics (driver 32.0.101.8724), NVIDIA GeForce RTX 5080 Laptop GPU (driver 591.74, runs under a 95 W software
power cap on AC), 31.5 GB RAM, Windows 11 Home build 26200.9711 (the table's times; the first pass, see METHOD.md §6).

## Cells

| cell | runtime | processor | model form |
|---|---|---|---|
| `cpu` | ONNX Runtime 1.31.1, CPU EP | CPU | fp32 ONNX (the reference) |
| `cuda32` | onnxruntime-gpu 1.31.1, CUDA EP | RTX | fp32 ONNX |
| `cuda16` | onnxruntime-gpu 1.31.1, CUDA EP | RTX | fp16 copy (`convert_float_to_float16`, inputs and outputs kept fp32) |
| `ovgpu` | ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1 | Intel GPU | fp32 ONNX, the EP's default precision |
| `ovnpu` | ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1 | NPU | fp32 ONNX as is, compiled when the session loads |
| `ovnpu_pre` | OpenVINO 2026.4.1 | NPU | `compile_model` → `export_model` blob, loaded with `import_model` |

## Reproduce

Windows 11 with an Intel Core Ultra NPU and an NVIDIA GPU, Python 3.12, git, and [uv](https://docs.astral.sh/uv/)
on PATH (`winget install astral-sh.uv`). In PowerShell, from an empty folder. Every script writes into the clone
(`onnx\`, `inputs\`, `runs\`, `out\`, `parity\`, `prof\`, `logs\`; all ignored by git). To try a few models first, add
`-Only places_fp16,dewarp` to steps 2, 4 and 5.

1. Code and venvs (four venvs, 4.2 GB on disk; versions in `pc\req\`):

   ```powershell
   git clone https://github.com/john-rocky/windows-zoo; cd windows-zoo
   $env:ZOO_WITH_CONVERT = "1"; powershell -ExecutionPolicy Bypass -File pc\setup_venvs.ps1
   ```

2. Rebuild the ONNX files from the zoo's `.tflite` on Hugging Face (the revisions and sha256 are pinned in
   `convert\recipe.json`): the 52 measured files with tf2onnx 1.17.0 as it is, and the 13 `_fixed` files with the
   FULLY_CONNECTED bias fix:

   ```powershell
   powershell -ExecutionPolicy Bypass -File pc\convert.ps1
   ```

   The last line of `logs\convert.log` counts the files whose graph hash matches the measured file
   (`convert\graph_signature.py`: ops, attributes, wiring and weights). tf2onnx does not write the same bytes twice,
   and for a few models it places a Transpose differently from run to run; step 6 checks those by their outputs.

3. Inputs (downloads the zoo's sample images at the pinned commit):

   ```powershell
   .venv-ort\Scripts\python.exe pc\make_inputs.py
   ```

4. First pass: every cell of every model, with node placement and saved outputs (the NPU blobs are compiled here).
   The NPU and RTX streams run at the same time, then the Intel GPU and CPU streams, then the parity check against
   the CPU (`logs\compare_first.log`):

   ```powershell
   powershell -ExecutionPolicy Bypass -File pc\first_pass.ps1
   ```

5. Quiet pass: the timing in the table, one unit at a time, each after the CPU, the RTX and the Intel GPU go idle
   (it starts with four short CPU runs of places_fp16 with and without ONNX Runtime profiling, a control):

   ```powershell
   powershell -ExecutionPolicy Bypass -File pc\quiet_pass.ps1
   ```

6. Check each ONNX file against its `.tflite` and your CPU outputs against the table's, then rebuild the results and
   the table from your runs:

   ```powershell
   .venv-tf\Scripts\python.exe convert\check_tflite.py
   .venv-ort\Scripts\python.exe check_cpu.py --src .
   .venv-ort\Scripts\python.exe merge.py --src .
   .venv-ort\Scripts\python.exe table.py
   ```

   `check_cpu.py` says `identical` (same output bytes), `close` (within 1e-3 of an output's standard deviation at 16
   fixed positions and in the mean) or `DIFFERENT` per model. `git diff results RESULTS_WINDOWS.md` then shows your
   numbers against the table's.

## Licences

The models keep the licences of their Hugging Face repos ([LICENSES.md](LICENSES.md)). The code here is
Apache-2.0 ([LICENSE](LICENSE)); `convert/fold_deq.py` is copied from
[LiteRT-Models](https://github.com/john-rocky/LiteRT-Models) (MIT). `aihub/results_wide.json` holds the Qualcomm AI
Hub profile records of the X2 Elite and Galaxy S26 columns.
