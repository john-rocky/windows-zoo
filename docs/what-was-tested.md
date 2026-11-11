# What was tested

A cell is one model on one runtime and processor. The line, or pass line, is the rule in [METHOD.md](../METHOD.md), section 3.

52 single-graph ONNX files of the [LiteRT zoo](https://github.com/john-rocky/LiteRT-Models)'s vision and audio models
(a few models are two graphs, such as a detector's encoder and decoder), on every processor of one Windows laptop: the
Intel NPU (compiled at load, and as a precompiled blob), the Intel GPU, the NVIDIA RTX GPU and the CPU. Each file runs
on each processor through ONNX Runtime or OpenVINO, its outputs are compared with the CPU's fp32 outputs, and the
times sit next to the same 52 files on a Snapdragon X2 Elite and a Galaxy S26 (Qualcomm AI Hub, measured earlier).
Each ONNX file is also checked against the `.tflite` it was converted from: 13 of the 52 lose the bias of some
FULLY_CONNECTED layers in tf2onnx 1.17.0, so those 13 are converted again with a one-line fix
(`convert/patch_tf2onnx.py`) and measured as `<model>_fixed`.

Results are in [RESULTS_WINDOWS.md](../RESULTS_WINDOWS.md): one row per file with its times, load times and parity, under
a five-line conclusion; `results/<model>/<cell>.json` holds each record. Counting only cells that pass the parity
line, the fastest processor is the RTX for 30 of the 52 files, the Intel GPU for 15 and the Intel NPU for 7 (either
RTX precision; either NPU form). Precompiled, the Intel NPU takes 1.55× the X2 Elite NPU's time at the median of the
per-file ratios (host to host here, AI Hub's device timer there). The precompiled blob loads in under 0.2 s (0.003 to
0.18 s), where the plain ONNX file takes 0.6 to 396 s to compile on the NPU at session load.

- The table: [RESULTS_WINDOWS.md](../RESULTS_WINDOWS.md)
- The files: a [Hugging Face collection](https://huggingface.co/collections/mlboydaisuke/windows-zoo-onnx-openvino-ir-and-intel-npu-blobs-6aeee0fd4065ed9b3e58d994) of 45 repos (`mlboydaisuke/<model>-OpenVINO`), each with the
  ONNX file, its OpenVINO IR (`openvino/`), the precompiled Intel NPU blob (`npu/`) and a `manifest.json`
- One JSON per model and cell: [results/](../results/) (`results/<model>/<cell>.json`: load time, the run times, node
  placement, the parity numbers, the error text of a cell that did not run, and the versions;
  `results/<model>/tflite.json`: the ONNX file against its `.tflite`)
- How each input is built: [pc/inputs_spec.json](../pc/inputs_spec.json) (the zoo's sample images at a pinned commit, the
  app's resize and normalisation)
- How it was measured and the pass line, written before measuring: [METHOD.md](../METHOD.md)

## The machine

MSI Stealth 16 AI+ (B3WI): Intel Core Ultra 9 386H with Intel AI Boost NPU (driver 32.0.100.4841) and Intel Graphics
(driver 32.0.101.8724), NVIDIA GeForce RTX 5080 Laptop GPU (driver 591.74, power limit 95 W as `nvidia-smi` reports
it), 31.5 GB RAM, on AC power in Windows' Balanced power mode. Windows 11 Home build 26200.9711 for the quiet pass
(the table's times); the first pass ran on 26200.6899 until a Windows Update restart (METHOD.md §6).

## Cells

| cell | runtime | processor | model form |
|---|---|---|---|
| `cpu` | ONNX Runtime 1.31.1, CPU EP | CPU | fp32 ONNX (the reference) |
| `cuda32` | onnxruntime-gpu 1.31.1, CUDA EP | RTX | fp32 ONNX |
| `cuda16` | onnxruntime-gpu 1.31.1, CUDA EP | RTX | fp16 copy (`convert_float_to_float16`, inputs and outputs kept fp32) |
| `ovgpu` | ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1 | Intel GPU | fp32 ONNX, the EP's default precision |
| `ovnpu` | ONNX Runtime 1.31.1 + OpenVINO EP 1.8.1 | NPU | fp32 ONNX as is, compiled when the session loads |
| `ovnpu_pre` | OpenVINO 2026.4.1 | NPU | `compile_model` → `export_model` blob, loaded with `import_model` |

## Licences

The models keep the licences of their Hugging Face repos ([LICENSES.md](../LICENSES.md)). The code here is
Apache-2.0 ([LICENSE](../LICENSE)); `convert/fold_deq.py` is copied from
[LiteRT-Models](https://github.com/john-rocky/LiteRT-Models) (MIT). `aihub/results_wide.json` holds the Qualcomm AI
Hub profile records of the X2 Elite and Galaxy S26 columns, and `aihub/results_recheck.json` two more jobs of the same
assets for 3 of the models.
