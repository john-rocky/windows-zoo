# Rebuild everything from an empty folder

A cell is one model on one runtime and processor. The line, or pass line, is the rule in [METHOD.md](../METHOD.md), section 3.

Windows 11 with an Intel Core Ultra NPU and an NVIDIA GPU, Python 3.12, git, and [uv](https://docs.astral.sh/uv/) on
PATH (`winget install astral-sh.uv`). In PowerShell, from an empty folder. Every script writes into the clone, into
folders git ignores (`onnx\`, `fp16\`, `blobs\`, `inputs\`, `samples\`, `runs\`, `out\`, `parity\`, `prof\`, `logs\`,
and the venvs). To try a few models first, add `-Only places_fp16,dewarp` to steps 2, 4 and 5 and
`--model places_fp16 dewarp` to the two checks of step 6 (dewarp has a cell that does not run and one that fails the
line).

1. Code and venvs (four venvs, 4.2 GB on disk; versions in `pc\req\`):

   ```powershell
   git clone https://github.com/john-rocky/windows-zoo; cd windows-zoo
   $env:ZOO_WITH_CONVERT = "1"; powershell -ExecutionPolicy Bypass -File pc\setup_venvs.ps1
   ```

2. Rebuild the ONNX files from the zoo's `.tflite` on Hugging Face (the revisions and sha256 are pinned in
   `convert\recipe.json`): the 52 measured files with tf2onnx 1.17.0 and the two earlier patches of
   `convert\patch_tf2onnx.py` (a missing `fused_activation_function` read as NONE; handlers for three TFLite ops), and
   the 13 `_fixed` files with the FULLY_CONNECTED bias fix as well:

   ```powershell
   powershell -ExecutionPolicy Bypass -File pc\convert.ps1
   ```

   The line before `convert done` in `logs\convert.log` counts the files whose graph hash matches the measured file
   (`convert\graph_signature.py`: ops, attributes, wiring and weights). tf2onnx does not write the same bytes twice,
   and from run to run it can also build a graph with another hash (cpga_fp16 did once: the same node and weight
   counts, with a Reshape after a ReduceMin instead of after a ReduceMax); step 6 checks such files by their outputs.

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
   fixed positions and in the mean) or `DIFFERENT` per model, and `not run` for yolact (ONNX Runtime 1.31.1 cannot
   load it, here or in the table). `git diff results RESULTS_WINDOWS.md` then shows your
   numbers against the table's.
