# Three venvs in the repo root (each runtime brings its own ONNX Runtime / OpenVINO, so one package family per venv).
#   .venv-ort  = onnxruntime (CPU EP, the reference) + onnxruntime-ep-openvino (Intel GPU / NPU through ONNX Runtime)
#   .venv-cuda = onnxruntime-gpu with the CUDA / cuDNN pip wheels (RTX)
#   .venv-ov   = openvino (NPU compile_model / export_model / import_model without ONNX Runtime)
# Versions = pc\req\<venv>.txt (the pip freeze of the venvs the table was measured with).
# Log = <root>\setup.log, marker = <root>\setup.done. Needs uv on PATH (winget install astral-sh.uv) and Python 3.12.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$log = "$root\setup.log"; $done = "$root\setup.done"
Remove-Item $done -ErrorAction SilentlyContinue
function Say([string]$t) { [IO.File]::AppendAllText($log, "[$(Get-Date -Format o)] $t`r`n") }
$uv = (Get-Command uv -ErrorAction SilentlyContinue).Source
if (-not $uv) { $uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\uv.exe" }
$env:UV_LINK_MODE = "hardlink"
$env:PYTHONUTF8 = "1"
Say "start pid=$PID"
function MkVenv([string]$name, [string]$req) {
    $v = "$root\$name"
    if (-not (Test-Path "$v\Scripts\python.exe")) {
        Say "venv $name"
        cmd /c "`"$uv`" venv --python 3.12 `"$v`" >> `"$log`" 2>&1"
    }
    Say "install $name : $req"
    cmd /c "`"$uv`" pip install --python `"$v\Scripts\python.exe`" -r `"$req`" >> `"$log`" 2>&1"
    Say "exit $name = $LASTEXITCODE"
    cmd /c "`"$uv`" pip freeze --python `"$v\Scripts\python.exe`" > `"$root\freeze$name.txt`" 2>&1"
}
MkVenv ".venv-ort"  "$PSScriptRoot\req\ort.txt"
MkVenv ".venv-cuda" "$PSScriptRoot\req\cuda.txt"
MkVenv ".venv-ov"   "$PSScriptRoot\req\ov.txt"
if ($env:ZOO_WITH_CONVERT -eq "1") {  # only to rebuild the ONNX files from the zoo's .tflite (convert\convert_all.py)
    MkVenv ".venv-tf" "$PSScriptRoot\req\tf.txt"
    cmd /c "`"$root\.venv-tf\Scripts\python.exe`" `"$root\convert\patch_tf2onnx.py`" >> `"$log`" 2>&1"
    Say "patch_tf2onnx exit = $LASTEXITCODE"
}
Say "done"
"ok" | Out-File $done -Encoding ascii
