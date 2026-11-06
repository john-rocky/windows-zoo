# Rebuild the ONNX files from the zoo's .tflite on Hugging Face (convert\convert_all.py in .venv-tf) into <root>\onnx.
# One model at a time; each waits for the other lanes' marker first (a conversion keeps a CPU core busy for minutes).
# Needs .venv-tf: run setup_venvs.ps1 with $env:ZOO_WITH_CONVERT = "1". Log = <root>\logs\convert.log
param([string]$Models = "", [string]$Only = "")
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\convert.log"
New-Item -ItemType Directory -Force "$Root\logs", "$Root\onnx" | Out-Null
$py = "$Root\.venv-tf\Scripts\python.exe"
$list = Get-Content $Models | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") }
if ($Only) { $list = $list | Where-Object { $_ -in ($Only -split ",") } }
Say $log "convert start pid=$PID models=$($list.Count)"
foreach ($m in $list) {
    $null = Wait-Other $log
    cmd /c "`"$py`" -u `"$Root\convert\convert_all.py`" --out `"$Root\onnx`" --model $m >> `"$log`" 2>&1"
    Say $log "$m convert exit=$LASTEXITCODE"
}
cmd /c "`"$py`" -u `"$Root\convert\convert_all.py`" --out `"$Root\onnx`" --summary >> `"$log`" 2>&1"
Say $log "convert done"
