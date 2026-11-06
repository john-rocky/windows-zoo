# Stage the per-model Hugging Face repo files (stage_hub.py in .venv-ov) for every model: ONNX, OpenVINO IR, NPU blob,
# manifest. Export work, not timing: each model waits while another lane holds the timing marker.
# Log = <root>\logs\stage_hub.log
param([string]$Models = "", [string]$Only = "")
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models52.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\stage_hub.log"
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
$list = Get-Content $Models | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") }
if ($Only) { $list = $list | Where-Object { $_ -in ($Only -split ",") } }
Say $log "stage start pid=$PID models=$($list.Count)"
foreach ($m in $list) {
    $null = Wait-Other $log
    cmd /c "`"$($Py.ov)`" -u `"$Root\pc\stage_hub.py`" --model $m >> `"$log`" 2>&1"
    Say $log "$m stage exit=$LASTEXITCODE"
}
Say $log "stage done"
