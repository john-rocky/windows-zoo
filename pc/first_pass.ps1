# First pass: streams N (NPU) and R (RTX) at the same time as two detached processes, then G (Intel GPU) and
# C (CPU) at the same time, then parity for every model. Each unit inside a stream waits for the optional marker.
# Log = <root>\logs\first_pass.log, end marker <root>\first_pass.done, parity log <root>\logs\compare_first.log
param([string]$Models = "", [string]$Only = "")
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\first_pass.log"
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
Remove-Item "$Root\first_pass.done" -ErrorAction SilentlyContinue
function Start-Stream([string]$s) {
    $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Root\pc\chain.ps1`" -Stream $s -Pass first -Models `"$Models`""
    if ($Only) { $cmd += " -Only $Only" }
    $r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmd; CurrentDirectory = $Root }
    Say $log "stream $s started pid=$($r.ProcessId) ret=$($r.ReturnValue)"
    return $r.ProcessId
}
function Wait-Streams([int[]]$ids) {
    while ($true) {
        $alive = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if ($alive.Count -eq 0) { return }
        Start-Sleep -Seconds 30
    }
}
Say $log "first pass start pid=$PID"
$a = Start-Stream "N"; $b = Start-Stream "R"
Wait-Streams @($a, $b)
Say $log "N and R ended"
$c = Start-Stream "G"; $d = Start-Stream "C"
Wait-Streams @($c, $d)
Say $log "G and C ended"
$sel = if ($Only) { "--model " + ($Only -split "," -join " ") } else { "" }
cmd /c "`"$($Py.ort)`" -u `"$Root\pc\compare.py`" --pass first $sel > `"$Root\logs\compare_first.log`" 2>&1"
Say $log "compare exit=$LASTEXITCODE"
Say $log "first pass done"
"ok" | Out-File "$Root\first_pass.done" -Encoding ascii
