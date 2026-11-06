# Quiet pass: the table's timing. Streams N, R, G, C one after the other, each unit after the quiet gate (chain.ps1
# -Quiet: the optional marker is held per unit and dropped for 90 s every 20 min). First, a control: places_fp16 on
# the CPU EP without and with ONNX Runtime profiling, alternating, in the same quiet window (does a profiled session
# keep its speed after end_profiling?). Log = <root>\logs\quiet_pass.log, end marker <root>\quiet_pass.done
param([string]$Models = "", [string]$Only = "", [switch]$NoControl)
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models52.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\quiet_pass.log"
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
Remove-Item "$Root\quiet_pass.done" -ErrorAction SilentlyContinue
Say $log "quiet pass start pid=$PID"
if (-not $NoControl) {
    foreach ($i in 1, 2) {
        foreach ($t in "noprof", "prof") {
            $null = Wait-Other $log
            if ($Busy) { [IO.File]::WriteAllText($Busy, "$Label $(Get-Date -Format o) (quiet pass, profiling control)") }
            $t0 = Get-Date
            while (((1..3 | ForEach-Object { Start-Sleep -Seconds 1; CpuPct } | Measure-Object -Maximum).Maximum -ge 15) -and ((Get-Date) - $t0).TotalMinutes -lt 30) { Start-Sleep -Seconds 20 }
            $flags = @("--model", "places_fp16", "--cell", "cpu", "--pass", "quiet", "--tag", "control_${t}$i")
            if ($t -eq "prof") { $flags += "--profile" }
            $rc = Run-Unit $Py.ort $flags "$Root\logs\quiet\places_fp16\cpu_control_${t}$i"
            Say $log "control $t$i rc=$rc :: $((Get-Content "$Root\logs\quiet\places_fp16\cpu_control_${t}$i.out" -ErrorAction SilentlyContinue) -join ' | ')"
        }
    }
    if ($Busy -and (Test-Path $Busy) -and ((Get-Content $Busy -Raw) -like "$Label *")) { Remove-Item $Busy -Force }
}
foreach ($s in "N", "R", "G", "C") {
    Say $log "stream $s"
    $sel = if ($Only) { "-Only $Only" } else { "" }
    cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Root\pc\chain.ps1`" -Stream $s -Pass quiet -Quiet -Models `"$Models`" $sel >> `"$log`" 2>&1"
    Say $log "stream $s ended"
}
Say $log "quiet pass done"
"ok" | Out-File "$Root\quiet_pass.done" -Encoding ascii
