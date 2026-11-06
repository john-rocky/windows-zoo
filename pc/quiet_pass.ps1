# Quiet pass: the table's timing. Streams N, R, G, C one after the other, each unit after the quiet gate (chain.ps1
# -Quiet: the optional marker is held per unit and dropped between units for 90 s every 20 min, or for 15 min after 15
# min while another job waits). First, a control: places_fp16 on
# the CPU EP without and with ONNX Runtime profiling, alternating, in the same quiet window (does a profiled session
# keep its speed after end_profiling?). Log = <root>\logs\quiet_pass.log, end marker <root>\quiet_pass.done
# -ResumeSince <time>: a restarted pass skips the units it already ran since that time (chain.ps1 -ResumeSince).
param([string]$Models = "", [string]$Only = "", [switch]$NoControl, [string]$ResumeSince = "")
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\quiet_pass.log"
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
Remove-Item "$Root\quiet_pass.done" -ErrorAction SilentlyContinue
Say $log "quiet pass start pid=$PID$(if ($ResumeSince) { " resume since $ResumeSince" })"
if (-not $NoControl) {
    foreach ($i in 1, 2) {
        foreach ($t in "noprof", "prof") {
            $null = Wait-Other $log
            while (-not (Acquire-Busy "quiet pass, profiling control")) { $null = Wait-Other $log }
            $t0 = Get-Date
            while (((1..3 | ForEach-Object { Start-Sleep -Seconds 1; CpuPct } | Measure-Object -Maximum).Maximum -ge 15) -and ((Get-Date) - $t0).TotalMinutes -lt 10) { Start-Sleep -Seconds 20 }
            $flags = @("--model", "places_fp16", "--cell", "cpu", "--pass", "quiet", "--tag", "control_${t}$i")
            if ($t -eq "prof") { $flags += "--profile" }
            $rc = Run-Unit $Py.ort $flags "$Root\logs\quiet\places_fp16\cpu_control_${t}$i"
            Say $log "control $t$i rc=$rc :: $((Get-Content "$Root\logs\quiet\places_fp16\cpu_control_${t}$i.out" -ErrorAction SilentlyContinue) -join ' | ')"
        }
    }
    Release-Busy
}
foreach ($s in "N", "R", "G", "C") {
    Say $log "stream $s"
    $sel = if ($Only) { "-Only $Only" } else { "" }
    if ($ResumeSince) { $sel += " -ResumeSince $ResumeSince" }
    cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Root\pc\chain.ps1`" -Stream $s -Pass quiet -Quiet -Models `"$Models`" $sel >> `"$log`" 2>&1"
    Say $log "stream $s ended"
}
# A unit that ran after 10 minutes without quiet (gate=timeout) is measured once more at the end (METHOD.md section 6)
$streamOf = @{ ovnpu = "N"; ovnpu_pre = "N"; cuda32 = "R"; cuda16 = "R"; ovgpu = "G"; cpu = "C" }
$again = @{}
foreach ($g in Get-ChildItem "$Root\logs\quiet" -Recurse -Filter "*.gate.json" -ErrorAction SilentlyContinue) {
    $j = Get-Content $g.FullName -Raw | ConvertFrom-Json
    if ($j.state -eq "timeout") {
        $cell = $g.Name -replace "\.gate\.json$", ""
        $s = $streamOf[$cell]
        if ($s) { if (-not $again[$s]) { $again[$s] = @() }; $again[$s] += $g.Directory.Name }
    }
}
foreach ($s in "N", "R", "G", "C") {
    if (-not $again[$s]) { continue }
    $mf = "$Root\logs\quiet_again_$s.txt"
    ($again[$s] | Sort-Object -Unique) | Out-File $mf -Encoding ascii
    Say $log "stream $s again for gate=timeout: $(($again[$s] | Sort-Object -Unique) -join ',')"
    cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File `"$Root\pc\chain.ps1`" -Stream $s -Pass quiet -Quiet -Models `"$mf`" >> `"$log`" 2>&1"
}
Say $log "quiet pass done"
"ok" | Out-File "$Root\quiet_pass.done" -Encoding ascii
