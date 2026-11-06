# Z1 chain: one stream of units over the model list, each unit after the other lanes' marker and with free RAM.
#   -Stream N : NPU      ovnpu (ONNX Runtime + OpenVINO EP, as is), then ovnpu_pre compile + run (OpenVINO blob)
#   -Stream R : RTX      cuda32, cuda16
#   -Stream G : Intel GPU ovgpu
#   -Stream C : CPU      cpu (the reference)
# -Pass first|quiet ; -Models = file with one model name per line ; -Quiet = take the marker and gate every unit
# Log = C:\src\zoo\logs\chain_<Stream>_<Pass>.log ; progress = C:\src\zoo\progress.txt
param([Parameter(Mandatory=$true)][string]$Stream, [string]$Pass = "first", [string]$Models = "",
      [switch]$Quiet, [int]$TimeoutMin = 30, [string]$Only = "")
. "$PSScriptRoot\common.ps1"
if (-not $Models) { $Models = "$PSScriptRoot\models52.txt" }  # $PSScriptRoot is empty in a param default under PowerShell 5.1 -File
$log = "$Root\logs\chain_${Stream}_$Pass.log"
New-Item -ItemType Directory -Force "$Root\logs" | Out-Null
function U([string]$PyKey, [string]$Cell, [string[]]$Flags) { [pscustomobject]@{ py = $PyKey; cell = $Cell; flags = $Flags } }
# one PSCustomObject per unit (not nested arrays: PowerShell flattens @(@(...)) with a single element)
$cells = @{
    N = @((U "ort" "ovnpu" @("--profile", "--save-outputs")), (U "ov" "ovnpu_pre" @("--phase", "compile")), (U "ov" "ovnpu_pre" @("--phase", "run", "--save-outputs")));
    R = @((U "cuda" "cuda32" @("--profile", "--save-outputs")), (U "cuda" "cuda16" @("--profile", "--save-outputs")));
    G = @((U "ort" "ovgpu" @("--profile", "--save-outputs")));
    C = @((U "ort" "cpu" @("--profile", "--save-outputs")))
}
if ($Pass -ne "first") {  # timing passes: no profiling, no saved outputs, and the blob from the first pass is reused
    foreach ($k in @($cells.Keys)) {
        $cells[$k] = @($cells[$k] | Where-Object { -not ($_.flags -contains "compile") } | ForEach-Object { U $_.py $_.cell @($_.flags | Where-Object { $_ -notin @("--profile", "--save-outputs") }) })
    }
}
function Gate {
    $t0 = Get-Date; $deadline = $t0.AddMinutes(60); $last = ""
    while ($true) {
        $c = (1..3 | ForEach-Object { Start-Sleep -Seconds 1; CpuPct } | Measure-Object -Maximum).Maximum
        $r = (RtxLine) -split ",\s*"
        $ig = IgpuPct
        $g = @{ cpu_pct_max3 = $c; rtx_util = [int]$r[0]; rtx_mem_mib = [int]$r[1]; rtx_pstate = $r[2]; igpu_util_sum = $ig; waited_s = [int]((Get-Date) - $t0).TotalSeconds; state = "quiet" }
        if ($c -lt 15 -and [int]$r[0] -eq 0 -and [int]$r[1] -lt 800 -and $ig -lt 5) { return $g }
        if ((Get-Date) -ge $deadline) { $g.state = "timeout"; Say $log "not quiet for 60 min, running: cpu $c rtx $($r[0]) % $($r[1]) MiB igpu $ig"; return $g }
        $t = "not quiet: cpu $c %, rtx $($r[0]) % $($r[1]) MiB, igpu $ig %"
        if ($t -ne $last) { Say $log $t; $last = $t }
        Start-Sleep -Seconds 20
    }
}
$list = Get-Content $Models | Where-Object { $_.Trim() -ne "" -and -not $_.StartsWith("#") }
if ($Only) { $list = $list | Where-Object { $_ -in ($Only -split ",") } }
Say $log "chain $Stream $Pass start pid=$PID models=$($list.Count) quiet=$Quiet"
$lastRelease = Get-Date
try {
    foreach ($m in $list) {
        if ($Quiet -and ((Get-Date) - $lastRelease).TotalMinutes -ge 20) {
            # give the other lanes a window: drop the marker for 90 s between models; if one takes it, Wait-Other waits
            if ($Busy -and (Test-Path $Busy) -and ((Get-Content $Busy -Raw) -like "$Label *")) { Remove-Item $Busy -Force }
            Say $log "marker released for 90 s"
            Start-Sleep -Seconds 90
            $lastRelease = Get-Date
        }
        foreach ($u in $cells[$Stream]) {
          try {  # a PowerShell error inside one unit is logged and the stream goes on
            $name = $u.cell + ($(if ($u.flags -contains "compile") { "_compile" } else { "" }))
            if ($name -eq "ovnpu_pre" -and -not (Test-Path "$Root\blobs\$m.blob")) { Mark $m $name "skip-no-blob"; continue }
            if ($Pass -ne "first") {
                $fj = "$Root\runs\first\$m\$name.json"
                if (-not (Test-Path $fj) -or -not ((Get-Content $fj -Raw) -match '"status": "OK"')) { Mark $m $name "skip-first-not-ok"; continue }
            }
            $w = Wait-Other $log
            $wr = Wait-Ram $log
            $extra = @()
            if ($Quiet) {
                if ($Busy) { [IO.File]::WriteAllText($Busy, "$Label $(Get-Date -Format o) (quiet pass, $m $name)") }
                $g = Gate; $g.other_marker_wait_s = $w; $g.ram_wait_s = $wr; $g.free_ram_gb = FreeRamGB
                $gf = "$Root\logs\$Pass\$m\$name.gate.json"
                New-Item -ItemType Directory -Force (Split-Path $gf) | Out-Null
                ($g | ConvertTo-Json -Compress) | Out-File -FilePath $gf -Encoding utf8
                $extra = @("--gate", $gf)
            }
            $rtxBefore = RtxLine
            Mark $m $name "start"
            $rc = Run-Unit $Py[$u.py] (@("--model", $m, "--cell", $u.cell, "--pass", $Pass) + @($u.flags) + $extra) "$Root\logs\$Pass\$m\$name" $TimeoutMin
            $state = switch ($rc) { 0 { "ok" } 3 { "error" } 124 { "timeout" } default { "crash-$rc" } }
            if ($rc -ne 0 -and $rc -ne 3) {  # no JSON from python: write one with the exit code and the stderr tail
                $j = "$Root\runs\$Pass\$m\$name.json"
                New-Item -ItemType Directory -Force (Split-Path $j) | Out-Null
                $tail = (Get-Content "$Root\logs\$Pass\$m\$name.err" -Tail 60 -ErrorAction SilentlyContinue) -join "`n"
                @{ model = $m; cell = $u.cell; pass = $Pass; status = "ERROR"; error_type = $state; exit_code = $rc; error = $tail; ended = (Get-Date -Format o) } | ConvertTo-Json | Out-File -FilePath $j -Encoding utf8
            }
            Mark $m $name $state
            Say $log "$m $name $state rc=$rc waited=$w ram_wait=$wr rtx_before=[$rtxBefore] rtx_after=[$(RtxLine)] :: $((Get-Content "$Root\logs\$Pass\$m\$name.out" -Tail 2 -ErrorAction SilentlyContinue) -join ' | ')"
            $free = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
            if ($free -lt 100) { Say $log "C: free $free GB < 100 GB" }
          } catch {
            Say $log "unit $m $($u.cell) FAILED in the chain: $_"
            Mark $m $u.cell "chain-error"
          }
        }
    }
    Say $log "chain $Stream $Pass done"
} catch {
    Say $log "chain FAILED $_"
} finally {
    if ($Quiet -and $Busy -and (Test-Path $Busy) -and ((Get-Content $Busy -Raw) -like "$Label *")) { Remove-Item $Busy -Force }
}
