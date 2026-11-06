# Shared helpers for the chain scripts (dot-source: . "$PSScriptRoot\common.ps1"). PowerShell 5.1.
$Root = Split-Path -Parent $PSScriptRoot   # the repo root: pc\ sits in it, and so do the venvs, onnx\, inputs\, runs\
$Busy = ""                                # optional: a marker file shared with other jobs on the PC (set in pc\local.ps1)
$Label = "windows-zoo"
if (Test-Path "$PSScriptRoot\local.ps1") { . "$PSScriptRoot\local.ps1" }
$Py = @{ ort = "$Root\.venv-ort\Scripts\python.exe"; cuda = "$Root\.venv-cuda\Scripts\python.exe"; ov = "$Root\.venv-ov\Scripts\python.exe" }
$Run = "$Root\pc\zoo_run.py"
$Smi = "$env:SystemRoot\System32\nvidia-smi.exe"
$Progress = "$Root\progress.txt"

# Two streams append to the same progress file: retry when the other one holds it for a moment.
function Append-Retry([string]$Path, [string]$Line) {
    foreach ($i in 1..50) {
        try { [IO.File]::AppendAllText($Path, $Line); return } catch { Start-Sleep -Milliseconds (20 + (Get-Random -Maximum 80)) }
    }
}
function Say([string]$Log, [string]$t) { Append-Retry $Log "[$(Get-Date -Format o)] $t`r`n" }
function Mark([string]$Model, [string]$Cell, [string]$State) { Append-Retry $Progress "$(Get-Date -Format o) $Model $Cell $State`r`n" }

# Wait while another lane holds the marker (younger than 2 h). The marker must then stay absent for 3 minutes
# (lanes chain plans back to back and drop the marker between them: about a second between plans, 1-2 minutes for a
# build; L1 asked for at least a full minute on 11-06).
# Returns the seconds waited.
# -IgnoreBuild (jobs that do not time anything: the first pass, conversion, export): a marker line that ends in the
# word `build` means the other lane uses the PC but is not timing (build, download, export, parity) = do not wait.
function Wait-Other([string]$Log, [switch]$IgnoreBuild) {
    if (-not $Busy) { return 0 }
    $t0 = Get-Date; $last = ""
    while ($true) {
        while (Test-Path $Busy) {
            $text = (Get-Content $Busy -Raw -ErrorAction SilentlyContinue)
            if ($text -and $text.StartsWith($Label)) { return [int]((Get-Date) - $t0).TotalSeconds }
            if ($IgnoreBuild -and $text -and ($text -match '\bbuild\b')) {
                if ($text -ne $last) { Say $Log "marker says build, not waiting: $($text.Trim())" }
                return [int]((Get-Date) - $t0).TotalSeconds
            }
            $age = ((Get-Date) - (Get-Item $Busy).LastWriteTime).TotalMinutes
            if ($age -ge 120) { Say $Log "marker $([int]$age) min old, not waiting: $text"; return [int]((Get-Date) - $t0).TotalSeconds }
            if ($text -ne $last) { Say $Log "waiting for the marker: $text"; $last = $text }
            Start-Sleep -Seconds 20
        }
        if ($last -eq "") {  # no marker at entry: look once more after 3 s (a chain drops its marker for ~1 s between plans)
            Start-Sleep -Seconds 3
            if (-not (Test-Path $Busy)) { return [int]((Get-Date) - $t0).TotalSeconds }
            continue
        }
        $clear = $true
        foreach ($i in 1..18) { Start-Sleep -Seconds 10; if (Test-Path $Busy) { $clear = $false; break } }
        if ($clear) { Say $Log "marker absent for 180 s"; return [int]((Get-Date) - $t0).TotalSeconds }
    }
}

function FreeRamGB { [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 1) }
function Wait-Ram([string]$Log, [double]$MinGB = 4) {
    $t0 = Get-Date
    while ((FreeRamGB) -lt $MinGB) { Say $Log "free RAM $(FreeRamGB) GB < $MinGB GB, waiting"; Start-Sleep -Seconds 30 }
    return [int]((Get-Date) - $t0).TotalSeconds
}

function CpuPct { (Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor | Where-Object { $_.Name -eq "_Total" }).PercentProcessorTime }
function RtxLine { (& $Smi "--query-gpu=utilization.gpu,memory.used,pstate,clocks.gr,clocks.mem,power.draw,clocks_event_reasons.sw_power_cap" "--format=csv,noheader,nounits" 2>$null) -join " " }
# The Intel GPU's adapter LUID in the counter names changes on reboot: read it once per script from OpenVINO.
$script:IgpuLuid = $null
function IgpuLuid {
    if ($null -eq $script:IgpuLuid) {
        $exe = $Py.ov
        $j = (& $exe "$Root\pc\luids.py" 2>$null) | ConvertFrom-Json
        $script:IgpuLuid = if ($j -and $j.GPU) { [string]$j.GPU } else { "" }
    }
    return $script:IgpuLuid
}
function IgpuPct {  # sum of the Intel GPU's engine utilisation; -1 = the adapter was not found
    $l = IgpuLuid
    if (-not $l) { return -1 }
    $s = (Get-Counter "\GPU Engine(*luid_${l}*)\Utilization Percentage" -ErrorAction SilentlyContinue).CounterSamples
    [math]::Round((($s | Measure-Object CookedValue -Sum).Sum), 1)
}

# Run one python unit with a timeout; stdout / stderr to files. Returns the exit code (124 = timeout).
function Run-Unit([string]$PyExe, [string[]]$UnitArgs, [string]$OutBase, [int]$TimeoutMin = 30) {
    New-Item -ItemType Directory -Force (Split-Path $OutBase) | Out-Null
    $p = Start-Process -FilePath $PyExe -ArgumentList (@("-u", $Run) + $UnitArgs) -RedirectStandardOutput "$OutBase.out" -RedirectStandardError "$OutBase.err" -PassThru -NoNewWindow
    $null = $p.Handle  # keep the handle so ExitCode stays readable after the exit
    if (-not $p.WaitForExit($TimeoutMin * 60 * 1000)) {
        & taskkill.exe /T /F /PID $p.Id 2>&1 | Out-Null
        return 124
    }
    return $p.ExitCode
}
