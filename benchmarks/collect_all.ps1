#requires -Version 5
# One-shot benchmark collection for paper.md section 8 (Windows): runs the
# presets with the paper protocol (5s warmup + 10s sample, vsync off), measures
# build times and binary size, and writes CSV/JSON to benchmarks/out/<TS>/.
[CmdletBinding()]
param(
    [int]$Repeats = 3,
    [switch]$SkipBuild,
    [ValidateSet('Debug', 'Release')]
    [string]$Config = 'Debug',
    [string[]]$Presets = @('Empty', 'Stress100', 'Stress1000', 'LightStress', 'TextureStress'),
    [int]$TimeoutSec = 60,
    [int]$MaxFrames = 30000
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# 1. Resolve paths
# ---------------------------------------------------------------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot  = Resolve-Path (Join-Path $ScriptDir '..')
$BuildDir  = Join-Path $RepoRoot ('build')
$BinDir    = Join-Path $BuildDir $Config
$Exe       = Join-Path $BinDir 'Danqing.exe'

$Ts        = Get-Date -Format 'yyyyMMdd_HHmmss'
$OutDir    = Join-Path $ScriptDir ("out\" + $Ts)
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# Tee stdout to a log file for cross-machine comparison
$LogFile = Join-Path $OutDir 'collect.log'
Start-Transcript -Path $LogFile -Append | Out-Null

Write-Host "==============================================="
Write-Host "  Renderer Benchmark Collector"
Write-Host "  repo    : $RepoRoot"
Write-Host "  build   : $BuildDir ($Config)"
Write-Host "  outDir  : $OutDir"
Write-Host "  presets : $($Presets -join ', ')   x$Repeats"
Write-Host "==============================================="

# ---------------------------------------------------------------------------
# 2. Collect machine and software fingerprint
# ---------------------------------------------------------------------------
Write-Host "`n[1/6] Collecting environment fingerprint..."
$os  = Get-CimInstance Win32_OperatingSystem
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$gpu = Get-CimInstance Win32_VideoController |
       Where-Object { $_.AdapterRAM -gt 0 } |
       Sort-Object { if ($_.Name -match 'NVIDIA|AMD|Radeon|GeForce') { 0 } else { 1 } } |
       Select-Object -First 1
$mem = Get-CimInstance Win32_ComputerSystem

function Get-ToolVersion {
    param([string]$Exe, [string]$Args = '--version')
    try {
        $v = (& $Exe $Args.Split(' ') 2>$null) | Select-Object -First 1
        if ($v) { return $v.ToString().Trim() } else { return 'not found' }
    } catch { return 'not found' }
}

$env = [ordered]@{
    timestamp        = (Get-Date -Format 'o')
    host             = $env:COMPUTERNAME
    user             = $env:USERNAME
    os_caption       = $os.Caption
    os_build         = $os.BuildNumber
    cpu_name         = $cpu.Name
    cpu_logical      = $cpu.NumberOfLogicalProcessors
    cpu_physical     = $cpu.NumberOfCores
    ram_total_gb     = [math]::Round($mem.TotalPhysicalMemory / 1GB, 1)
    gpu_name         = $gpu.Name
    gpu_driver       = $gpu.DriverVersion
    gpu_vram_mb      = if ($gpu.AdapterRAM) { [math]::Round($gpu.AdapterRAM / 1MB) } else { 'n/a' }
    cmake_version    = Get-ToolVersion 'cmake' '--version'
    git_commit       = $(try { & git -C $RepoRoot rev-parse HEAD 2>$null } catch { 'n/a' })
    git_dirty        = $(try { (& git -C $RepoRoot status --porcelain 2>$null) -ne $null } catch { $false })
    config           = $Config
    repeats          = $Repeats
    presets          = ($Presets -join ',')
    benchmark_proto  = '5s warmup + 10s sample @ 1920x1080, vsync off'
}
$env | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $OutDir 'env.json') -Encoding UTF8
Write-Host ("  host={0}, cpu={1}, gpu={2}" -f $env.host, $env.cpu_name, $env.gpu_name)

# ---------------------------------------------------------------------------
# 3. Measure build time + binary size (table 8.3)
# ---------------------------------------------------------------------------
if (-not $SkipBuild) {
    Write-Host "`n[2/6] Measuring build time and binary size..."
    & (Join-Path $ScriptDir 'measure_build.ps1') `
        -RepoRoot $RepoRoot `
        -Config $Config `
        -OutFile (Join-Path $OutDir 'build.json')
} else {
    Write-Host "`n[2/6] SkipBuild: skipping build measurement."
}

# Earlier steps may have rebuilt the exe; if it is still missing, run at least
# one configure + build
if (-not (Test-Path $Exe)) {
    Write-Host "  exe still missing, running 'cmake --build' to generate it..."
    if (-not (Test-Path $BuildDir)) {
        & cmake -S $RepoRoot -B $BuildDir | Out-Null
    }
    & cmake --build $BuildDir --config $Config | Out-Null
}
if (-not (Test-Path $Exe)) {
    Stop-Transcript | Out-Null
    throw "Executable not produced: $Exe"
}

# ---------------------------------------------------------------------------
# 4. Table 8.2 body: presets x Repeats runs
# ---------------------------------------------------------------------------
Write-Host "`n[3/6] Running benchmark presets..."
$benchSink  = Join-Path $OutDir 'bench_csv'
New-Item -ItemType Directory -Force -Path $benchSink | Out-Null
$rawRows = New-Object System.Collections.Generic.List[object]

foreach ($preset in $Presets) {
    for ($r = 1; $r -le $Repeats; $r++) {
        Write-Host ("  [{0}] run {1}/{2}" -f $preset, $r, $Repeats)
        $args = @('--benchmark', $preset, '--auto-exit', '--frames', $MaxFrames)
        $stdout = Join-Path $OutDir ("_run_${preset}_${r}.stdout.log")
        $stderr = Join-Path $OutDir ("_run_${preset}_${r}.stderr.log")
        $p = Start-Process -FilePath $Exe -ArgumentList $args `
                -WorkingDirectory $BinDir `
                -RedirectStandardOutput $stdout `
                -RedirectStandardError  $stderr -PassThru
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            Write-Warning ("    timeout, killing pid {0}" -f $p.Id)
            Stop-Process -Id $p.Id -Force
            continue
        }
        # The exe writes CSV to cwd/benchmarks/result_*.csv
        $cwdBench = Join-Path $BinDir 'benchmarks'
        $csv = Get-ChildItem -Path $cwdBench -Filter "result_${preset}_*.csv" -ErrorAction SilentlyContinue |
                 Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if (-not $csv) {
            Write-Warning ("    no CSV for {0}#{1}" -f $preset, $r)
            continue
        }
        $row = Import-Csv $csv.FullName | Select-Object -First 1
        $row | Add-Member -NotePropertyName preset -NotePropertyValue $preset -Force
        $row | Add-Member -NotePropertyName repeat -NotePropertyValue $r -Force
        $rawRows.Add($row)
        # Back up the raw CSV
        Copy-Item $csv.FullName -Destination (Join-Path $benchSink ("{0}_run{1}.csv" -f $preset, $r))
    }
}

# Write the aggregated SUMMARY (one row per preset; mean/worst over -Repeats runs)
$summary = New-Object System.Collections.Generic.List[object]
foreach ($preset in $Presets) {
    $rows = $rawRows | Where-Object { $_.preset -eq $preset }
    if ($rows.Count -eq 0) { continue }
    function Avg($prop) { ($rows | Measure-Object -Property $prop -Average).Average }
    function Min($prop) { ($rows | Measure-Object -Property $prop -Minimum).Minimum }
    function Max($prop) { ($rows | Measure-Object -Property $prop -Maximum).Maximum }
    $summary.Add([pscustomobject]@{
        preset           = $preset
        runs             = $rows.Count
        fps_avg_mean     = [math]::Round((Avg 'fps_avg'), 2)
        fps_p95_mean     = [math]::Round((Avg 'fps_p95'), 2)
        fps_max_mean     = [math]::Round((Avg 'fps_max'), 2)
        frame_ms_avg     = [math]::Round((Avg 'frame_ms_avg'), 3)
        frame_ms_p95     = [math]::Round((Avg 'frame_ms_p95'), 3)
        gpu_ms_avg       = [math]::Round((Avg 'gpu_ms_avg'), 3)
        cpu_ms_avg       = [math]::Round((Avg 'cpu_ms_avg'), 3)
        draws_avg        = [math]::Round((Avg 'draws_avg'), 1)
        # DrawCallStats five buckets (geometry_avg is the Unity-URP-comparable one)
        geometry_avg     = [math]::Round((Avg 'geometry_avg'), 1)
        light_volume_avg = [math]::Round((Avg 'light_volume_avg'), 1)
        lighting_avg     = [math]::Round((Avg 'lighting_avg'), 1)
        gizmo_avg        = [math]::Round((Avg 'gizmo_avg'), 1)
        imgui_avg        = [math]::Round((Avg 'imgui_avg'), 1)
        samples          = [int](Avg 'samples')
        fps_avg_worst    = [math]::Round((Min 'fps_avg'), 2)
        frame_ms_worst   = [math]::Round((Max 'frame_ms_avg'), 3)
    })
}
$summaryCsv = Join-Path $OutDir ("SUMMARY_{0}.csv" -f $Ts)
$summary | Export-Csv -NoTypeInformation -Path $summaryCsv
$rawCsv = Join-Path $OutDir ("RAW_{0}.csv" -f $Ts)
$rawRows | Export-Csv -NoTypeInformation -Path $rawCsv
Write-Host ("  summary -> {0}" -f $summaryCsv)
$summary | Format-Table preset, fps_avg_mean, fps_p95_mean, frame_ms_avg, gpu_ms_avg, cpu_ms_avg, draws_avg

# ---------------------------------------------------------------------------
# 5. First-frame latency / headless exit time (table 8.3 part 2)
# ---------------------------------------------------------------------------
Write-Host "`n[4/6] Measuring first-frame latency and headless exit time..."
$ffJson = Join-Path $OutDir 'first_frame.json'
$ff = [ordered]@{}

# Cold start to exit: smallest preset (Empty) with --frames 1
for ($i = 1; $i -le 3; $i++) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Exe `
            -ArgumentList @('--benchmark', 'Empty', '--auto-exit', '--frames', '1') `
            -WorkingDirectory $BinDir -PassThru -WindowStyle Hidden
    $p.WaitForExit($TimeoutSec * 1000) | Out-Null
    $sw.Stop()
    $ff["cold_start_to_exit_ms_run$i"] = $sw.ElapsedMilliseconds
}
# Wall-clock time of the full 5s+10s benchmark run (includes process spawn, context init)
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe `
        -ArgumentList @('--benchmark', 'Empty', '--auto-exit', '--frames', $MaxFrames) `
        -WorkingDirectory $BinDir -PassThru -WindowStyle Hidden
$p.WaitForExit($TimeoutSec * 1000) | Out-Null
$sw.Stop()
$ff['headless_full_run_ms'] = $sw.ElapsedMilliseconds
$ff | ConvertTo-Json | Set-Content $ffJson -Encoding UTF8
Write-Host ("  first-frame summary -> {0}" -f $ffJson)

# ---------------------------------------------------------------------------
# 6. Placeholder collection for figures 8.4 / 8.5
# ---------------------------------------------------------------------------
Write-Host "`n[5/6] Optional matrices (script overhead / culling strategy)..."

# Figure 8.4: script count vs CPU frame time. The engine has no CLI to mount N
# scripts yet; emit a placeholder marked TODO until main gains --script-stress N.
$scriptCsv = Join-Path $OutDir 'script_overhead.csv'
"# TODO: requires CLI '--script-stress N' in main.cpp, currently not implemented." | Set-Content $scriptCsv
"scripts,cpu_ms_avg,frame_ms_avg" | Add-Content $scriptCsv
Write-Host ("  script_overhead -> {0} (placeholder)" -f $scriptCsv)

# Figure 8.5: culling strategy. The engine has frustumCulling/bvhCulling
# switches, but no CLI exposure yet.
$cullCsv = Join-Path $OutDir 'culling.csv'
"# TODO: requires CLI '--culling {none|frustum|bvh}' in main.cpp." | Set-Content $cullCsv
"strategy,entities,cpu_ms_avg" | Add-Content $cullCsv
Write-Host ("  culling -> {0} (placeholder)" -f $cullCsv)

# ---------------------------------------------------------------------------
# 7. Report output paths
# ---------------------------------------------------------------------------
Write-Host "`n[6/6] Done."
Write-Host "------------------------------------------------"
Write-Host (" Output directory:`n  {0}" -f $OutDir)
Write-Host (" Suggested next step:`n  python ./benchmarks/aggregate_report.py {0}" -f $OutDir)
Write-Host "------------------------------------------------"

Stop-Transcript | Out-Null
