#requires -Version 5
# Runs the benchmark presets sequentially (--benchmark <preset> --auto-exit
# --frames <maxFrames>) and merges each preset's latest CSV into
# benchmarks/SUMMARY_<timestamp>.csv.
[CmdletBinding()]
param(
    [string[]]$Presets = @('Empty', 'Stress100', 'Stress1000', 'LightStress', 'TextureStress'),
    [int]$MaxFrames = 3000,
    [int]$TimeoutSec = 60,
    [string]$BinDir = ''
)

$ErrorActionPreference = 'Stop'

# Resolve BinDir
if (-not $BinDir) {
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $BinDir = Resolve-Path (Join-Path $scriptDir '..\build\Debug') -ErrorAction SilentlyContinue
    if (-not $BinDir) {
        Write-Error "Cannot auto-locate build\Debug; pass -BinDir explicitly."
        exit 1
    }
}
$exe = Join-Path $BinDir 'Danqing.exe'
if (-not (Test-Path $exe)) {
    Write-Error "Executable not found: $exe"
    exit 1
}

$benchDir = Join-Path $BinDir 'benchmarks'
New-Item -ItemType Directory -Force -Path $benchDir | Out-Null

Write-Host "==============================================="
Write-Host " Renderer Benchmark Batch Runner"
Write-Host "  exe     : $exe"
Write-Host "  benchDir: $benchDir"
Write-Host "  presets : $($Presets -join ', ')"
Write-Host "==============================================="

$batchTs = Get-Date -Format 'yyyyMMdd_HHmmss'
$results = @()

foreach ($preset in $Presets) {
    Write-Host ""
    Write-Host ("[Batch] Running preset '{0}' (timeout={1}s)..." -f $preset, $TimeoutSec)
    $args = @('--benchmark', $preset, '--auto-exit', '--frames', $MaxFrames)
    $stdout = Join-Path $benchDir "_batch_${preset}_stdout.log"
    $stderr = Join-Path $benchDir "_batch_${preset}_stderr.log"
    $p = Start-Process -FilePath $exe -ArgumentList $args -WorkingDirectory $BinDir `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    $exited = $p.WaitForExit($TimeoutSec * 1000)
    if (-not $exited) {
        Write-Warning ("[Batch] preset '{0}' TIMEOUT, killing process..." -f $preset)
        Stop-Process -Id $p.Id -Force
        continue
    }
    $ec = $p.ExitCode
    if ($null -ne $ec -and $ec -ne 0) {
        Write-Warning ("[Batch] preset '{0}' exited with code {1}" -f $preset, $ec)
    }

    # Find the CSV this run just produced (latest by LastWriteTime matching the preset)
    $csv = Get-ChildItem -Path $benchDir -Filter "result_${preset}_*.csv" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $csv) {
        Write-Warning ("[Batch] no CSV produced for '{0}'" -f $preset)
        continue
    }

    $row = Import-Csv $csv.FullName | Select-Object -First 1
    $results += $row
    Write-Host ("[Batch] OK '{0}'  fps_avg={1}  frame_ms_avg={2}  csv={3}" -f `
        $preset, $row.fps_avg, $row.frame_ms_avg, $csv.Name)

    Remove-Item $stdout, $stderr -ErrorAction SilentlyContinue
}

if ($results.Count -gt 0) {
    $summary = Join-Path $benchDir "SUMMARY_${batchTs}.csv"
    $results | Export-Csv -NoTypeInformation -Path $summary
    Write-Host ""
    Write-Host "==============================================="
    Write-Host (" Summary written: {0}" -f $summary)
    Write-Host "==============================================="
    $results | Format-Table scene, fps_avg, fps_p95, frame_ms_avg, gpu_ms_avg, cpu_ms_avg, draws_avg, samples
} else {
    Write-Warning "No successful benchmark runs."
    exit 2
}
