#requires -Version 5
# Measures first-build time, incremental-build time and final binary size,
# writing build.json (table 8.3). Kept separate from collect_all.ps1 because a
# cold build takes minutes. WARNING: rebuilds and deletes build/.
[CmdletBinding()]
param(
    [Parameter(Mandatory)] [string]$RepoRoot,
    [Parameter(Mandatory)] [string]$Config,
    [Parameter(Mandatory)] [string]$OutFile
)

$ErrorActionPreference = 'Stop'
$BuildDir = Join-Path $RepoRoot 'build'
$Exe      = Join-Path $BuildDir (Join-Path $Config 'Danqing.exe')

function Get-DirSizeMB {
    param([string]$Path)
    if (-not (Test-Path $Path)) { return 0 }
    $bytes = (Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue |
              Measure-Object -Property Length -Sum).Sum
    return [math]::Round(($bytes / 1MB), 2)
}

# --- 1. Cold-start build ---
if (Test-Path $BuildDir) {
    Write-Host "  removing existing build/ for cold-start measurement..."
    Remove-Item $BuildDir -Recurse -Force
}

Write-Host "  cmake configure..."
$swCfg = [System.Diagnostics.Stopwatch]::StartNew()
& cmake -S $RepoRoot -B $BuildDir -A x64 | Out-Null
$swCfg.Stop()
$cfgSec = [math]::Round($swCfg.Elapsed.TotalSeconds, 2)

Write-Host "  cmake build (cold)..."
$swBuild = [System.Diagnostics.Stopwatch]::StartNew()
& cmake --build $BuildDir --config $Config | Out-Null
$swBuild.Stop()
$firstBuildSec = [math]::Round($swBuild.Elapsed.TotalSeconds, 2)

# --- 2. Incremental build: touch main.cpp ---
$mainCpp = Join-Path $RepoRoot 'src/main.cpp'
(Get-Item $mainCpp).LastWriteTime = Get-Date
Write-Host "  cmake build (incremental, after touching main.cpp)..."
$swInc = [System.Diagnostics.Stopwatch]::StartNew()
& cmake --build $BuildDir --config $Config | Out-Null
$swInc.Stop()
$incSec = [math]::Round($swInc.Elapsed.TotalSeconds, 2)

# --- 3. Sizes ---
$exeSizeMb   = if (Test-Path $Exe) { [math]::Round(((Get-Item $Exe).Length / 1MB), 2) } else { 'n/a' }
$buildSizeMb = Get-DirSizeMB $BuildDir
# src + shaders: rough proxy for project code size
$repoCodeMb  = (Get-DirSizeMB (Join-Path $RepoRoot 'src')) + `
               (Get-DirSizeMB (Join-Path $RepoRoot 'shaders'))
$repoAllMb   = Get-DirSizeMB $RepoRoot

$result = [ordered]@{
    config                 = $Config
    cmake_configure_sec    = $cfgSec
    first_build_sec        = $firstBuildSec
    incremental_build_sec  = $incSec
    exe_path               = $Exe
    exe_size_mb            = $exeSizeMb
    build_dir_size_mb      = $buildSizeMb
    src_plus_shaders_mb    = [math]::Round($repoCodeMb, 2)
    repo_total_mb          = $repoAllMb
}

$result | ConvertTo-Json -Depth 4 | Set-Content $OutFile
Write-Host "  build measurement saved -> $OutFile"
$result | Format-List
