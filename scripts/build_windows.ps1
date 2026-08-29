[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Spec = Join-Path $ProjectRoot "UniversalUpscaler.spec"
$BuildDirectory = Join-Path $ProjectRoot "build"
$DistDirectory = Join-Path $ProjectRoot "dist"
$ApplicationDirectory = Join-Path $DistDirectory "UniversalUpscaler"
$Executable = Join-Path $ApplicationDirectory "UniversalUpscaler.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Virtual-environment Python was not found: $Python"
}
if (-not (Test-Path -LiteralPath $Spec -PathType Leaf)) {
    throw "PyInstaller spec was not found: $Spec"
}

try {
    $PyInstallerVersion = (& $Python -m PyInstaller --version 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $PyInstallerVersion) {
        throw "PyInstaller is not installed in the project virtual environment."
    }
} catch {
    throw "Unable to run PyInstaller with $Python. Install it in .venv first. $($_.Exception.Message)"
}

foreach ($StaleDirectory in @($BuildDirectory, $DistDirectory)) {
    if (Test-Path -LiteralPath $StaleDirectory) {
        $ResolvedStale = (Resolve-Path -LiteralPath $StaleDirectory).Path
        if (-not $ResolvedStale.StartsWith($ProjectRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to remove a directory outside the project: $ResolvedStale"
        }
        Remove-Item -LiteralPath $ResolvedStale -Recurse -Force
    }
}

Write-Host "Building UniversalUpscaler with PyInstaller $PyInstallerVersion..."
& $Python -m PyInstaller --noconfirm --clean $Spec
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE."
}
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Build completed without producing the expected executable: $Executable"
}

$ExecutableItem = Get-Item -LiteralPath $Executable
$FolderSize = (Get-ChildItem -LiteralPath $ApplicationDirectory -File -Recurse | Measure-Object Length -Sum).Sum
$ModelPaths = Get-ChildItem -LiteralPath $ApplicationDirectory -File -Recurse -Filter "*.onnx" |
    ForEach-Object { $_.FullName.Substring($ApplicationDirectory.Length + 1) }
$NativeLibraries = Get-ChildItem -LiteralPath $ApplicationDirectory -File -Recurse -Filter "*.dll" |
    ForEach-Object { $_.FullName.Substring($ApplicationDirectory.Length + 1) }
$PythonVersion = (& $Python --version 2>&1 | Out-String).Trim()

$Manifest = [ordered]@{
    executable_path = $Executable
    executable_size_bytes = $ExecutableItem.Length
    total_folder_size_bytes = $FolderSize
    bundled_model_paths = @($ModelPaths)
    bundled_native_libraries = @($NativeLibraries)
    build_timestamp_utc = [DateTime]::UtcNow.ToString("o")
    pyinstaller_version = $PyInstallerVersion
    python_version = $PythonVersion
}
$ManifestPath = Join-Path $ApplicationDirectory "packaging-manifest.json"
$Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

Write-Host "Build complete."
Write-Host "Executable: $Executable"
Write-Host ("Folder size: {0:N2} MiB" -f ($FolderSize / 1MB))
Write-Host "Manifest: $ManifestPath"
