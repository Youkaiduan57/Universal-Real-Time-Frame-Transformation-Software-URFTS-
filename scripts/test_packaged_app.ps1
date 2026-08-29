[CmdletBinding()]
param(
    [int]$InitializationSeconds = 8
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ApplicationDirectory = Join-Path $ProjectRoot "dist\UniversalUpscaler"
$Executable = Join-Path $ApplicationDirectory "UniversalUpscaler.exe"
$InternalDirectory = Join-Path $ApplicationDirectory "_internal"
$Results = [ordered]@{}

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class UniversalUpscalerWindowCloser {
    private delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern bool EnumWindows(EnumWindowsProc callback, IntPtr lParam);

    [DllImport("user32.dll")]
    private static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint processId);

    [DllImport("user32.dll")]
    private static extern bool PostMessage(IntPtr hwnd, uint message, IntPtr wParam, IntPtr lParam);

    public static bool CloseProcessWindows(int targetProcessId) {
        bool posted = false;
        EnumWindows(delegate(IntPtr hwnd, IntPtr lParam) {
            uint processId;
            GetWindowThreadProcessId(hwnd, out processId);
            if (processId == (uint)targetProcessId) {
                posted |= PostMessage(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero);
            }
            return true;
        }, IntPtr.Zero);
        return posted;
    }
}
"@

function Assert-PackagedFile {
    param([string]$Name, [string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Name was not found: $Path"
    }
    $Results[$Name] = $Path.Substring($ApplicationDirectory.Length + 1)
}

Assert-PackagedFile "executable" $Executable
Assert-PackagedFile "srvgg_model" (Join-Path $InternalDirectory "models\SRVGGNetCompact_x2.onnx")
Assert-PackagedFile "rife_model" (Join-Path $InternalDirectory "models\RIFE_v3.6.onnx")
Assert-PackagedFile "model_provenance" (Join-Path $InternalDirectory "models\RIFE_v3.6_PROVENANCE.md")
Assert-PackagedFile "ifrnet_model" (Join-Path $InternalDirectory "models\IFRNet_S_Vimeo90K.onnx")
Assert-PackagedFile "ifrnet_license" (Join-Path $InternalDirectory "models\IFRNet_LICENSE.txt")
Assert-PackagedFile "ifrnet_provenance" (Join-Path $InternalDirectory "models\IFRNet_S_Vimeo90K_PROVENANCE.md")
Assert-PackagedFile "runtime_config" (Join-Path $InternalDirectory "configs\runtime_profile.json")

foreach ($ConflictingIcu in @("icuuc.dll", "icudt78.dll")) {
    $ConflictingPath = Join-Path $InternalDirectory $ConflictingIcu
    if (Test-Path -LiteralPath $ConflictingPath -PathType Leaf) {
        throw "Conflicting Poppler ICU library was bundled: $ConflictingPath"
    }
}
$Results["conflicting_poppler_icu_excluded"] = $true

$QtPlatformPlugin = Get-ChildItem -LiteralPath $ApplicationDirectory -File -Recurse -Filter "qwindows.dll" | Select-Object -First 1
if ($null -eq $QtPlatformPlugin) { throw "Qt platform plugin qwindows.dll is missing." }
$Results["qt_platform_plugin"] = $QtPlatformPlugin.FullName.Substring($ApplicationDirectory.Length + 1)

$OnnxLibraries = @(Get-ChildItem -LiteralPath $ApplicationDirectory -File -Recurse -Filter "*.dll" |
    Where-Object { $_.Name -match "^(onnxruntime|DirectML)" })
if ($OnnxLibraries.Count -eq 0) { throw "ONNX Runtime native libraries are missing." }
if (-not ($OnnxLibraries.Name -contains "onnxruntime.dll")) { throw "onnxruntime.dll is missing." }
if (-not ($OnnxLibraries.Name -contains "DirectML.dll")) {
    throw "DirectML.dll is missing."
}
$Results["onnx_runtime_libraries"] = @($OnnxLibraries | ForEach-Object { $_.FullName.Substring($ApplicationDirectory.Length + 1) })

$StandardError = Join-Path $env:TEMP "UniversalUpscaler-packaged-stderr-$PID.txt"
$StandardOutput = Join-Path $env:TEMP "UniversalUpscaler-packaged-stdout-$PID.txt"
$Process = $null
try {
    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $Executable
    $StartInfo.WorkingDirectory = $env:TEMP
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardError = $true
    $StartInfo.RedirectStandardOutput = $true
    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo
    if (-not $Process.Start()) {
        throw "The packaged GUI process could not be started."
    }
    Start-Sleep -Seconds $InitializationSeconds
    $Process.Refresh()
    if ($Process.HasExited) {
        $CapturedError = $Process.StandardError.ReadToEnd()
        throw "Packaged GUI exited during initialization (exit code $($Process.ExitCode)). $CapturedError"
    }
    $Results["launched_without_python_command"] = ($Process.Path -eq $Executable)
    $Results["remained_alive_for_seconds"] = $InitializationSeconds

    $CloseRequested = [UniversalUpscalerWindowCloser]::CloseProcessWindows($Process.Id)
    if (-not $CloseRequested) {
        $CloseRequested = $Process.CloseMainWindow()
    }
    if (-not $CloseRequested) {
        throw "The packaged GUI did not accept a normal window-close request."
    }
    if (-not $Process.WaitForExit(10000)) {
        throw "The packaged GUI did not close cleanly within 10 seconds."
    }
    if ($Process.ExitCode -ne 0) {
        throw "The packaged GUI returned exit code $($Process.ExitCode) after closing."
    }

    $ErrorText = $Process.StandardError.ReadToEnd()
    if ($ErrorText -match "ModuleNotFoundError|ImportError|DLL load failed|Failed to load dynlib") {
        throw "Missing-module or missing-DLL error detected: $ErrorText"
    }
    $Results["closed_cleanly"] = $true
    $Results["missing_module_or_dll_errors"] = $false
} finally {
    if ($null -ne $Process -and -not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    }
}

$ReportPath = Join-Path $ProjectRoot "packaging-report-smoke.json"
$Report = [ordered]@{
    tested_at_utc = [DateTime]::UtcNow.ToString("o")
    executable_path = $Executable
    results = $Results
}
$Report | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $ReportPath -Encoding UTF8

Write-Host "Packaged application smoke test passed."
$Results.GetEnumerator() | ForEach-Object { Write-Host ("  {0}: {1}" -f $_.Key, ($_.Value -join ", ")) }
Write-Host "Report: $ReportPath"
