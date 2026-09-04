param([switch]$BuildTestSender)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Sdk = Join-Path $Root "native\third_party\spout2\SPOUTSDK"
$Out = Join-Path $Root "native\build\obs_spout"
$Vs = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$Includes = (& py -3.12 -m pybind11 --includes).Trim()
$PythonBase = (& py -3.12 -c "import sysconfig; print(sysconfig.get_config_var('installed_platbase'))").Trim()
$Sources = @("$PSScriptRoot\receiver.cpp", "$Sdk\SpoutDirectX\SpoutDX\SpoutDX.cpp")
foreach ($Name in @("SpoutCopy", "SpoutDirectX", "SpoutFrameCount", "SpoutSenderNames", "SpoutSharedMemory", "SpoutUtils")) {
    $Sources += "$Sdk\SpoutGL\$Name.cpp"
}
$QuotedSources = ($Sources | ForEach-Object { '"' + $_ + '"' }) -join ' '
$Command = "call `"$Vs`" -arch=x64 -host_arch=x64 && cl /nologo /std:c++20 /O2 /EHsc /MD /LD /DSPOUT_BUILD_STATIC $Includes /I`"$Sdk\SpoutDirectX\SpoutDX`" $QuotedSources /link /OUT:`"$Out\_urfts_obs_spout.pyd`" /LIBPATH:`"$PythonBase\libs`" python312.lib d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib shell32.lib advapi32.lib winmm.lib version.lib shlwapi.lib"
Push-Location $Out
try {
    cmd.exe /d /s /c $Command
    if ($LASTEXITCODE -ne 0) { throw "OBS receiver build failed" }
    if ($BuildTestSender) {
        $TestCommand = "call `"$Vs`" -arch=x64 -host_arch=x64 && cl /nologo /std:c++20 /O2 /EHsc /MD /DSPOUT_BUILD_STATIC /I`"$Sdk\SpoutDirectX\SpoutDX`" `"$PSScriptRoot\test_sender.cpp`" SpoutDX.obj SpoutCopy.obj SpoutDirectX.obj SpoutFrameCount.obj SpoutSenderNames.obj SpoutSharedMemory.obj SpoutUtils.obj /link /OUT:`"$Out\test_sender.exe`" d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib shell32.lib advapi32.lib winmm.lib version.lib shlwapi.lib"
        cmd.exe /d /s /c $TestCommand
        if ($LASTEXITCODE -ne 0) { throw "Synthetic sender build failed" }
    }
} finally { Pop-Location }
Copy-Item "$Out\_urfts_obs_spout.pyd" "$Root\src\_urfts_obs_spout.pyd" -Force
