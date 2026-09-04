[CmdletBinding()]
param()
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$VsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
$Ort = Join-Path $Root "native\third_party\onnxruntime-directml"
$Dml = Join-Path $Root "native\third_party\directml"
$Output = Join-Path $Root "native\build"
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$PyIncludes = (& py -3.12 -m pybind11 --includes).Trim()
$PythonBase = (& py -3.12 -c "import sysconfig; print(sysconfig.get_config_var('installed_platbase'))").Trim()
$Command = @"
call "$VsDevCmd" -arch=x64 -host_arch=x64 && cl /nologo /std:c++20 /O2 /EHsc /MD /LD $PyIncludes /I"$Ort\build\native\include" /I"$Dml\include" "$PSScriptRoot\bridge.cpp" /link /OUT:"$Output\_urfts_directml.pyd" /LIBPATH:"$PythonBase\libs" /LIBPATH:"$Ort\runtimes\win-x64\native" /LIBPATH:"$Dml\bin\x64-win" python312.lib onnxruntime.lib DirectML.lib d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib
"@
cmd.exe /d /s /c $Command
if ($LASTEXITCODE -ne 0) { throw "Native bridge build failed with exit code $LASTEXITCODE" }
Copy-Item "$Ort\runtimes\win-x64\native\onnxruntime.dll" $Output -Force
Copy-Item "$Ort\runtimes\win-x64\native\onnxruntime_providers_shared.dll" $Output -Force
Copy-Item "$Dml\bin\x64-win\DirectML.dll" $Output -Force
Copy-Item "$Output\_urfts_directml.pyd" (Join-Path $Root "src\_urfts_directml.pyd") -Force
Copy-Item "$Output\onnxruntime.dll" (Join-Path $Root "src\onnxruntime.dll") -Force
Copy-Item "$Output\onnxruntime_providers_shared.dll" (Join-Path $Root "src\onnxruntime_providers_shared.dll") -Force
Copy-Item "$Output\DirectML.dll" (Join-Path $Root "src\DirectML.dll") -Force
Write-Host "Built $Output\_urfts_directml.pyd"
