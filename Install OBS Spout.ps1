# Installs the separately distributed OBS plugin, not Python packages.
# OBS 32.1.2 / plugin 1.12.0. Checksum is published in the official release notes.
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
if (Get-Process obs64 -ErrorAction SilentlyContinue) {
    throw 'Close OBS before installing its plugin. Recording/streaming will not be interrupted automatically.'
}
$Cache = Join-Path $PSScriptRoot 'native\third_party\obs-spout-plugin'
$Archive = Join-Path $Cache 'plugin.zip'
$ExpectedHash = '4dee90dac2a2ea743dceec76bb95410e63d924d5b5f7e4816d263ceb9eece7bf'
New-Item -ItemType Directory -Force -Path $Cache | Out-Null
if (-not (Test-Path -LiteralPath $Archive)) {
    Invoke-WebRequest 'https://github.com/Off-World-Live/obs-spout2-plugin/releases/download/1.12.0/win-spout-1.12.0-windows-x64.zip' -OutFile $Archive
}
if ((Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ExpectedHash) {
    throw 'OBS Spout archive checksum mismatch. No plugin files have been installed.'
}
$Unpacked = Join-Path $Cache 'unpacked'
Expand-Archive -LiteralPath $Archive -DestinationPath $Unpacked -Force
$Source = Join-Path $Unpacked 'win-spout'
$Destination = 'C:\ProgramData\obs-studio\plugins\win-spout'
$Files = Get-ChildItem -LiteralPath $Source -File -Recurse | Where-Object { $_.Extension -ne '.pdb' }
# Do not replace an existing different plugin installation silently.
foreach ($File in $Files) {
    $Relative = $File.FullName.Substring($Source.Length + 1)
    $Target = Join-Path $Destination $Relative
    if ((Test-Path -LiteralPath $Target) -and
        (Get-FileHash -LiteralPath $Target).Hash -ne (Get-FileHash -LiteralPath $File.FullName).Hash) {
        throw "A different plugin file already exists at $Target. No replacement was performed."
    }
}
foreach ($File in $Files) {
    $Relative = $File.FullName.Substring($Source.Length + 1)
    $Target = Join-Path $Destination $Relative
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Target) | Out-Null
    Copy-Item -LiteralPath $File.FullName -Destination $Target -Force
    if ((Get-FileHash -LiteralPath $Target).Hash -ne (Get-FileHash -LiteralPath $File.FullName).Hash) {
        throw "Installed file verification failed: $Target"
    }
}
Write-Output "Installed and hash-verified OBS Spout 1.12.0 at $Destination"
Write-Output 'Restart OBS. Tools -> Spout Output Settings: enable sender named URFTS.'
