# Creates "Hypersponge" shortcuts (Desktop, Start menu, this folder) that start the explorer with
# one click and without a console window:  .venv\Scripts\pythonw.exe -m hypersponge
#   powershell -ExecutionPolicy Bypass -File tools\make_shortcuts.ps1 [-NoDesktop]
param([switch]$NoDesktop)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$pyw = Join-Path $root ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $pyw)) { throw "Missing $pyw - run 'Install Hypersponge.bat' first." }
$icon = Join-Path $root "hypersponge\assets\hypersponge.ico"
$ws = New-Object -ComObject WScript.Shell
$targets = @($root, [Environment]::GetFolderPath("Programs"))
if (-not $NoDesktop) { $targets += [Environment]::GetFolderPath("Desktop") }
foreach ($dir in $targets) {
    $path = Join-Path $dir "Hypersponge.lnk"
    $lnk = $ws.CreateShortcut($path)
    $lnk.TargetPath = $pyw
    $lnk.Arguments = "-m hypersponge"
    $lnk.WorkingDirectory = $root
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "Hypersponge - Sierpinski carpet and Menger sponge explorer"
    $lnk.Save()
    Write-Host "  shortcut: $path"
}
