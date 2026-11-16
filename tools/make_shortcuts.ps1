# Creates "Hypersponge" shortcuts (Desktop, Start menu, this folder) that open the window
# without a console:  .venv\Scripts\pythonw.exe -m hypersponge
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
    $lnk.WorkingDirectory = [Environment]::GetFolderPath("MyPictures")
    $lnk.IconLocation = "$icon,0"
    $lnk.Description = "Hypersponge - the Sierpinski carpet and the Menger sponge with unlimited zoom"
    $lnk.Save()
    Write-Host "  shortcut: $path"
}
