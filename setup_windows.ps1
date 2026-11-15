# Hypersponge - one-time environment setup for Windows (PowerShell)
#   Right-click > "Run with PowerShell", or:  powershell -ExecutionPolicy Bypass -File setup_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

# 1. uv (Python package and environment manager from Astral)
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv ..." -ForegroundColor Cyan
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
uv --version

# 2. Python 3.12 managed by uv (does not touch any system Python)
uv python install 3.12

# 3. Create .venv and install the locked dependencies (+ pytest and ruff)
uv sync --all-extras

# 4. Smoke test: OpenGL version and the 2D shader against numpy on this GPU
uv run python -m hypersponge info

Write-Host ""
Write-Host "Setup done." -ForegroundColor Green
Write-Host "Start the explorer with:   .\run.bat" -ForegroundColor Green
Write-Host "Run the tests with:        uv run pytest -q" -ForegroundColor Green
