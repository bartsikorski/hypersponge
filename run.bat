@echo off
rem Hypersponge launcher.  Usage:  run.bat                 (the window)
rem                             run.bat render -o deep.png --dive 1500 --cursor 137.5,-120.5
rem                             run.bat info            (OpenGL check)
cd /d "%~dp0"
where uv >nul 2>nul
if %errorlevel%==0 (
    uv run python -m hypersponge %*
) else (
    .venv\Scripts\python.exe -m hypersponge %*
)
