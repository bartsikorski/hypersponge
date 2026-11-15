@echo off
rem Double-click me.  Sets up Python and the packages (first time: a few minutes), checks the
rem shaders on this GPU, and puts a Hypersponge icon on the Desktop and in the Start menu.
setlocal
cd /d "%~dp0"
echo.
echo  === Hypersponge setup ===
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
if errorlevel 1 goto failed
echo.
echo Creating shortcuts ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\make_shortcuts.ps1"
if errorlevel 1 goto failed
echo.
echo  Done.  Start Hypersponge from the Desktop icon or the Start menu.
echo.
choice /c YN /m "Start Hypersponge now"
if errorlevel 2 exit /b 0
start "" "%~dp0Hypersponge.lnk"
exit /b 0
:failed
echo.
echo  Setup failed - see the messages above.
pause
exit /b 1
