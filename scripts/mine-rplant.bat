@echo off
REM Mine ALPHA (blake3-an) on rplant with OpenCL (AMD/NVIDIA) or CPU fallback.
setlocal
set ROOT=%~dp0
set BIN=%ROOT%alpha-miner.exe
if not exist "%BIN%" set BIN=%ROOT%..\build-win\alpha-miner.exe

if "%~1"=="" (
  echo Usage: mine-rplant.bat YOUR_ALPHA_WALLET [worker] [region]
  echo   region: eu ^| na ^| asia  (default eu^)
  exit /b 1
)
set WALLET=%~1
set WORKER=%~2
if "%WORKER%"=="" set WORKER=rig1
set REGION=%~3
if "%REGION%"=="" set REGION=eu

"%BIN%" -o %REGION%.rplant.xyz:7176 -u %WALLET%.%WORKER% -p x -b auto -k "%ROOT%blake3_an.cl"
endlocal
