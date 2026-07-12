@echo off
setlocal EnableExtensions EnableDelayedExpansion
REM Build alpha-miner for Windows x64 (CPU + OpenCL for AMD/NVIDIA GPUs)
REM Requires: VS 2022 Build Tools, CMake

set ROOT=%~dp0..
cd /d "%ROOT%"

set VCVARS=
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
)
if exist "%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
  set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)
if "%VCVARS%"=="" (
  echo ERROR: vcvars64.bat not found. Install VS 2022 Build Tools with C++.
  exit /b 1
)

call "%VCVARS%"
if errorlevel 1 exit /b 1

set CMAKE=cmake
where cmake >nul 2>&1
if errorlevel 1 (
  if exist "%ProgramFiles%\CMake\bin\cmake.exe" set "CMAKE=%ProgramFiles%\CMake\bin\cmake.exe"
)

echo Using: %CMAKE%
echo.

"%CMAKE%" -S . -B build-win -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release ^
  -DALPHA_MINER_HIP=OFF ^
  -DALPHA_MINER_CUDA=OFF ^
  -DALPHA_MINER_OPENCL=ON ^
  -DOpenCL_INCLUDE_DIR="%ROOT%\third_party\OpenCL\include" ^
  -DOpenCL_LIBRARY="%ROOT%\third_party\OpenCL\lib\OpenCL.lib"
if errorlevel 1 exit /b 1

"%CMAKE%" --build build-win --config Release
if errorlevel 1 exit /b 1

if not exist dist\windows mkdir dist\windows
copy /Y build-win\alpha-miner.exe dist\windows\alpha-miner.exe >nul
copy /Y kernels\blake3_an.cl dist\windows\blake3_an.cl >nul
copy /Y scripts\mine-rplant.bat dist\windows\mine-rplant.bat >nul
copy /Y README.md dist\windows\README.txt >nul 2>nul

echo.
echo === BUILD OK ===
echo Binary: %ROOT%\dist\windows\alpha-miner.exe
echo.
echo Example:
echo   dist\windows\alpha-miner.exe -o eu.rplant.xyz:7176 -u YOUR_WALLET.rig1 -b opencl -k blake3_an.cl
echo.
endlocal
