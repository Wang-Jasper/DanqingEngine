@echo off
REM ============================================================
REM Clean CMake cache and rebuild. Run this after moving the project.
REM ============================================================
echo [rebuild] Cleaning build directory...
if exist build rmdir /s /q build
echo [rebuild] Configuring CMake...
cmake -B build -DCMAKE_BUILD_TYPE=Debug
if %ERRORLEVEL% neq 0 (
    echo [rebuild] CMake configure FAILED!
    pause
    exit /b 1
)
echo [rebuild] Building Debug...
cmake --build build --config Debug
if %ERRORLEVEL% neq 0 (
    echo [rebuild] Build FAILED!
    pause
    exit /b 1
)
echo [rebuild] SUCCESS! Executable: build\Debug\Danqing.exe
pause
