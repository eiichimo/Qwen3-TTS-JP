@echo off
chcp 65001 >nul
title Qwen3-TTS-JP
cd /d "%~dp0"

echo ============================================================
echo   Qwen3-TTS-JP  -  Text-to-Speech
echo ============================================================
echo.
echo   [1] Base Model (Voice Clone / Voice Design / Custom Voice)
echo   [2] CustomVoice Model (Preset Speakers Only)
echo.
echo   * Base Model is recommended (all features available)
echo ============================================================
echo.
set /p choice="Select model [1/2] (default: 1): "

if "%choice%"=="2" (
    echo.
    echo Starting with CustomVoice model...
    .venv\Scripts\python.exe launch_gui.py --custom-voice
) else (
    echo.
    echo Starting with Base model...
    .venv\Scripts\python.exe launch_gui.py
)

echo.
echo ============================================================
echo   Server stopped. Press any key to close.
echo ============================================================
pause >nul
