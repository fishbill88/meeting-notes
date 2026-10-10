@echo off
setlocal enabledelayedexpansion
title Meeting Notes AI - Desktop

echo ============================================
echo  Meeting Notes AI - Desktop
echo ============================================
echo.

cd /d "%~dp0"

set "APP=%~dp0desktop\src-tauri\target\release\meeting-notes-desktop.exe"
set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"

REM --- 1. Is the app built? -------------------------------------------------
if not exist "%APP%" (
    echo [X] The desktop app has not been built yet.
    echo.
    echo     Expected: %APP%
    echo.
    echo     Build it with:
    echo         python scripts\build_sidecar.py
    echo         cd desktop ^&^& npm run tauri build
    echo.
    pause
    exit /b 1
)

REM --- 2. Ollama ------------------------------------------------------------
REM Without Ollama the notes silently fall back to a much weaker extractive
REM summariser. No error is shown, so check it here rather than let it
REM surprise you mid-meeting.
echo Checking Ollama...
curl -s -o nul -m 3 http://localhost:11434/api/tags
if errorlevel 1 (
    if exist "%OLLAMA%" (
        echo   Not running - starting it...
        start "" /min "%OLLAMA%" serve
        REM Give the server a moment to bind before the app asks for models.
        timeout /t 4 /nobreak >nul
        curl -s -o nul -m 3 http://localhost:11434/api/tags
        if errorlevel 1 (
            echo   [!] Still not responding. Notes will use the basic
            echo       summariser until Ollama is up.
        ) else (
            echo   [OK] Ollama started.
        )
    ) else (
        echo   [!] Ollama is not installed. Notes will use the basic
        echo       summariser. Install from https://ollama.com then run:
        echo           ollama pull llama3.2
    )
) else (
    echo   [OK] Ollama is already running.
)

REM --- 3. Speech model cached? ---------------------------------------------
REM large-v3-turbo is ~1.55 GB and downloads on first transcription. On a slow
REM connection that is a long, silent wait - better to know now.
echo Checking speech model...
if exist "%USERPROFILE%\.cache\huggingface\hub\models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" (
    echo   [OK] large-v3-turbo is cached.
) else (
    echo   [!] large-v3-turbo is NOT cached ^(~1.55 GB^).
    echo       The first transcription will download it, which can take a
    echo       while. Do that before a meeting, not during one.
)

REM --- 4. Launch ------------------------------------------------------------
echo.
echo Starting Meeting Notes AI...
echo   ^(First launch loads the speech model - allow a few seconds.^)
echo.
start "" "%APP%"

echo Launched. This window can be closed.
timeout /t 5 /nobreak >nul
endlocal
