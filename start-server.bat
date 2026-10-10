@echo off
echo Starting Meeting Notes API on port 8811...
start "Meeting Notes Server" cmd /k "& "C:\Users\fishb\AppData\Local\Python\pythoncore-3.14-64\Scripts\meeting-notes.exe" serve"

echo Waiting for server to start...
timeout /t 3 /nobreak >nul

echo Starting tunnel...
echo The public URL will appear below - copy it into the app Settings tab.
echo.
npx localtunnel --port 8811
pause
