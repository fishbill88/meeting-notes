@echo off
echo ============================================
echo  Meeting Notes AI - Start Dev Server
echo ============================================
echo.

set ANDROID_HOME=C:\Users\fishb\AppData\Local\Android\Sdk
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
set PATH=%PATH%;%ANDROID_HOME%\platform-tools

:: Free port 8081 if in use
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8081 " 2^>nul') do (
    taskkill /PID %%a /F >nul 2>&1
)

cd /d "%~dp0mobile"
echo Starting dev server...
echo Open the Meeting Notes AI app on your phone to connect.
echo.
node_modules\.bin\expo.cmd start --dev-client --lan --clear
pause
