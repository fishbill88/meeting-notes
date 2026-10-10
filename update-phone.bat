@echo off
echo ============================================
echo  Meeting Notes AI - Update Phone
echo ============================================
echo.

:: Set environment variables
set ANDROID_HOME=C:\Users\fishb\AppData\Local\Android\Sdk
set JAVA_HOME=C:\Program Files\Android\Android Studio\jbr
set PATH=%PATH%;%ANDROID_HOME%\platform-tools

:: Check device is connected
echo Checking for connected device...
adb devices
echo.

:: Kill any process on port 8081
echo Freeing port 8081...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8081 "') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: Build and install
echo Building and installing on phone (this takes ~5 min)...
echo.
cd /d "%~dp0mobile"
node_modules\.bin\expo.cmd run:android

echo.
echo ============================================
echo  Done! App updated on your phone.
echo  Run start-expo.bat to start the dev server.
echo ============================================
pause
