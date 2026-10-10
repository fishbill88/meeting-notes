@echo off
echo ============================================
echo  Meeting Notes AI - EAS Build
echo ============================================
echo.

cd /d "%~dp0mobile"

echo Step 1: Checking EAS CLI...
call npx eas-cli --version >nul 2>&1
if errorlevel 1 (
    echo Installing EAS CLI...
    call npm install -g eas-cli
)

echo.
echo Step 2: Login to Expo (browser will open)...
call npx eas-cli login

echo.
echo Step 3: Configure EAS project (first time only)...
call npx eas-cli build:configure

echo.
echo Step 4: Building Android APK (preview)...
echo This uploads to EAS cloud and takes 5-10 minutes.
echo You will get a download link when done.
echo.
call npx eas-cli build --platform android --profile preview

echo.
echo ============================================
echo  Build submitted! Check your Expo dashboard
echo  or the link above to download the APK.
echo ============================================
pause
