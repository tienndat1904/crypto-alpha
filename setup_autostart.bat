@echo off
REM Setup Windows Task Scheduler to auto-start Crypto Alpha Bot on login
REM Run this file as Administrator ONE TIME

echo ============================================
echo   Crypto Alpha Bot - Auto-Start Setup
echo ============================================
echo.

REM Create scheduled task that runs on user logon
schtasks /create /tn "CryptoAlphaBot" /tr "cmd /c \"D:\crypto-alpha\start_bot.bat\"" /sc onlogon /rl highest /f

if "%ERRORLEVEL%"=="0" (
    echo.
    echo [OK] Task scheduled successfully!
    echo     Name: CryptoAlphaBot
    echo     Trigger: On user logon
    echo     Action: Start bot with 30min interval
    echo.
    echo To remove: schtasks /delete /tn "CryptoAlphaBot" /f
    echo To check:  schtasks /query /tn "CryptoAlphaBot"
) else (
    echo.
    echo [ERROR] Failed to create task.
    echo Please run this script as Administrator.
    echo Right-click ^> Run as administrator
)

echo.
pause
