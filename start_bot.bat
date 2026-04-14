@echo off
REM Crypto Alpha Bot - Auto Start Script
REM Used by Windows Task Scheduler to auto-start bot on login

cd /d D:\crypto-alpha
set PYTHONIOENCODING=utf-8

REM Check if bot is already running
tasklist /FI "WINDOWTITLE eq CryptoAlphaBot" 2>NUL | find /I "cmd.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo Bot is already running. Exiting.
    exit /b 0
)

REM Start bot
title CryptoAlphaBot
echo Starting Crypto Alpha Bot...
echo %date% %time% - Bot starting >> logs\bot_start.log
python -m trading.paper_trader --run --interval 0.5
echo %date% %time% - Bot stopped >> logs\bot_start.log
pause
