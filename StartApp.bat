@echo off
chcp 65001 >nul
cd /d "%~dp0"

set "CODEX_CMD=%APPDATA%\npm\codex.cmd"

if not exist "%CODEX_CMD%" (
    echo.
    echo ❌ 오류: codex.cmd 파일을 찾을 수 없습니다.
    echo PowerShell 에서 'npm install -g @openai/codex' 를 먼저 실행해주세요.
    pause
    exit
)

echo.
echo 🚀 Codex 세션을 복원합니다...
echo.

start wt.exe -d "%~dp0" powershell -NoExit -Command "& '%CODEX_CMD%' resume --last"
exit