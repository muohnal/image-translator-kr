@echo off
title Image Translation App
chcp 65001 >nul
cd /d "%~dp0"

echo 🚀 Streamlit 앱을 시작합니다...
echo.
echo 브라우저에서 http://localhost:8501로 접속하세요
echo.
echo 앱을 종료하려면 이 창에서 Ctrl+C를 누르세요
echo.

where tesseract >nul 2>nul
if errorlevel 1 (
    echo ⚠️ Tesseract OCR을 PATH에서 찾을 수 없습니다.
    echo    README.md의 설치 안내를 확인하거나 앱 사이드바에서 실행 파일 경로를 지정하세요.
    echo.
)

python -m streamlit run app.py
pause