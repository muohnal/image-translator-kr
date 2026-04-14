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

python -m streamlit run app.py
pause