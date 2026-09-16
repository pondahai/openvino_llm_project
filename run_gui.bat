@echo off
title OpenVINO LLM Web GUI Launcher
cd /d "%~dp0"
echo ======================================================
echo   Starting OpenVINO LLM Web GUI...
echo   Hardware: Intel Iris Xe Graphics + 64GB RAM
echo ======================================================
start http://localhost:8501
python -m streamlit run app.py --server.port 8501
pause
