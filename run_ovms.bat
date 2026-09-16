@echo off
title OpenVINO Model Server (Port 8000)
cd /d "%~dp0"
echo ======================================================
echo   Starting OpenVINO Model Server (OVMS)...
echo   Port: 8000  (API: http://127.0.0.1:8000/v3)
echo   Model: Qwen3.8-27B-int4-ov  (VLM_CB, GPU)
echo ======================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_ovms.ps1" %*
pause
