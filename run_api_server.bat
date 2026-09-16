@echo off
title OpenVINO OpenAI-Compatible Server (Port 1234)
cd /d "%~dp0"
echo ======================================================
echo   Starting OpenVINO OpenAI-Compatible API Server...
echo   Port: 1234 (Same as LM Studio API default)
echo   Hardware: Intel Iris Xe Graphics + 64GB RAM
echo   Model: Qwen3.8-27B-int4-ov
echo ======================================================
python openai_server.py
pause
