@echo off
title OpenVINO OpenAI-Compatible Server (Port 1234)
cd /d "%~dp0"
echo ======================================================
echo   Starting OpenVINO OpenAI-Compatible API Server...
echo   Port: 1234 (Same as LM Studio API default)
echo   Hardware: Intel Iris Xe Graphics + 64GB RAM
echo   Model: %~1 (default qwen3.8-27b; or qwen3.6-35b-a3b)
echo ======================================================
if not "%~1"=="" set LLM_MODEL=%~1
python openai_server.py
pause
