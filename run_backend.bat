@echo off
REM 🚗 AI Driving Co-Pilot — Startup Script (Windows)
REM This script starts the FastAPI backend and opens the frontend in browser

echo.
echo ====================================================
echo  🚗 AI Driving Co-Pilot - FastAPI Backend
echo ====================================================
echo.

REM Check if venv exists
if exist venv (
    echo ✅ Virtual environment found
    call venv\Scripts\activate
) else (
    echo ⚠️  No virtual environment found
    echo Creating venv...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install -r requirements.txt
)

echo.
echo 🚀 Starting FastAPI backend...
echo.
echo Frontend will be available at: http://localhost:8000
echo.

REM Start the server
python main.py
