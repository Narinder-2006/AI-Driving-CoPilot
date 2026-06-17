#!/bin/bash
# 🚗 AI Driving Co-Pilot — Startup Script (Linux/Mac)

echo ""
echo "===================================================="
echo " 🚗 AI Driving Co-Pilot - FastAPI Backend"
echo "===================================================="
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "✅ Virtual environment found"
    source venv/bin/activate
else
    echo "⚠️ No virtual environment found"
    echo "Creating venv..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Starting FastAPI backend..."
echo ""
echo "Frontend will be available at: http://localhost:8000"
echo ""

# Start the server
python main.py
