#!/bin/bash
# Quick start script for the chatbot web interface

echo "=========================================="
echo "Voyage Optimization Chatbot - Web Interface"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "Installing Flask and dependencies..."
    pip install -r requirements_web.txt
fi

# Check if diagrams directory exists
if [ ! -d "diagrams" ]; then
    echo "Creating diagrams directory..."
    mkdir -p diagrams
fi

# Generate diagrams if they don't exist
if [ ! -f "diagrams/system_architecture.png" ]; then
    echo "Generating base diagrams..."
    python3 generate_diagrams.py 2>/dev/null || echo "Note: Some diagrams may not be available"
fi

# Check API keys
if grep -q "your-team-api-key-here" chatbot_config.txt 2>/dev/null; then
    echo ""
    echo "⚠️  WARNING: Please update chatbot_config.txt with your API keys!"
    echo ""
fi

echo "Starting Flask server..."
echo "Open http://localhost:5000 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 chatbot_app.py
