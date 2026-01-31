"""
Chatbot Web Application Backend
================================
Flask backend for the voyage optimization chatbot with visualization support.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import base64
from pathlib import Path
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from voyage_chatbot import VoyageChatbot
from load_env import load_env_file
from generate_diagrams import (
    create_system_architecture_diagram,
    create_optimization_workflow_diagram,
    create_risk_simulation_flow,
    create_portfolio_profit_breakdown,
    create_scenario_analysis_flow
)
try:
    from visualization_generator import (
        create_portfolio_profit_chart,
        create_voyage_comparison_chart
    )
    HAS_ENHANCED_VIZ = True
except ImportError:
    HAS_ENHANCED_VIZ = False
    print("Warning: Enhanced visualization generator not available")

app = Flask(__name__)
CORS(app)

# Initialize chatbot
chatbot = None

def init_chatbot():
    """Initialize the chatbot with API keys from config."""
    global chatbot
    try:
        config = load_env_file("chatbot_config.txt")
        team_key = config.get('TEAM_API_KEY')
        shared_key = config.get('SHARED_OPENAI_KEY')
        model = config.get('CHATBOT_MODEL', 'global.anthropic.claude-sonnet-4-5-20250929-v1:0')
        
        if team_key and shared_key and team_key != "your-team-api-key-here":
            chatbot = VoyageChatbot(
                team_api_key=team_key,
                shared_openai_key=shared_key,
                model=model
            )
            return True
    except Exception as e:
        print(f"Error initializing chatbot: {e}")
    return False

@app.route('/')
def index():
    """Serve the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized. Please check API keys in chatbot_config.txt'}), 500
    
    try:
        data = request.json
        user_message = data.get('message', '')
        
        if not user_message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Get response from chatbot
        response = chatbot.chat(user_message)
        
        # Clean up response (remove any extra whitespace)
        if response:
            response = response.strip()
        
        # Check if response should include visualizations
        visualizations = detect_and_generate_visualizations(user_message, response)
        
        return jsonify({
            'response': response,
            'visualizations': visualizations
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset conversation history."""
    if chatbot is None:
        return jsonify({'error': 'Chatbot not initialized'}), 500
    
    try:
        chatbot.reset_conversation()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualization/<viz_type>', methods=['GET'])
def get_visualization(viz_type):
    """Generate and return a specific visualization."""
    try:
        img_buffer = BytesIO()
        
        if viz_type == 'system_architecture':
            create_system_architecture_diagram()
            img_path = Path('diagrams/system_architecture.png')
        elif viz_type == 'optimization_workflow':
            create_optimization_workflow_diagram()
            img_path = Path('diagrams/optimization_workflow.png')
        elif viz_type == 'risk_simulation':
            create_risk_simulation_flow()
            img_path = Path('diagrams/risk_simulation_flow.png')
        elif viz_type == 'portfolio_profit':
            # Use enhanced visualization generator if available
            if HAS_ENHANCED_VIZ:
                img_path = create_portfolio_profit_chart()
                if img_path is None:
                    # Fallback to original
                    create_portfolio_profit_breakdown()
                    img_path = Path('diagrams/portfolio_profit_breakdown.png')
            else:
                create_portfolio_profit_breakdown()
                img_path = Path('diagrams/portfolio_profit_breakdown.png')
        elif viz_type == 'voyage_comparison':
            if HAS_ENHANCED_VIZ:
                # Extract vessel names from query if possible
                img_path = create_voyage_comparison_chart()
                if img_path is None:
                    return jsonify({'error': 'No data available for comparison'}), 404
            else:
                return jsonify({'error': 'Enhanced visualization generator not available'}), 503
        elif viz_type == 'scenario_analysis':
            create_scenario_analysis_flow()
            img_path = Path('diagrams/scenario_analysis_flow.png')
        else:
            return jsonify({'error': 'Unknown visualization type'}), 400
        
        if img_path.exists():
            return send_file(str(img_path), mimetype='image/png')
        else:
            return jsonify({'error': 'Visualization not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def detect_and_generate_visualizations(user_message: str, response: str) -> list:
    """Detect if user query requires visualizations and generate them."""
    visualizations = []
    message_lower = user_message.lower()
    response_lower = response.lower()
    
    # System architecture
    if any(word in message_lower for word in ['architecture', 'system', 'overview', 'components']):
        visualizations.append({
            'type': 'system_architecture',
            'title': 'System Architecture',
            'url': '/api/visualization/system_architecture'
        })
    
    # Optimization workflow
    if any(word in message_lower for word in ['workflow', 'process', 'optimization', 'steps', 'how does it work']):
        visualizations.append({
            'type': 'optimization_workflow',
            'title': 'Optimization Workflow',
            'url': '/api/visualization/optimization_workflow'
        })
    
    # Risk simulation
    if any(word in message_lower for word in ['risk', 'simulation', 'ml risk', 'uncertainty', 'delays']):
        visualizations.append({
            'type': 'risk_simulation',
            'title': 'ML Risk Simulation Flow',
            'url': '/api/visualization/risk_simulation'
        })
    
    # Portfolio profit
    if any(word in message_lower for word in ['profit', 'portfolio', 'revenue', 'tce', 'financial', 'economics']):
        visualizations.append({
            'type': 'portfolio_profit',
            'title': 'Portfolio Profit Breakdown',
            'url': '/api/visualization/portfolio_profit'
        })
    
    # Voyage comparison
    if any(word in message_lower for word in ['compare', 'comparison', 'versus', 'vs', 'difference']):
        visualizations.append({
            'type': 'voyage_comparison',
            'title': 'Voyage Comparison',
            'url': '/api/visualization/voyage_comparison'
        })
    
    # Scenario analysis
    if any(word in message_lower for word in ['scenario', 'sensitivity', 'threshold', 'robustness', 'what if']):
        visualizations.append({
            'type': 'scenario_analysis',
            'title': 'Scenario Analysis Flow',
            'url': '/api/visualization/scenario_analysis'
        })
    
    return visualizations

@app.route('/api/status', methods=['GET'])
def status():
    """Get chatbot status."""
    return jsonify({
        'initialized': chatbot is not None,
        'has_data': chatbot is not None and chatbot.portfolio_data is not None
    })

if __name__ == '__main__':
    # Ensure diagrams directory exists
    Path('diagrams').mkdir(exist_ok=True)
    
    # Initialize chatbot
    if init_chatbot():
        print("✓ Chatbot initialized successfully")
    else:
        print("⚠ Chatbot not initialized - check chatbot_config.txt")
    
    # Run Flask app
    # Use port 5001 if 5000 is in use (macOS ControlCenter sometimes uses 5000)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 5000
    if sock.connect_ex(('localhost', 5000)) == 0:
        port = 5001
        print(f"⚠ Port 5000 in use, using port {port} instead")
    sock.close()
    app.run(debug=True, host='0.0.0.0', port=port)
