# Voyage Optimization Chatbot - Web Interface

A professional web-based interface for the Voyage Optimization Chatbot with Claude-like UI and dynamic visualization capabilities.

## Features

- **Professional UI**: Clean, modern interface inspired by Claude's design
- **Real-time Chat**: Interactive conversation with the voyage optimization assistant
- **Dynamic Visualizations**: Automatically generates and displays relevant diagrams based on user queries
- **Report-Ready Charts**: High-quality visualizations suitable for project reports
- **Responsive Design**: Works on desktop and mobile devices

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Configure API Keys**:
   - Open `chatbot_config.txt`
   - Update `TEAM_API_KEY` and `SHARED_OPENAI_KEY` with your actual keys

3. **Generate Initial Diagrams** (optional):
   ```bash
   python generate_diagrams.py
   python visualization_generator.py
   ```

## Running the Application

1. **Start the Flask Server**:
   ```bash
   python chatbot_app.py
   ```

2. **Open in Browser**:
   - Navigate to `http://localhost:5000`
   - The chat interface will load automatically

## Usage

### Basic Chat
- Type your question in the input box
- Press Enter or click the send button
- The assistant will respond with analysis and recommendations

### Example Queries
- "What is the best voyage for PACIFIC GLORY?"
- "Compare the top 3 most profitable voyages"
- "Show me the portfolio profit breakdown"
- "Explain the optimization workflow"
- "What are the risk factors affecting the portfolio?"

### Visualizations
The chatbot automatically detects when visualizations would be helpful and displays them inline with responses. Visualizations include:

- **System Architecture**: Overall system design
- **Optimization Workflow**: Step-by-step optimization process
- **ML Risk Simulation Flow**: Risk modeling process
- **Portfolio Profit Breakdown**: Financial analysis charts
- **Scenario Analysis Flow**: Sensitivity analysis methodology
- **Voyage Comparison**: Side-by-side voyage metrics

### Reset Conversation
Click the "New Chat" button in the sidebar to start a fresh conversation.

## Architecture

### Backend (`chatbot_app.py`)
- Flask web server
- RESTful API endpoints
- Chatbot integration
- Visualization generation
- File serving

### Frontend
- **HTML** (`templates/index.html`): Main chat interface
- **CSS** (`static/css/style.css`): Professional styling
- **JavaScript** (`static/js/app.js`): Interactive functionality

### Visualization Generator (`visualization_generator.py`)
- Enhanced chart generation
- Report-ready formatting
- Dynamic data loading
- High-resolution output (300 DPI)

## API Endpoints

- `GET /` - Main chat interface
- `POST /api/chat` - Send chat message
- `POST /api/reset` - Reset conversation
- `GET /api/status` - Check chatbot status
- `GET /api/visualization/<type>` - Get visualization image

## Customization

### Adding New Visualizations
1. Create visualization function in `visualization_generator.py`
2. Add detection logic in `detect_and_generate_visualizations()` in `chatbot_app.py`
3. Add route handler in `chatbot_app.py`

### Styling
Modify `static/css/style.css` to customize colors, fonts, and layout.

## Troubleshooting

**Chatbot not initialized**:
- Check that `chatbot_config.txt` has valid API keys
- Verify API keys are correct and not placeholders

**Visualizations not loading**:
- Ensure `diagrams/` directory exists
- Run `python generate_diagrams.py` to generate base diagrams
- Check browser console for errors

**Port already in use**:
- Change port in `chatbot_app.py`: `app.run(port=5001)`

## Production Deployment

For production deployment:

1. Use a production WSGI server (e.g., Gunicorn):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 chatbot_app:app
   ```

2. Set up reverse proxy (nginx) for static files

3. Configure environment variables for API keys instead of config file

4. Enable HTTPS with SSL certificates

## License

Part of the Cargill Ocean Transportation Voyage Optimization System.
