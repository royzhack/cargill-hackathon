# Quick Start Guide - Chatbot Web Interface

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements_web.txt
```

### Step 2: Configure API Keys
Edit `chatbot_config.txt` and update:
```
TEAM_API_KEY=your-actual-team-api-key
SHARED_OPENAI_KEY=your-actual-shared-openai-key
```

### Step 3: Start the Server
```bash
python chatbot_app.py
```

Then open **http://localhost:5000** in your browser!

## 🎨 Features

✅ **Professional Claude-like UI** - Clean, modern interface  
✅ **Real-time Chat** - Interactive conversation  
✅ **Auto Visualizations** - Diagrams appear based on your questions  
✅ **Report-Ready Charts** - High-quality 300 DPI images  
✅ **Responsive Design** - Works on all devices  

## 📊 Example Queries

- "What is the best voyage for PACIFIC GLORY?"
- "Compare the top 3 most profitable voyages"
- "Show me the portfolio profit breakdown"
- "Explain the optimization workflow"
- "What are the risk factors?"

## 🖼️ Visualizations

The chatbot automatically shows relevant diagrams:
- System Architecture
- Optimization Workflow  
- ML Risk Simulation Flow
- Portfolio Profit Charts
- Scenario Analysis Flow
- Voyage Comparisons

## 📝 For Your Report

All visualizations are saved in `diagrams/` at 300 DPI - perfect for reports!

## ⚠️ Troubleshooting

**Port in use?** Change port in `chatbot_app.py` (line 150)

**API keys not working?** Check `chatbot_config.txt` has actual keys (not placeholders)

**Visualizations missing?** Run `python generate_diagrams.py` first
