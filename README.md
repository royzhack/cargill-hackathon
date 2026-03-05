# Cargill Ocean Transportation - Voyage Optimization System

A comprehensive freight calculator and voyage optimization model for Capesize vessels with ML-based risk simulation, scenario analysis, and an interactive chatbot interface.

## 🎯 Project Overview

This system optimizes vessel-cargo assignments to maximize total portfolio profit while ensuring all committed cargoes are lifted. The solution integrates:

- **Multi-leg Routing Optimization**: CP-SAT solver for optimal vessel-cargo assignments
- **ML-based Risk Simulation**: Probabilistic modeling of weather delays, port congestion, and voyage uncertainty
- **Scenario Analysis**: Robustness testing with threshold detection for port delays and fuel price changes
- **Interactive Chatbot**: Web-based interface with natural language queries and automatic visualizations
- **Map Visualization**: Interactive maps showing vessel routes, cargo locations, and port markers

## 📊 Key Results

- **Total Portfolio Profit**: $8.01M (risk-adjusted)
- **Committed Cargoes**: 3/3 lifted (100% coverage)
- **Market Cargoes**: 4 additional cargoes assigned
- **Total Assignments**: 7 voyage legs
- **Test Coverage**: 133 tests, 100% pass rate

## 🗂️ Repository Structure

```
├── data/                          # Input data files
│   ├── Cargill_Capesize_Vessels.csv
│   ├── Market_Vessels_Formatted.csv
│   ├── Cargill_Committed_Cargoes_Structured.csv
│   ├── Market_Cargoes_Structured.csv
│   ├── Port Distances.csv
│   ├── port_locations.csv
│   ├── bunker_forward_curve.csv
│   └── freight_rates.csv
├── processed/                     # Processed data and outputs
│   ├── portfolio_summary.json
│   └── portfolio_summary_risk_adjusted.json
├── diagrams/                      # Generated diagrams and visualizations
├── templates/                     # Web interface templates
│   └── index.html
├── static/                        # Web interface assets
│   ├── css/style.css
│   └── js/app.js
├── vessel_cargo_optimization_multileg_v1.ipynb  # Main optimization notebook
├── chatbot_app.py                 # Flask web application
├── voyage_chatbot.py              # Chatbot logic
├── ml_risk_simulation.py          # ML risk simulation module
├── scenario_analysis.py            # Scenario analysis module
├── map_generator.py                # Map visualization generator
├── generate_diagrams.py            # Diagram generation
├── visualization_generator.py     # Enhanced visualizations
├── freight_calculator.py          # Freight calculation utilities
├── test_optimization.py            # Comprehensive test suite
├── test_chatbot.py                # Chatbot tests
├── requirements_web.txt           # Python dependencies
├── FREIGHT_CALCULATOR_DOCUMENTATION.md  # Complete documentation
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Jupyter Notebook
- API keys for Amazon Bedrock (for chatbot)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cargill-hackathon
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements_web.txt
   ```

3. **Configure API keys**:
   Create `chatbot_config.txt` in the root directory:
   ```
   TEAM_API_KEY=your-team-api-key
   SHARED_OPENAI_KEY=your-shared-openai-key
   CHATBOT_MODEL=global.anthropic.claude-sonnet-4-5-20250929-v1:0
   ```

### Running the Optimization

1. **Open the main notebook**:
   ```bash
   jupyter notebook vessel_cargo_optimization_multileg_v1.ipynb
   ```
   
   **Note**: You can also add your final notebook with a different name - the system will work with any notebook that follows the same structure.

2. **Run all cells** to:
   - Load and process input data
   - Generate feasible arcs with ML risk simulation
   - Solve the optimization problem
   - Generate outputs and visualizations

3. **Outputs generated**:
   - `processed/portfolio_summary.json` - Portfolio summary
   - `processed/portfolio_summary_risk_adjusted.json` - Risk-adjusted results
   - `diagrams/` - System diagrams and visualizations

### Running the Web Application

1. **Start the Flask server**:
   ```bash
   python chatbot_app.py
   ```

2. **Open in browser**:
   Navigate to `http://localhost:5000` (or `http://localhost:5001` if 5000 is in use)

3. **Interact with the chatbot**:
   - Ask questions about vessel assignments, profits, TCE
   - Request visualizations: "Show me the routes on a map"
   - Compare voyages: "Compare the top 3 most profitable voyages"

## 📋 Core Features

### 1. Optimization Model

**CP-SAT Solver** for exact optimization:
- Multi-leg voyage support
- Constraint satisfaction (all committed cargoes lifted)
- Portfolio profit maximization
- Market vessel assignment for committed cargoes only

**Key Constraints**:
- Each committed cargo must be assigned exactly once
- Vessels can execute multiple sequential legs
- Market vessels only for committed cargoes
- Time window constraints (laycan dates)

### 2. ML Risk Simulation

**Evidence-based probabilistic modeling**:
- **Weather Delays**: Seasonal and route-specific risk factors
- **Port Congestion**: Port-specific probability (20-30% industry average)
- **Voyage Uncertainty**: 5-10% coefficient of variation
- **Demurrage Risk**: $20k-$30k/day, 10-20% occurrence probability

**Integration**: Risk simulation runs **before** optimization to ensure risk-adjusted profits are used in decision-making.

### 3. Scenario Analysis

**Robustness testing** with threshold detection:
- **Port Delay Scenario**: Identifies minimum delay (days) at Chinese ports where optimal solution changes
- **Bunker Price Scenario**: Identifies fuel price increase (%) threshold for solution switch
- **Binary Search**: Efficient threshold detection
- **Economic Intuition**: Clear explanations of why recommendations change

### 4. Interactive Chatbot

**Natural language interface**:
- Professional shipping analyst persona
- Automatic visualization generation
- Map-based route visualization
- Voyage comparison and trade-off analysis

**Example Queries**:
- "What is the best voyage for PACIFIC GLORY?"
- "Show me the routes on a map"
- "Compare the top 3 most profitable voyages"
- "Explain the risk factors affecting the portfolio"

### 5. Map Visualization

**Interactive Folium maps** showing:
- Vessel routes (colored lines per vessel)
- Load ports (blue for Committed, orange for Market)
- Discharge ports (red markers)
- Cargo details in popups (ID, quantity, profit, TCE)

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Main optimization tests
python test_optimization.py

# Chatbot tests
python test_chatbot.py
```

**Test Coverage**:
- Data integrity and validation
- Profit and TCE calculations
- Constraint satisfaction
- ML risk simulation
- Optimality verification
- 133 tests total, 100% pass rate

## 📖 Documentation

### Main Documentation

**`FREIGHT_CALCULATOR_DOCUMENTATION.md`** contains:
- Complete system architecture
- Detailed optimization methodology
- ML risk simulation parameters and evidence
- Scenario analysis methodology
- Testing and validation results
- Exact numerical examples from the solution

### Key Sections

1. **Input Data**: Exact column names and data structure
2. **Methodology**: Step-by-step calculations
3. **Optimization Model**: CP-SAT formulation details
4. **ML Risk Simulation**: Evidence-based parameters
5. **Scenario Analysis**: Threshold detection methodology
6. **Results**: Portfolio summary with exact numbers

## 🔧 Technical Details

### Optimization Solver
- **Tool**: Google OR-Tools CP-SAT
- **Type**: Constraint Programming with Boolean Satisfiability
- **Objective**: Maximize total portfolio profit (risk-adjusted)
- **Constraints**: Cargo coverage, time windows, vessel capacity

### ML Risk Models
- **Weather**: Seasonal multipliers, route-specific factors
- **Congestion**: Port-specific probabilities (22% base, 10-32% by port)
- **Uncertainty**: 7% coefficient of variation for voyage duration
- **Demurrage**: $25k/day base rate, 15% occurrence probability

### Data Sources
- **Port Distances**: `Port Distances.csv` + searoute library for missing routes
- **Bunker Prices**: `bunker_forward_curve.csv` with port-to-hub mapping
- **Freight Rates**: `freight_rates.csv` (FFA 5TC rates)
- **Port Locations**: `port_locations.csv` (latitude/longitude)

## 📊 Output Files

### Generated by Optimization
- `processed/portfolio_summary.json` - Base portfolio results
- `processed/portfolio_summary_risk_adjusted.json` - Risk-adjusted results

### Generated by Visualization
- `diagrams/system_architecture.png` - System design
- `diagrams/optimization_workflow.png` - Process flow
- `diagrams/risk_simulation_flow.png` - Risk modeling flow
- `diagrams/portfolio_profit_breakdown.png` - Financial charts
- `diagrams/scenario_analysis_flow.png` - Scenario methodology
- `diagrams/vessel_routes_map.html` - Interactive map

## 🎓 Usage Examples

### Example 1: Run Optimization
```python
# Open vessel_cargo_optimization_multileg.ipynb
# Run all cells sequentially
# Results saved to processed/portfolio_summary.json
```

### Example 2: Query Chatbot
```
User: "What is the best voyage for PACIFIC GLORY?"
Bot: "PACIFIC GLORY → MARKET_5 delivers $59,876/day TCE, best in portfolio. 
     Short voyage (37 days) with manageable fuel exposure."
```

### Example 3: View Map
```
User: "Show me the routes on a map"
Bot: [Displays interactive map with all vessel routes and cargo locations]
```

## 🔐 Configuration

### API Keys
Set in `chatbot_config.txt` (not committed to git):
```
TEAM_API_KEY=your-actual-key
SHARED_OPENAI_KEY=your-actual-key
CHATBOT_MODEL=global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### Environment Variables (Optional)
For production deployment, can use environment variables:
```bash
export TEAM_API_KEY=your-key
export SHARED_OPENAI_KEY=your-key
```

## 🐛 Troubleshooting

### Chatbot not initializing
- Check `chatbot_config.txt` exists and has valid API keys
- Verify API keys are correct (no quotes, no spaces)
- Check API key format matches requirements

### Map not loading
- Ensure `folium` is installed: `pip install folium`
- Verify `data/port_locations.csv` exists
- Check `processed/portfolio_summary.json` exists

### Optimization errors
- Ensure all input CSV files are in `data/` directory
- Check that port names match between files
- Verify Python version is 3.12+

### Visualization errors
- Install matplotlib: `pip install matplotlib`
- Ensure `diagrams/` directory exists
- Check that processed data files are available

## 📝 Dependencies

See `requirements_web.txt` for complete list. Key dependencies:
- `flask>=3.0.0` - Web framework
- `pandas>=2.0.0` - Data processing
- `ortools>=9.0.0` - Optimization solver
- `folium>=0.14.0` - Map visualization
- `matplotlib>=3.7.0` - Chart generation
- `openai>=1.0.0` - Chatbot API client

## 🎯 Project Highlights

1. **Evidence-Based ML Models**: All risk parameters backed by industry benchmarks
2. **Pre-Optimization Risk Integration**: Risk simulation runs before optimization
3. **Comprehensive Testing**: 133 tests covering all components
4. **Professional Documentation**: Detailed with exact numerical examples
5. **Interactive Interface**: Web-based chatbot with automatic visualizations
6. **Scenario Analysis**: Structured robustness testing with threshold detection

## 📞 Support

For questions or issues:
1. Check `FREIGHT_CALCULATOR_DOCUMENTATION.md` for detailed explanations
2. Review test files for usage examples
3. Check that all required data files are present

## 📄 License

Part of the Cargill Ocean Transportation Hackathon project.

---

**Last Updated**: 2024  
**Version**: 1.0  
**Status**: Production Ready
