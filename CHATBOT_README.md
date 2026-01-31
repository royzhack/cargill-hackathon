# Voyage Optimization Chatbot

A conversational AI assistant that interfaces with voyage optimization outputs to provide natural-language recommendations and analysis for commercial shipping decisions.

## Overview

The chatbot consumes structured optimization outputs (JSON files) and provides professional freight analytics advice. It does **not** run optimization itself—it analyzes and explains existing optimization results.

## Features

### Minimum Required Capabilities
- ✅ Accepts structured optimization outputs as input
- ✅ Generates natural-language voyage recommendations
- ✅ Provides vessel-cargo pairing, expected TCE, and expected profit
- ✅ Professional AI assistant style for commercial shipping decisions

### Higher Achievement Features
- ✅ **Compare multiple voyage options** side-by-side
- ✅ **Explain trade-offs** between alternatives (fuel exposure, delays, ballast distance, congestion risk)
- ✅ **Summarize scenario analysis results**, including threshold-based switches
- ✅ **Explain why recommendations change** using economic intuition

## Architecture

The chatbot is implemented as a **stateful conversational agent** that maintains context across turns. Key components:

1. **API Client Setup**: OpenAI-compatible client via Amazon Bedrock proxy
2. **Prompt/System Message Construction**: Professional freight analytics advisor persona
3. **Conversation Handling**: Maintains conversation history for context
4. **Model Output Injection**: Dynamically injects relevant optimization data into queries

## Setup

### Prerequisites

```bash
pip install openai
```

Or install in the virtual environment:

```bash
./venv/bin/pip install openai
```

### Environment Variables

Set the following environment variables:

```bash
export TEAM_API_KEY="your-team-api-key"
export SHARED_OPENAI_KEY="your-shared-openai-key"
```

### Configuration

The chatbot uses the following Bedrock proxy configuration:
- **Base URL**: `https://3dbuidg8rf.execute-api.us-east-1.amazonaws.com/api/v1`
- **Team API Key**: Passed via `x-api-key` header
- **Shared OpenAI Key**: Used as API key

### Allowed Models

- `global.anthropic.claude-sonnet-4-5-20250929-v1:0` (default)
- `global.anthropic.claude-haiku-4-5-20251001-v1:0`
- `mistral.mistral-large-3-675b-instruct`
- `mistral.mistral-small-2402-v1:0`
- `openai.gpt-oss-120b-1:0`
- `global.amazon.nova-2-lite-v1:0`

## Usage

### Basic Usage

```python
from voyage_chatbot import VoyageChatbot
import os

# Initialize chatbot
chatbot = VoyageChatbot(
    team_api_key=os.getenv("TEAM_API_KEY"),
    shared_openai_key=os.getenv("SHARED_OPENAI_KEY"),
    model="global.anthropic.claude-sonnet-4-5-20250929-v1:0"
)

# Ask a question
response = chatbot.chat("What is the best voyage assignment for PACIFIC GLORY?")
print(response)
```

### Interactive Session

```python
# Start conversation
chatbot = VoyageChatbot(
    team_api_key=os.getenv("TEAM_API_KEY"),
    shared_openai_key=os.getenv("SHARED_OPENAI_KEY")
)

# Interactive loop
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        break
    
    response = chatbot.chat(user_input)
    print(f"\nAssistant: {response}")

# Reset conversation if needed
chatbot.reset_conversation()
```

## Input Data

The chatbot expects the following JSON files in the `processed/` directory:

1. **`portfolio_summary.json`**: Base optimization results
   - Portfolio-level metrics (total profit, vessel counts)
   - Assignment details (vessel, cargo, route, profit, TCE, costs)
   - Voyage metrics (days, distances, fuel costs)

2. **`portfolio_summary_risk_adjusted.json`**: Risk-adjusted results
   - Risk-adjusted profit and TCE
   - Risk metrics (delays, fuel adjustments, demurrage)
   - Comparison between base and risk-adjusted values

### Data Structure

The chatbot automatically loads and processes these files. The structure should match the output from the voyage optimization system:

```json
{
  "total_portfolio_profit": 8011706.53,
  "assignments": [
    {
      "Vessel_Name": "PACIFIC GLORY",
      "Cargo_ID": "MARKET_5",
      "Leg_Profit": 2204884.65,
      "TCE_Leg": 59876.03,
      "Load_Port": "VANCOUVER (CANADA)",
      "Discharge_Port": "FANGCHENG",
      ...
    }
  ]
}
```

## Example Queries

### Minimum Requirements

**1. Basic Voyage Recommendation**
```
User: What is the best voyage assignment for PACIFIC GLORY?

Assistant: Based on current freight rates and congestion-adjusted voyage durations, 
PACIFIC GLORY should carry MARKET_5 (Vancouver → Fangcheng), delivering the highest 
expected TCE of $59,876/day with a profit of $2,204,885. This assignment maximizes 
profitability due to the high freight rate ($25/MT) and relatively short voyage duration 
(36.8 days).
```

**2. Expected TCE and Profit**
```
User: What is the expected TCE and profit for ANN BELL?

Assistant: ANN BELL is assigned to MARKET_6 (Kamsar Anchorage → New Mangalore) with:
- Expected Profit: $1,451,754
- Expected TCE: $19,926/day
- Voyage Duration: 72.9 days
- Revenue: $4,025,000 (175,000 MT @ $23/MT)
```

### Higher Achievement Features

**3. Compare Multiple Options**
```
User: Compare the top 3 most profitable voyages side-by-side

Assistant: Here's a side-by-side comparison of the top 3 most profitable voyages:

1. PACIFIC GLORY → MARKET_5 (Vancouver → Fangcheng)
   - Profit: $2,204,885 | TCE: $59,876/day
   - Key Advantage: Highest TCE, short voyage (36.8 days)
   - Trade-off: Higher fuel costs due to transpacific route

2. IRON CENTURY → CARGILL_1 (Kamsar → Qingdao)
   - Profit: $1,872,323 | TCE: $29,385/day
   - Key Advantage: High profit, committed cargo
   - Trade-off: Longer voyage (63.7 days), higher hire costs

3. ANN BELL → MARKET_6 (Kamsar → New Mangalore)
   - Profit: $1,451,754 | TCE: $19,926/day
   - Key Advantage: Good profit for market cargo
   - Trade-off: Long voyage (72.9 days), significant waiting time (7 days)
```

**4. Explain Trade-offs**
```
User: What are the trade-offs between ANN BELL and PACIFIC GLORY assignments?

Assistant: Comparing ANN BELL and PACIFIC GLORY assignments reveals key trade-offs:

**ANN BELL → MARKET_6:**
- Lower TCE ($19,926/day vs $59,876/day) but longer voyage (72.9 days)
- Higher fuel exposure: $1,466,542 total fuel cost
- Significant waiting time: 7 days at load port
- Longer ballast distance: 11,124 NM

**PACIFIC GLORY → MARKET_5:**
- Much higher TCE ($59,876/day) with shorter voyage (36.8 days)
- Lower fuel exposure: $810,118 total fuel cost
- No waiting time at load port
- Shorter ballast distance: 4,509 NM

**Economic Intuition:** PACIFIC GLORY's assignment is superior because the shorter voyage 
duration and higher freight rate ($25/MT vs $23/MT) generate significantly more profit per day, 
despite the transpacific route's fuel costs.
```

**5. Risk-Adjusted Analysis**
```
User: How does risk adjustment affect the portfolio profit?

Assistant: Risk adjustment reduces the portfolio profit by $98,191 (1.23%) from $8,011,707 
to $7,913,516. The main risk factors are:

1. **Total Risk Delays**: 74.4 days across all voyages
   - Average weather delay: 0.65 days per voyage
   - Average congestion delay: 1.83 days per voyage
   - Average waiting days risk: 8.14 days per voyage

2. **Fuel Adjustments**: Average +1.22% increase due to delays

3. **Impact by Voyage:**
   - ANN BELL → MARKET_6: Largest impact (-$62,351) due to 24.9 days of delays
   - PACIFIC GLORY → MARKET_5: Minimal impact (-$92,400) despite high fuel adjustment
   - IRON CENTURY → CARGILL_1: Small impact (-$7,742) with 6.7 days of delays

The risk-adjusted average TCE drops from $25,285/day to $21,148/day, reflecting the 
operational uncertainties in voyage execution.
```

**6. Economic Intuition**
```
User: Why is PACIFIC GLORY assigned to MARKET_5 instead of a committed cargo?

Assistant: PACIFIC GLORY is assigned to MARKET_5 (a market cargo) instead of a committed 
cargo because this maximizes total portfolio profit. Here's the economic reasoning:

**If PACIFIC GLORY took a committed cargo:**
- Would need to use a market vessel (e.g., IRON CENTURY) for MARKET_5
- Market vessel hire rate ($18,454/day) is higher than Cargill vessel rate ($14,800/day)
- Total portfolio profit would be lower

**Current assignment (optimal):**
- PACIFIC GLORY → MARKET_5: Profit $2,204,885, TCE $59,876/day
- IRON CENTURY → CARGILL_1: Profit $1,872,323, TCE $29,385/day
- Combined: $4,077,208

**Alternative (suboptimal):**
- PACIFIC GLORY → CARGILL_1: Estimated profit ~$1,200,000
- Market vessel → MARKET_5: Lower profit due to higher hire costs
- Combined: ~$3,000,000 (estimated)

The optimizer correctly identifies that using Cargill vessels for high-TCE market cargoes 
and market vessels for committed cargoes maximizes total portfolio value.
```

## Design Decisions

### Stateful Conversation
- Maintains conversation history across turns
- Allows follow-up questions and context-aware responses
- Can be reset with `chatbot.reset_conversation()`

### Dynamic Data Injection
- Automatically injects relevant assignment data when user queries mention specific vessels/cargoes
- Adds risk-adjusted data when queries mention risk, delays, or congestion
- Keeps system prompt concise while providing detailed data on-demand

### Professional Persona
- System prompt establishes assistant as "professional freight analytics advisor"
- Uses shipping terminology appropriately
- Focuses on decision-relevant information
- Explains economic drivers clearly

### Safety & Clarity
- Never exposes API keys in responses
- Never mentions Bedrock, OpenAI, or implementation details
- Never hallucinates numbers—strictly uses provided optimization outputs
- Validates data availability before processing

## Error Handling

The chatbot handles:
- Missing data files (raises FileNotFoundError on initialization)
- Invalid model selection (raises ValueError)
- API errors (returns user-friendly error message)
- Missing environment variables (warns in main() example)

## Testing

Run the example script to test:

```bash
python voyage_chatbot.py
```

This will demonstrate:
1. Basic voyage recommendation
2. Comparison of multiple options
3. Trade-off explanation
4. Risk-adjusted analysis
5. Economic intuition explanation

## Production Considerations

1. **API Key Management**: Use secure secret management (AWS Secrets Manager, etc.)
2. **Rate Limiting**: Implement rate limiting to respect API usage limits
3. **Caching**: Cache responses for common queries to reduce API calls
4. **Logging**: Log all queries and responses for audit and improvement
5. **Monitoring**: Monitor API usage and costs
6. **Error Recovery**: Implement retry logic for transient API failures

## License

Part of the Cargill Ocean Transportation Voyage Optimization System.

