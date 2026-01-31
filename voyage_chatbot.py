"""
Voyage Optimization Chatbot Agent
==================================
A conversational AI assistant that interfaces with voyage optimization outputs
to provide natural-language recommendations and analysis for commercial shipping decisions.

Uses OpenAI-compatible client via Amazon Bedrock proxy.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from openai import OpenAI
from datetime import datetime

# Try to load from config file or .env file
try:
    from load_env import setup_env_from_file
    setup_env_from_file(".env")
except ImportError:
    pass
except Exception:
    pass


class VoyageChatbot:
    """
    Stateful conversational agent for voyage optimization analysis.
    
    Maintains conversation context and provides professional freight analytics advice
    based on structured optimization outputs.
    """
    
    def __init__(
        self,
        team_api_key: str,
        shared_openai_key: str,
        model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        base_url: str = "https://3dbuidg8rf.execute-api.us-east-1.amazonaws.com/api/v1",
        portfolio_data_path: Optional[str] = None,
        risk_adjusted_data_path: Optional[str] = None
    ):
        """
        Initialize the chatbot with API configuration and optimization data.
        
        Args:
            team_api_key: Team API key for x-api-key header
            shared_openai_key: Shared OpenAI API key
            model: Model identifier (must be from allowed list)
            base_url: Bedrock proxy base URL
            portfolio_data_path: Path to portfolio_summary.json
            risk_adjusted_data_path: Path to portfolio_summary_risk_adjusted.json
        """
        # API Client Setup
        self.client = OpenAI(
            api_key=shared_openai_key,
            base_url=base_url,
            default_headers={
                "x-api-key": team_api_key
            }
        )
        self.model = model
        
        # Allowed models list
        self.allowed_models = [
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "global.anthropic.claude-haiku-4-5-20251001-v1:0",
            "mistral.mistral-large-3-675b-instruct",
            "mistral.mistral-small-2402-v1:0",
            "openai.gpt-oss-120b-1:0",
            "global.amazon.nova-2-lite-v1:0"
        ]
        
        if model not in self.allowed_models:
            raise ValueError(f"Model {model} not in allowed list: {self.allowed_models}")
        
        # Load optimization data
        if portfolio_data_path is None:
            portfolio_data_path = Path("processed/portfolio_summary.json")
        if risk_adjusted_data_path is None:
            risk_adjusted_data_path = Path("processed/portfolio_summary_risk_adjusted.json")
        
        self.portfolio_data = self._load_json(portfolio_data_path)
        self.risk_adjusted_data = self._load_json(risk_adjusted_data_path)
        
        # Conversation state
        self.conversation_history: List[Dict[str, str]] = []
        self.system_prompt = self._build_system_prompt()
        
        # Initialize conversation with system message
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Load JSON data from file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_system_prompt(self) -> str:
        """
        Construct the system prompt that defines the assistant's role and capabilities.
        
        The prompt establishes the assistant as a professional freight analytics advisor
        with access to optimization outputs.
        """
        # Format portfolio data for context
        portfolio_summary = self._format_portfolio_summary()
        risk_summary = self._format_risk_summary()
        
        prompt = f"""You are an experienced shipping analyst and chartering manager for a dry bulk shipping company. 
You speak like a commercial decision-maker, not a textbook. Your role is to provide decisive, actionable recommendations 
based on voyage optimization analysis.

**Current Optimization Results:**

{portfolio_summary}

**Risk-Adjusted Analysis:**

{risk_summary}

**Response Style - CRITICAL:**
- Speak like an experienced shipping analyst or chartering manager, not a textbook
- Prioritize what drives the decision: TCE, profit, fuel exposure, delays, and risk
- Keep answers concise and confident; avoid unnecessary explanation unless asked
- Use plain commercial language:
  * "more exposed to fuel prices" (not "has higher fuel cost sensitivity")
  * "delay risk outweighs higher freight" (not "the probability of delays reduces the economic advantage")
  * "better risk-adjusted return" (not "superior risk-adjusted profitability metrics")
- If comparing options, clearly state what wins and why in the FIRST sentence
- Summarize trade-offs in 1-2 sharp points rather than long paragraphs
- If uncertainty matters, acknowledge it briefly and explain how it affects the recommendation
- Sound helpful, decisive, and commercially intuitive

**Guidelines:**
1. Always base recommendations on the provided optimization data - never invent numbers
2. Use commercial shipping terminology naturally (TCE, ballast, laden, laycan, demurrage)
3. Lead with the decision, then brief reasoning
4. When comparing options, start with: "[Option X] wins because [key reason]. Trade-off: [1-2 points]"
5. Format numbers clearly (use USD, days, MT as appropriate)
6. Be direct - if something is clearly better, say it clearly

**Examples of Good Responses:**
- "PACIFIC GLORY → MARKET_5 delivers $59,876/day TCE, best in portfolio. Short voyage (37 days) with manageable fuel exposure."
- "ANN BELL is more exposed to fuel prices due to 11,124 NM ballast leg, but the $23/MT rate still makes it profitable."
- "Delay risk in China ports outweighs the higher freight rate - risk-adjusted TCE drops 29% vs base case."

Remember: You're a commercial decision-maker. Be decisive, concise, and focus on what matters for the business."""
        
        return prompt
    
    def _format_portfolio_summary(self) -> str:
        """Format portfolio summary data for inclusion in system prompt."""
        data = self.portfolio_data
        
        summary = f"""**Portfolio Overview:**
- Total Portfolio Profit: ${data.get('total_portfolio_profit', 0):,.2f}
- Cargill Vessel Profit: ${data.get('cargill_vessel_profit', 0):,.2f}
- Market Vessel Profit: ${data.get('market_vessel_cost', 0):,.2f}
- Committed Cargoes: {data.get('committed_cargoes_assigned', 0)}/3
- Market Cargoes: {data.get('market_cargoes_taken', 0)}/8
- Total Assignments: {data.get('total_assignments', 0)}
- Idle Vessels: {data.get('idle_vessels', 0)}

**Vessel-Cargo Assignments:**"""
        
        for assignment in data.get('assignments', [])[:10]:  # Limit to first 10 for prompt size
            summary += f"""
- {assignment.get('Vessel_Name', 'Unknown')} → {assignment.get('Cargo_ID', 'Unknown')}
  Route: {assignment.get('Load_Port', '')} → {assignment.get('Discharge_Port', '')}
  Profit: ${assignment.get('Leg_Profit', 0):,.2f} | TCE: ${assignment.get('TCE_Leg', 0):,.2f}/day
  Days: {assignment.get('Leg_Days', 0):.1f} | Quantity: {assignment.get('Quantity_MT', 0):,} MT"""
        
        return summary
    
    def _format_risk_summary(self) -> str:
        """Format risk-adjusted summary data for inclusion in system prompt."""
        data = self.risk_adjusted_data
        
        summary = f"""**Risk-Adjusted Portfolio:**
- Base Profit: ${data.get('base_portfolio_profit', 0):,.2f}
- Risk-Adjusted Profit: ${data.get('risk_adjusted_portfolio_profit', 0):,.2f}
- Risk Impact: ${data.get('risk_impact', 0):,.2f} ({data.get('risk_impact', 0) / max(abs(data.get('base_portfolio_profit', 1)), 1) * 100:.2f}%)
- Average TCE (Base): ${data.get('average_tce', 0):,.2f}/day
- Average TCE (Risk-Adjusted): ${data.get('risk_adjusted_average_tce', 0):,.2f}/day
- Total Risk Delay: {data.get('total_risk_delay_days', 0):.1f} days
- Average Fuel Adjustment: {data.get('average_fuel_adjustment_pct', 0):.2f}%

**Risk Metrics:**
- Average Weather Delay: {data.get('risk_metrics', {}).get('average_weather_delay_days', 0):.2f} days
- Average Congestion Delay: {data.get('risk_metrics', {}).get('average_congestion_delay_days', 0):.2f} days
- Average Waiting Days Risk: {data.get('risk_metrics', {}).get('average_waiting_days_risk', 0):.2f} days"""
        
        return summary
    
    def _inject_data_context(self, user_query: str) -> str:
        """
        Enhance user query with relevant data context if needed.
        
        This allows the model to access specific assignment details when answering queries.
        """
        # For detailed queries, append relevant assignment data
        query_lower = user_query.lower()
        
        # Check if query asks about specific vessel or cargo
        assignments_context = ""
        for assignment in self.portfolio_data.get('assignments', []):
            vessel = assignment.get('Vessel_Name', '').lower()
            cargo = assignment.get('Cargo_ID', '').lower()
            
            if vessel in query_lower or cargo in query_lower:
                assignments_context += f"\n\n**Assignment Details for {assignment.get('Vessel_Name')} → {assignment.get('Cargo_ID')}:**"
                assignments_context += f"\n- Route: {assignment.get('Load_Port')} → {assignment.get('Discharge_Port')}"
                assignments_context += f"\n- Profit: ${assignment.get('Leg_Profit', 0):,.2f}"
                assignments_context += f"\n- TCE: ${assignment.get('TCE_Leg', 0):,.2f}/day"
                assignments_context += f"\n- Voyage Days: {assignment.get('Leg_Days', 0):.1f}"
                assignments_context += f"\n- Revenue: ${assignment.get('Revenue', 0):,.2f}"
                assignments_context += f"\n- Fuel Cost: ${assignment.get('Fuel_Cost', 0):,.2f}"
                assignments_context += f"\n- Hire Cost: ${assignment.get('Hire_Cost', 0):,.2f}"
                assignments_context += f"\n- Ballast Distance: {assignment.get('Ballast_Distance', 0):,.0f} NM"
                assignments_context += f"\n- Laden Distance: {assignment.get('Laden_Distance', 0):,.0f} NM"
        
        # Add risk-adjusted data if query mentions risk
        if any(word in query_lower for word in ['risk', 'delay', 'congestion', 'weather', 'adjusted']):
            risk_context = "\n\n**Risk-Adjusted Details:**"
            for assignment in self.risk_adjusted_data.get('assignments', []):
                risk_context += f"\n\n{assignment.get('vessel')} → {assignment.get('cargo')}:"
                risk_context += f"\n- Base Profit: ${assignment.get('base_profit', 0):,.2f}"
                risk_context += f"\n- Risk-Adjusted Profit: ${assignment.get('risk_adjusted_profit', 0):,.2f}"
                risk_context += f"\n- Base TCE: ${assignment.get('base_tce', 0):,.2f}/day"
                risk_context += f"\n- Risk-Adjusted TCE: ${assignment.get('risk_adjusted_tce', 0):,.2f}/day"
                risk_context += f"\n- Total Delay: {assignment.get('total_delay_days', 0):.1f} days"
                risk_context += f"\n- Fuel Adjustment: {assignment.get('fuel_adjustment_pct', 0):.2f}%"
            
            assignments_context += risk_context
        
        if assignments_context:
            return user_query + "\n\n" + assignments_context
        
        return user_query
    
    def chat(self, user_message: str, include_context: bool = True) -> str:
        """
        Process a user message and return the assistant's response.
        
        Args:
            user_message: User's query or message
            include_context: Whether to inject relevant data context into the query
        
        Returns:
            Assistant's response as a string
        """
        # Enhance query with data context if requested
        if include_context:
            enhanced_query = self._inject_data_context(user_message)
        else:
            enhanced_query = user_message
        
        # Add user message to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": enhanced_query
        })
        
        try:
            # Call the API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.7,  # Balance between creativity and consistency
                max_tokens=2000   # Sufficient for detailed responses
            )
            
            # Extract assistant response
            assistant_message = response.choices[0].message.content
            
            # Add assistant response to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error processing your request: {str(e)}"
            return error_msg
    
    def reset_conversation(self):
        """Reset the conversation history while keeping system prompt and data."""
        self.conversation_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get the current portfolio summary data."""
        return self.portfolio_data
    
    def get_risk_adjusted_summary(self) -> Dict[str, Any]:
        """Get the current risk-adjusted summary data."""
        return self.risk_adjusted_data


def main():
    """
    Example usage of the VoyageChatbot.
    
    This demonstrates both minimum and higher-achievement features.
    """
    # Try to load from config file
    TEAM_API_KEY = None
    SHARED_OPENAI_KEY = None
    CHATBOT_MODEL = None
    
    try:
        from load_env import load_env_file
        config_vars = load_env_file("chatbot_config.txt")
        TEAM_API_KEY = config_vars.get('TEAM_API_KEY')
        SHARED_OPENAI_KEY = config_vars.get('SHARED_OPENAI_KEY')
        CHATBOT_MODEL = config_vars.get('CHATBOT_MODEL')
    except:
        pass
    
    # Fall back to environment variables
    if not TEAM_API_KEY or TEAM_API_KEY == "your-team-api-key-here":
        TEAM_API_KEY = os.getenv("TEAM_API_KEY")
    if not SHARED_OPENAI_KEY or SHARED_OPENAI_KEY == "your-shared-openai-key-here":
        SHARED_OPENAI_KEY = os.getenv("SHARED_OPENAI_KEY")
    if not CHATBOT_MODEL:
        CHATBOT_MODEL = os.getenv("CHATBOT_MODEL", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    
    if not TEAM_API_KEY or not SHARED_OPENAI_KEY:
        print("⚠️  Warning: API keys not found")
        print()
        print("Please update chatbot_config.txt with your API keys:")
        print("  1. Open chatbot_config.txt")
        print("  2. Replace 'your-team-api-key-here' with your actual TEAM_API_KEY")
        print("  3. Replace 'your-shared-openai-key-here' with your actual SHARED_OPENAI_KEY")
        print()
        print("Alternatively, set environment variables:")
        print("  export TEAM_API_KEY='your-key'")
        print("  export SHARED_OPENAI_KEY='your-key'")
        return
    
    # Initialize chatbot
    print("=" * 70)
    print("VOYAGE OPTIMIZATION CHATBOT")
    print("=" * 70)
    print("\nInitializing chatbot...")
    
    try:
        chatbot = VoyageChatbot(
            team_api_key=TEAM_API_KEY,
            shared_openai_key=SHARED_OPENAI_KEY,
            model=CHATBOT_MODEL
        )
        print("✓ Chatbot initialized successfully")
        print(f"✓ Loaded portfolio data: {len(chatbot.portfolio_data.get('assignments', []))} assignments")
        print(f"✓ Loaded risk-adjusted data: {len(chatbot.risk_adjusted_data.get('assignments', []))} assignments")
    except Exception as e:
        print(f"✗ Error initializing chatbot: {e}")
        return
    
    print("\n" + "=" * 70)
    print("EXAMPLE QUERIES")
    print("=" * 70)
    
    # Example 1: Minimum requirement - Basic recommendation
    print("\n[Example 1: Basic Voyage Recommendation]")
    print("User: What is the best voyage assignment for PACIFIC GLORY?")
    print("\nAssistant:")
    response1 = chatbot.chat("What is the best voyage assignment for PACIFIC GLORY?")
    print(response1)
    
    # Example 2: Higher achievement - Comparison
    print("\n" + "-" * 70)
    print("\n[Example 2: Compare Multiple Options]")
    print("User: Compare the top 3 most profitable voyages")
    print("\nAssistant:")
    response2 = chatbot.chat("Compare the top 3 most profitable voyages side-by-side, highlighting key trade-offs")
    print(response2)
    
    # Example 3: Higher achievement - Trade-offs
    print("\n" + "-" * 70)
    print("\n[Example 3: Explain Trade-offs]")
    print("User: What are the trade-offs between ANN BELL and PACIFIC GLORY assignments?")
    print("\nAssistant:")
    response3 = chatbot.chat("What are the trade-offs between ANN BELL and PACIFIC GLORY assignments? Consider fuel exposure, delays, and ballast distance")
    print(response3)
    
    # Example 4: Higher achievement - Risk analysis
    print("\n" + "-" * 70)
    print("\n[Example 4: Risk-Adjusted Analysis]")
    print("User: How does risk adjustment affect the portfolio profit?")
    print("\nAssistant:")
    response4 = chatbot.chat("How does risk adjustment affect the portfolio profit? Explain the main risk factors")
    print(response4)
    
    # Example 5: Higher achievement - Economic intuition
    print("\n" + "-" * 70)
    print("\n[Example 5: Economic Intuition]")
    print("User: Why is PACIFIC GLORY assigned to MARKET_5 instead of a committed cargo?")
    print("\nAssistant:")
    response5 = chatbot.chat("Why is PACIFIC GLORY assigned to MARKET_5 instead of a committed cargo? Explain the economic reasoning")
    print(response5)
    
    print("\n" + "=" * 70)
    print("CHATBOT READY FOR INTERACTIVE USE")
    print("=" * 70)
    print("\nYou can now use chatbot.chat('your question') for interactive queries")


if __name__ == "__main__":
    main()

