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
import pandas as pd

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
        # Use risk_adjusted file as primary source since it has the correct base numbers
        if risk_adjusted_data_path is None:
            risk_adjusted_data_path = Path("processed/portfolio_summary_risk_adjusted.json")
        
        self.risk_adjusted_data = self._load_json(risk_adjusted_data_path)
        
        # For portfolio_data, use risk_adjusted file but extract base values
        # This ensures we use the correct numbers from the actual optimization
        if portfolio_data_path is None:
            # Use risk_adjusted data as the source of truth
            self.portfolio_data = self._extract_base_data_from_risk_adjusted(self.risk_adjusted_data)
        else:
            self.portfolio_data = self._load_json(portfolio_data_path)
            # If the base file has wrong numbers, prefer risk_adjusted file
            if self.portfolio_data.get('total_portfolio_profit', 0) < 5000000:
                # Old/wrong data detected, use risk_adjusted file instead
                self.portfolio_data = self._extract_base_data_from_risk_adjusted(self.risk_adjusted_data)
        
        # Load vessel specs to identify all Cargill vessels and idle vessels
        self.all_cargill_vessels = self._load_cargill_vessels()
        self.idle_vessels = self._calculate_idle_vessels()
        
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
            # Return empty structure instead of raising error
            print(f"Warning: Data file not found: {path}, using empty structure")
            return {
                "assignments": [],
                "total_portfolio_profit": 0,
                "base_portfolio_profit": 0,
                "total_assignments": 0,
                "committed_cargoes_assigned": 0,
                "market_cargoes_taken": 0
            }
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                # Ensure required structure exists
                if "assignments" not in data:
                    data["assignments"] = []
                return data
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {path}: {e}, using empty structure")
            return {
                "assignments": [],
                "total_portfolio_profit": 0,
                "base_portfolio_profit": 0,
                "total_assignments": 0,
                "committed_cargoes_assigned": 0,
                "market_cargoes_taken": 0
            }
    
    def _load_cargill_vessels(self) -> List[str]:
        """Load all Cargill vessel names from vessel_specs.csv."""
        try:
            vessel_specs_path = Path("processed/vessel_specs.csv")
            if not vessel_specs_path.exists():
                # Try alternative path
                vessel_specs_path = Path("data/vessel_specs.csv")
            
            if vessel_specs_path.exists():
                vessels_df = pd.read_csv(vessel_specs_path)
                cargill_vessels = vessels_df[vessels_df["Fleet"] == "Cargill"]["Vessel Name"].tolist()
                return cargill_vessels
            else:
                print(f"Warning: Vessel specs file not found at {vessel_specs_path}")
                # Fallback: extract from assignments if available
                assigned_vessels = set()
                for assignment in self.portfolio_data.get('assignments', []):
                    vessel = assignment.get('Vessel_Name') or assignment.get('vessel', '')
                    if vessel:
                        assigned_vessels.add(vessel)
                # Also check risk_adjusted_data
                for assignment in self.risk_adjusted_data.get('assignments', []):
                    vessel = assignment.get('vessel', '')
                    if vessel and assignment.get('fleet') == 'Cargill':
                        assigned_vessels.add(vessel)
                return list(assigned_vessels)
        except Exception as e:
            print(f"Warning: Error loading vessel specs: {e}")
            return []
    
    def _calculate_idle_vessels(self) -> List[str]:
        """Calculate which Cargill vessels are idle (not assigned to any cargo)."""
        # Get all assigned vessels from both portfolio_data and risk_adjusted_data
        assigned_vessels = set()
        
        # Check portfolio_data assignments
        for assignment in self.portfolio_data.get('assignments', []):
            vessel = assignment.get('Vessel_Name') or assignment.get('vessel', '')
            if vessel:
                assigned_vessels.add(vessel.upper())
        
        # Check risk_adjusted_data assignments (more reliable)
        for assignment in self.risk_adjusted_data.get('assignments', []):
            vessel = assignment.get('vessel', '')
            if vessel and assignment.get('fleet') == 'Cargill':
                assigned_vessels.add(vessel.upper())
        
        # Find idle vessels (Cargill vessels not in assignments)
        idle = []
        for vessel in self.all_cargill_vessels:
            if vessel.upper() not in assigned_vessels:
                idle.append(vessel)
        
        return idle
    
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
    
    def _normalize_assignment(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize assignment data to expected format."""
        # Handle different JSON structures
        normalized = {}
        
        # Vessel name
        normalized['Vessel_Name'] = assignment.get('Vessel_Name') or assignment.get('vessel', 'Unknown')
        
        # Cargo ID
        normalized['Cargo_ID'] = assignment.get('Cargo_ID') or assignment.get('cargo_id') or assignment.get('cargo') or assignment.get('route', 'Unknown')
        
        # Profit (try multiple field names)
        normalized['Leg_Profit'] = (
            assignment.get('Leg_Profit') or 
            assignment.get('total_profit') or 
            assignment.get('base_profit') or
            assignment.get('risk_adjusted_profit') or
            assignment.get('profit', 0)
        )
        
        # TCE (try multiple field names)
        normalized['TCE_Leg'] = (
            assignment.get('TCE_Leg') or 
            assignment.get('TCE') or 
            assignment.get('tce') or
            assignment.get('base_tce') or
            assignment.get('risk_adjusted_tce', 0)
        )
        
        # Ports - handle different formats
        load_port = assignment.get('Load_Port') or assignment.get('load_port')
        discharge_port = assignment.get('Discharge_Port') or assignment.get('discharge_port')
        
        # If route is a string like "PORT1 → PORT2", parse it
        if not load_port or not discharge_port:
            route = assignment.get('route', '')
            if '→' in route or '->' in route:
                parts = route.replace('→', '->').split('->')
                if len(parts) == 2:
                    load_port = load_port or parts[0].strip()
                    discharge_port = discharge_port or parts[1].strip()
        
        normalized['Load_Port'] = load_port or 'Unknown'
        normalized['Discharge_Port'] = discharge_port or 'Unknown'
        
        # Days
        normalized['Leg_Days'] = assignment.get('Leg_Days') or assignment.get('voyage_days') or assignment.get('days', 0)
        
        # Cargo type
        normalized['Cargo_Type'] = assignment.get('Cargo_Type') or assignment.get('cargo_type', 'Unknown')
        
        # Quantity
        normalized['Quantity_MT'] = assignment.get('Quantity_MT') or assignment.get('quantity', 0)
        
        return normalized
    
    def _extract_base_data_from_risk_adjusted(self, risk_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract base case data from risk-adjusted file structure."""
        # Get assigned Cargill vessels to calculate idle count
        assigned_cargill_vessels = set()
        for assignment in risk_data.get('assignments', []):
            if assignment.get('fleet') == 'Cargill':
                vessel = assignment.get('vessel', '')
                if vessel:
                    assigned_cargill_vessels.add(vessel.upper())
        
        # Calculate idle vessels count (will be updated later when vessel specs are loaded)
        total_cargill_vessels = risk_data.get('cargill_vessels', 0)
        idle_count = max(0, total_cargill_vessels - len(assigned_cargill_vessels))
        
        base_data = {
            "total_portfolio_profit": risk_data.get('base_portfolio_profit', 0),
            "cargill_vessel_profit": 0,  # Will calculate from assignments
            "market_vessel_cost": 0,  # Will calculate from assignments
            "committed_cargoes_assigned": risk_data.get('committed_cargoes', 0),
            "market_cargoes_taken": risk_data.get('market_cargoes', 0),
            "total_assignments": len(risk_data.get('assignments', [])),
            "idle_vessels": idle_count,
            "assignments": []
        }
        
        # Convert risk-adjusted assignments to base format
        for assignment in risk_data.get('assignments', []):
            base_assignment = {
                "Vessel_Name": assignment.get('vessel', 'Unknown'),
                "Cargo_ID": assignment.get('cargo', assignment.get('route', 'Unknown')),
                "Leg_Profit": assignment.get('base_profit', 0),
                "TCE_Leg": assignment.get('base_tce', 0),
                "Load_Port": "",
                "Discharge_Port": "",
                "Leg_Days": assignment.get('voyage_days', 0),
                "Quantity_MT": 0
            }
            
            # Parse route if available
            route = assignment.get('route', '')
            if '→' in route or '->' in route:
                parts = route.replace('→', '->').split('->')
                if len(parts) == 2:
                    base_assignment["Load_Port"] = parts[0].strip()
                    base_assignment["Discharge_Port"] = parts[1].strip()
            
            # Calculate fleet-specific profits
            fleet = assignment.get('fleet', '')
            base_profit = assignment.get('base_profit', 0)
            if fleet == 'Cargill':
                base_data["cargill_vessel_profit"] += base_profit
            elif fleet == 'Market':
                base_data["market_vessel_cost"] += base_profit
            
            base_data["assignments"].append(base_assignment)
        
        return base_data
    
    def _format_portfolio_summary(self) -> str:
        """Format portfolio summary data for inclusion in system prompt."""
        data = self.portfolio_data
        assignments = data.get('assignments', [])
        
        # Calculate totals from assignments if not directly available
        if 'total_portfolio_profit' in data:
            total_profit = data.get('total_portfolio_profit', 0)
        elif 'base_portfolio_profit' in data:
            total_profit = data.get('base_portfolio_profit', 0)
        else:
            total_profit = sum(
                a.get('Leg_Profit') or a.get('total_profit') or a.get('base_profit') or a.get('risk_adjusted_profit', 0)
                for a in assignments
            )
        
        cargill_profit = data.get('cargill_vessel_profit', 0)
        market_profit = data.get('market_vessel_cost', 0)
        
        committed = data.get('committed_cargoes_assigned', 0) or data.get('committed_cargoes', 0)
        market = data.get('market_cargoes_taken', 0) or data.get('market_cargoes', 0)
        total_assignments = data.get('total_assignments', len(assignments))
        idle_count = len(self.idle_vessels)
        
        # Build Cargill fleet summary
        cargill_fleet_info = f"""**Cargill Fleet ({len(self.all_cargill_vessels)} vessels):**
- All Cargill Vessels: {', '.join(self.all_cargill_vessels)}"""
        
        if self.idle_vessels:
            cargill_fleet_info += f"""
- Idle Vessels ({idle_count}): {', '.join(self.idle_vessels)}"""
        else:
            cargill_fleet_info += f"""
- Idle Vessels: None (all vessels assigned)"""
        
        summary = f"""**Portfolio Overview:**
- Total Portfolio Profit: ${total_profit:,.2f}
- Cargill Vessel Profit: ${cargill_profit:,.2f}
- Market Vessel Profit: ${market_profit:,.2f}
- Committed Cargoes: {committed}/3
- Market Cargoes: {market}/8
- Total Assignments: {total_assignments}
- Idle Vessels: {idle_count}

{cargill_fleet_info}

**Vessel-Cargo Assignments:**"""
        
        for assignment in assignments[:10]:  # Limit to first 10 for prompt size
            # Normalize assignment data
            norm = self._normalize_assignment(assignment)
            summary += f"""
- {norm.get('Vessel_Name', 'Unknown')} → {norm.get('Cargo_ID', 'Unknown')}
  Route: {norm.get('Load_Port', '')} → {norm.get('Discharge_Port', '')}
  Profit: ${norm.get('Leg_Profit', 0):,.2f} | TCE: ${norm.get('TCE_Leg', 0):,.2f}/day
  Days: {norm.get('Leg_Days', 0):.1f} | Quantity: {norm.get('Quantity_MT', 0):,} MT"""
        
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
            # Normalize assignment
            norm = self._normalize_assignment(assignment)
            vessel = norm.get('Vessel_Name', '').lower()
            cargo = norm.get('Cargo_ID', '').lower()
            
            if vessel in query_lower or cargo in query_lower:
                assignments_context += f"\n\n**Assignment Details for {norm.get('Vessel_Name')} → {norm.get('Cargo_ID')}:**"
                assignments_context += f"\n- Route: {norm.get('Load_Port')} → {norm.get('Discharge_Port')}"
                assignments_context += f"\n- Profit: ${norm.get('Leg_Profit', 0):,.2f}"
                assignments_context += f"\n- TCE: ${norm.get('TCE_Leg', 0):,.2f}/day"
                assignments_context += f"\n- Voyage Days: {norm.get('Leg_Days', 0):.1f}"
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

