"""
Explainability & Interpretability Module
========================================
Provides interpretability for voyage optimization decisions.

Features:
1. Feature Importance Analysis - Identifies which parameters most influence decisions
2. Sensitivity Analysis - Shows how changes in inputs affect outputs
3. SHAP-style Explanations - Conceptual framework for explaining model decisions
4. Stakeholder Communication - Clear, non-technical explanations

This module helps commercial teams understand:
- Why a particular voyage was selected
- Which factors drive profitability
- How sensitive the solution is to changes
- What risks are associated with each decision
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class FeatureImportance:
    """Feature importance metrics for a voyage decision."""
    feature_name: str
    importance_score: float
    impact_direction: str  # 'positive', 'negative', 'neutral'
    impact_magnitude: float  # USD impact
    description: str


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis for a parameter."""
    parameter_name: str
    base_value: float
    base_profit: float
    sensitivity_pct: float  # % change in profit per % change in parameter
    break_even_value: Optional[float]  # Value where profit becomes zero
    description: str


class VoyageExplainability:
    """
    Explainability engine for voyage optimization decisions.
    
    Provides interpretability for:
    - Why specific voyages were selected
    - Which parameters drive profitability
    - Sensitivity to changes in key inputs
    - Risk-adjusted decision rationale
    """
    
    def __init__(self):
        """Initialize explainability engine."""
        self.feature_importance_cache = {}
        self.sensitivity_cache = {}
    
    def calculate_feature_importance(
        self,
        voyage_profit: float,
        voyage_params: Dict[str, float],
        base_profit: float = 0.0
    ) -> List[FeatureImportance]:
        """
        Calculate feature importance for voyage decision.
        
        Uses a conceptual SHAP-style approach to attribute profit to features.
        
        Parameters:
        -----------
        voyage_profit : float
            Profit for this voyage
        voyage_params : dict
            Dictionary of voyage parameters (freight_rate, bunker_cost, etc.)
        base_profit : float
            Baseline profit for comparison
        
        Returns:
        --------
        List[FeatureImportance] : Feature importance scores
        """
        importances = []
        
        # Revenue-related features
        if 'freight_rate' in voyage_params and 'quantity' in voyage_params:
            revenue = voyage_params['freight_rate'] * voyage_params['quantity']
            revenue_impact = revenue * 0.95  # After commission
            importances.append(FeatureImportance(
                feature_name='Freight Rate',
                importance_score=abs(revenue_impact / max(voyage_profit, 1)),
                impact_direction='positive',
                impact_magnitude=revenue_impact,
                description=f'Freight rate of ${voyage_params["freight_rate"]:.2f}/MT contributes ${revenue_impact:,.0f} to revenue'
            ))
        
        # Distance-related features
        if 'ballast_distance' in voyage_params and 'laden_distance' in voyage_params:
            total_dist = voyage_params['ballast_distance'] + voyage_params['laden_distance']
            # Distance affects fuel cost and time
            distance_impact = -total_dist * 50  # Rough estimate: $50 per NM
            importances.append(FeatureImportance(
                feature_name='Total Distance',
                importance_score=abs(distance_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=distance_impact,
                description=f'Total distance of {total_dist:.0f} NM increases costs by approximately ${abs(distance_impact):,.0f}'
            ))
        
        # Bunker cost
        if 'bunker_cost' in voyage_params:
            bunker_impact = -voyage_params['bunker_cost']
            importances.append(FeatureImportance(
                feature_name='Bunker Cost',
                importance_score=abs(bunker_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=bunker_impact,
                description=f'Bunker cost of ${voyage_params["bunker_cost"]:,.0f} reduces profit'
            ))
        
        # Hire rate
        if 'hire_rate' in voyage_params and 'voyage_days' in voyage_params:
            hire_cost = voyage_params['hire_rate'] * voyage_params['voyage_days']
            hire_impact = -hire_cost
            importances.append(FeatureImportance(
                feature_name='Hire Rate',
                importance_score=abs(hire_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=hire_impact,
                description=f'Daily hire rate of ${voyage_params["hire_rate"]:,.0f}/day costs ${hire_cost:,.0f} for {voyage_params["voyage_days"]:.1f} days'
            ))
        
        # Port costs
        if 'port_cost' in voyage_params:
            port_impact = -voyage_params['port_cost']
            importances.append(FeatureImportance(
                feature_name='Port Costs',
                importance_score=abs(port_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=port_impact,
                description=f'Port costs of ${voyage_params["port_cost"]:,.0f} reduce profit'
            ))
        
        # Laycan risk
        if 'laycan_risk' in voyage_params:
            laycan_impact = -voyage_params['laycan_risk'] * 50000  # Rough estimate
            importances.append(FeatureImportance(
                feature_name='Laycan Risk',
                importance_score=abs(laycan_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=laycan_impact,
                description=f'Laycan risk factor of {voyage_params["laycan_risk"]:.2f} indicates potential delays'
            ))
        
        # Congestion risk
        if 'congestion_risk' in voyage_params:
            congestion_impact = -voyage_params['congestion_risk'] * 30000
            importances.append(FeatureImportance(
                feature_name='Port Congestion Risk',
                importance_score=abs(congestion_impact / max(voyage_profit, 1)),
                impact_direction='negative',
                impact_magnitude=congestion_impact,
                description=f'Congestion risk of {voyage_params["congestion_risk"]:.2f} may cause delays'
            ))
        
        # Sort by importance score
        importances.sort(key=lambda x: x.importance_score, reverse=True)
        
        return importances
    
    def sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        profit_function,
        parameters_to_test: List[str],
        variation_range: float = 0.20  # ±20% variation
    ) -> List[SensitivityResult]:
        """
        Perform sensitivity analysis on key parameters.
        
        Parameters:
        -----------
        base_params : dict
            Base parameter values
        profit_function : callable
            Function that calculates profit given parameters
        parameters_to_test : list
            List of parameter names to test
        variation_range : float
            Range of variation to test (±variation_range)
        
        Returns:
        --------
        List[SensitivityResult] : Sensitivity analysis results
        """
        results = []
        base_profit = profit_function(base_params)
        
        for param_name in parameters_to_test:
            if param_name not in base_params:
                continue
            
            base_value = base_params[param_name]
            
            # Test variations
            variations = [-variation_range, -variation_range/2, 0, variation_range/2, variation_range]
            profits = []
            
            for var in variations:
                test_params = base_params.copy()
                test_params[param_name] = base_value * (1 + var)
                profit = profit_function(test_params)
                profits.append(profit)
            
            # Calculate sensitivity (elasticity)
            # Sensitivity = (% change in profit) / (% change in parameter)
            if base_value != 0 and base_profit != 0:
                # Use central difference
                profit_change_pct = ((profits[4] - profits[0]) / base_profit) * 100
                param_change_pct = variation_range * 2 * 100
                sensitivity = profit_change_pct / param_change_pct if param_change_pct != 0 else 0
            else:
                sensitivity = 0.0
            
            # Find break-even point (where profit becomes zero)
            break_even = None
            if base_profit > 0:
                # Linear interpolation to find where profit = 0
                for i in range(len(variations) - 1):
                    if profits[i] * profits[i+1] <= 0:  # Sign change
                        # Interpolate
                        x1, y1 = variations[i], profits[i]
                        x2, y2 = variations[i+1], profits[i+1]
                        if y2 != y1:
                            zero_crossing = x1 - y1 * (x2 - x1) / (y2 - y1)
                            break_even = base_value * (1 + zero_crossing)
                            break
            
            # Generate description
            if abs(sensitivity) < 0.1:
                desc = f"Low sensitivity: {param_name} changes have minimal impact on profit"
            elif abs(sensitivity) < 0.5:
                desc = f"Moderate sensitivity: {param_name} changes moderately affect profit"
            else:
                desc = f"High sensitivity: {param_name} changes significantly affect profit"
            
            results.append(SensitivityResult(
                parameter_name=param_name,
                base_value=base_value,
                base_profit=base_profit,
                sensitivity_pct=sensitivity * 100,
                break_even_value=break_even,
                description=desc
            ))
        
        # Sort by absolute sensitivity
        results.sort(key=lambda x: abs(x.sensitivity_pct), reverse=True)
        
        return results
    
    def explain_voyage_selection(
        self,
        selected_voyage: Dict[str, any],
        alternative_voyages: List[Dict[str, any]],
        risk_profile: Optional[Dict[str, any]] = None
    ) -> Dict[str, any]:
        """
        Generate human-readable explanation for why a voyage was selected.
        
        Parameters:
        -----------
        selected_voyage : dict
            Details of the selected voyage
        alternative_voyages : list
            List of alternative voyages that were not selected
        risk_profile : dict, optional
            Risk profile from ML risk simulation
        
        Returns:
        --------
        dict : Explanation in plain language
        """
        explanation = {
            'selected_voyage': selected_voyage.get('vessel_name', 'Unknown'),
            'route': selected_voyage.get('route', 'Unknown'),
            'profit': selected_voyage.get('profit', 0),
            'tce': selected_voyage.get('tce', 0),
            'key_drivers': [],
            'why_selected': [],
            'risks': [],
            'alternatives_comparison': []
        }
        
        # Key drivers
        if selected_voyage.get('freight_rate', 0) > 0:
            explanation['key_drivers'].append(
                f"High freight rate: ${selected_voyage.get('freight_rate', 0):.2f}/MT"
            )
        
        if selected_voyage.get('distance', 0) < 5000:
            explanation['key_drivers'].append(
                f"Short distance: {selected_voyage.get('distance', 0):.0f} NM reduces fuel costs"
            )
        
        if selected_voyage.get('tce', 0) > selected_voyage.get('hire_rate', 0) * 1.2:
            explanation['key_drivers'].append(
                f"Strong TCE: ${selected_voyage.get('tce', 0):,.0f}/day significantly exceeds hire rate"
            )
        
        # Why selected
        if alternative_voyages:
            best_alt = max(alternative_voyages, key=lambda x: x.get('profit', 0))
            profit_advantage = selected_voyage.get('profit', 0) - best_alt.get('profit', 0)
            
            if profit_advantage > 100000:
                explanation['why_selected'].append(
                    f"Selected voyage generates ${profit_advantage:,.0f} more profit than the best alternative"
                )
            
            if selected_voyage.get('tce', 0) > best_alt.get('tce', 0):
                explanation['why_selected'].append(
                    f"Higher TCE than alternatives: ${selected_voyage.get('tce', 0):,.0f}/day vs ${best_alt.get('tce', 0):,.0f}/day"
                )
        else:
            explanation['why_selected'].append(
                "This is the only feasible voyage option given constraints"
            )
        
        # Risks
        if risk_profile:
            total_delay = risk_profile.get('total_delay_days', 0)
            if total_delay > 2:
                explanation['risks'].append(
                    f"Moderate delay risk: Expected {total_delay:.1f} days of operational delays"
                )
            
            demurrage = risk_profile.get('demurrage_risk', {}).get('demurrage_cost_usd', 0)
            if demurrage > 50000:
                explanation['risks'].append(
                    f"Demurrage exposure: Potential ${demurrage:,.0f} in demurrage costs"
                )
            
            laycan_breach = risk_profile.get('waiting_risk', {}).get('laycan_breach_prob', 0)
            if laycan_breach > 0.1:
                explanation['risks'].append(
                    f"Laycan risk: {laycan_breach*100:.0f}% probability of missing laycan window"
                )
        else:
            explanation['risks'].append("No significant risks identified")
        
        # Alternatives comparison
        for alt in alternative_voyages[:3]:  # Top 3 alternatives
            explanation['alternatives_comparison'].append({
                'vessel': alt.get('vessel_name', 'Unknown'),
                'route': alt.get('route', 'Unknown'),
                'profit': alt.get('profit', 0),
                'why_not_selected': f"Profit ${selected_voyage.get('profit', 0) - alt.get('profit', 0):,.0f} lower than selected option"
            })
        
        return explanation
    
    def generate_portfolio_explanation(
        self,
        portfolio_assignments: pd.DataFrame,
        feature_importances: Dict[str, List[FeatureImportance]],
        sensitivity_results: Dict[str, List[SensitivityResult]]
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation for portfolio-level decisions.
        
        Parameters:
        -----------
        portfolio_assignments : DataFrame
            All vessel-cargo assignments
        feature_importances : dict
            Feature importance for each assignment
        sensitivity_results : dict
            Sensitivity analysis results
        
        Returns:
        --------
        dict : Portfolio-level explanation
        """
        total_profit = portfolio_assignments['Total_Voyage_Profit_USD'].sum()
        
        explanation = {
            'total_portfolio_profit': total_profit,
            'total_assignments': len(portfolio_assignments),
            'key_insights': [],
            'top_drivers': [],
            'sensitivity_summary': [],
            'recommendations': []
        }
        
        # Aggregate feature importances
        all_features = {}
        for vessel, importances in feature_importances.items():
            for imp in importances:
                if imp.feature_name not in all_features:
                    all_features[imp.feature_name] = {
                        'total_impact': 0,
                        'count': 0,
                        'avg_importance': 0
                    }
                all_features[imp.feature_name]['total_impact'] += abs(imp.impact_magnitude)
                all_features[imp.feature_name]['count'] += 1
        
        # Calculate averages
        for feature_name, data in all_features.items():
            data['avg_importance'] = data['total_impact'] / data['count'] if data['count'] > 0 else 0
        
        # Top drivers
        sorted_features = sorted(
            all_features.items(),
            key=lambda x: x[1]['avg_importance'],
            reverse=True
        )
        
        for feature_name, data in sorted_features[:5]:
            explanation['top_drivers'].append({
                'feature': feature_name,
                'average_impact': data['avg_importance'],
                'appears_in': f"{data['count']} voyages"
            })
        
        # Sensitivity summary
        for vessel, sensitivities in sensitivity_results.items():
            top_sensitive = max(sensitivities, key=lambda x: abs(x.sensitivity_pct))
            explanation['sensitivity_summary'].append({
                'vessel': vessel,
                'most_sensitive_parameter': top_sensitive.parameter_name,
                'sensitivity': f"{top_sensitive.sensitivity_pct:.1f}%",
                'description': top_sensitive.description
            })
        
        # Key insights
        explanation['key_insights'].append(
            f"Portfolio generates ${total_profit:,.0f} in total profit"
        )
        
        avg_tce = portfolio_assignments['TCE_USD_per_day'].mean()
        explanation['key_insights'].append(
            f"Average TCE: ${avg_tce:,.0f}/day across all voyages"
        )
        
        # Recommendations
        if avg_tce > 20000:
            explanation['recommendations'].append(
                "Strong portfolio performance - consider expanding similar voyages"
            )
        else:
            explanation['recommendations'].append(
                "Monitor TCE closely - some voyages may be marginal"
            )
        
        return explanation
    
    def export_explanation_report(
        self,
        explanation: Dict[str, any],
        output_path: str
    ):
        """
        Export explanation to JSON for stakeholder communication.
        
        Parameters:
        -----------
        explanation : dict
            Explanation dictionary
        output_path : str
            Path to save JSON file
        """
        with open(output_path, 'w') as f:
            json.dump(explanation, f, indent=2, default=str)
        
        print(f"Explanation report saved to {output_path}")

