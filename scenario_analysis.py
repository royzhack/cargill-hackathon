"""
Structured Scenario Analysis Module
===================================
Tests robustness of optimal voyage recommendations by identifying threshold points
where the optimal solution changes under different scenarios.

Scenarios:
1. Port Delay in China - Incremental port delays at Chinese ports
2. Bunker Price Increase - Uniform VLSFO price increases across all ports

All scenarios maintain ML-based risk simulation integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
from copy import deepcopy
from ortools.sat.python import cp_model


class ScenarioAnalyzer:
    """
    Analyzes robustness of optimal solutions under different scenarios.
    """
    
    def __init__(self, base_assignments_df: pd.DataFrame, 
                 base_portfolio_profit: float,
                 base_optimal_arcs: List[Dict],
                 evaluate_leg_fn,
                 get_bunker_price_fn,
                 get_market_freight_rate_fn,
                 distance_lookup: Dict,
                 cargill_vessels_processed: pd.DataFrame,
                 market_vessels_processed: pd.DataFrame,
                 cargill_cargoes_processed: pd.DataFrame,
                 market_cargoes_processed: pd.DataFrame,
                 risk_simulator=None):
        """
        Initialize scenario analyzer.
        
        Parameters:
        -----------
        base_assignments_df : pd.DataFrame
            Base-case optimal assignments
        base_portfolio_profit : float
            Base-case total portfolio profit
        base_optimal_arcs : List[Dict]
            List of optimal arcs from base solution
        evaluate_leg_fn : callable
            Function to evaluate a leg (with risk simulation)
        get_bunker_price_fn : callable
            Function to get bunker price (will be modified for scenarios)
        get_market_freight_rate_fn : callable
            Function to get market freight rate
        distance_lookup : Dict
            Port-to-port distance lookup
        cargill_vessels_processed : pd.DataFrame
            Processed Cargill vessels
        market_vessels_processed : pd.DataFrame
            Processed market vessels
        cargill_cargoes_processed : pd.DataFrame
            Processed committed cargoes
        market_cargoes_processed : pd.DataFrame
            Processed market cargoes
        risk_simulator : MLRiskSimulator, optional
            Risk simulator (if ML risk is enabled)
        """
        self.base_assignments_df = base_assignments_df
        self.base_portfolio_profit = base_portfolio_profit
        self.base_optimal_arcs = base_optimal_arcs
        self.evaluate_leg_fn = evaluate_leg_fn
        self.get_bunker_price_fn = get_bunker_price_fn
        self.get_market_freight_rate_fn = get_market_freight_rate_fn
        self.distance_lookup = distance_lookup
        self.cargill_vessels_processed = cargill_vessels_processed
        self.market_vessels_processed = market_vessels_processed
        self.cargill_cargoes_processed = cargill_cargoes_processed
        self.market_cargoes_processed = market_cargoes_processed
        self.risk_simulator = risk_simulator
        
        # Identify China ports from base assignments
        self.china_ports = self._identify_china_ports()
    
    def _identify_china_ports(self) -> List[str]:
        """Identify all Chinese ports from cargo data."""
        china_ports = set()
        
        # Check load and discharge ports
        for _, cargo in pd.concat([self.cargill_cargoes_processed, self.market_cargoes_processed]).iterrows():
            load_port = str(cargo.get('load_port', '')).upper()
            discharge_port = str(cargo.get('discharge_port', '')).upper()
            
            # Common Chinese port indicators
            china_keywords = ['QINGDAO', 'SHANGHAI', 'FANGCHENG', 'LIANYUNGANG', 
                            'CAOFEIDIAN', 'JINGTANG', 'TIANJIN', 'XINGANG', 
                            'ZHANGJIAGANG', 'NANTONG', 'NINGBO', 'XIAMEN']
            
            for keyword in china_keywords:
                if keyword in load_port:
                    china_ports.add(cargo.get('load_port', ''))
                if keyword in discharge_port:
                    china_ports.add(cargo.get('discharge_port', ''))
        
        return sorted(list(china_ports))
    
    def _create_bunker_price_fn_with_multiplier(self, multiplier: float):
        """Create modified bunker price function with multiplier."""
        def get_bunker_price_modified(port_name, fuel_grade='VLSFO', date=None):
            base_price = self.get_bunker_price_fn(port_name, fuel_grade, date)
            if base_price is None:
                return None
            return base_price * multiplier
        return get_bunker_price_modified
    
    def _create_evaluate_leg_with_port_delay(self, additional_delay_days: float, china_ports: List[str]):
        """Create modified evaluate_leg function with port delays."""
        def evaluate_leg_modified(start_port, start_time, vessel_row, cargo_row, 
                                  distance_lookup, get_bunker_price_fn, get_market_freight_rate_fn):
            # Check if this cargo involves a China port
            load_port = cargo_row.get('load_port', '')
            discharge_port = cargo_row.get('discharge_port', '')
            
            # Create copy of cargo to modify
            cargo_row_modified = cargo_row.copy()
            
            # Add delay to discharge port if it's a China port
            if discharge_port in china_ports:
                original_discharge_hours = cargo_row.get('discharge_turn_time_hours', 0)
                cargo_row_modified['discharge_turn_time_hours'] = original_discharge_hours + (additional_delay_days * 24.0)
            
            # Also add delay to load port if it's a China port
            if load_port in china_ports:
                original_load_hours = cargo_row.get('load_turn_time_hours', 0)
                cargo_row_modified['load_turn_time_hours'] = original_load_hours + (additional_delay_days * 24.0)
            
            # Call original evaluate_leg with modified cargo
            return self.evaluate_leg_fn(start_port, start_time, vessel_row, cargo_row_modified,
                                       distance_lookup, get_bunker_price_fn, get_market_freight_rate_fn)
        
        return evaluate_leg_modified
    
    def _run_optimization_with_scenario(self, 
                                        port_delay_days: float = 0.0,
                                        bunker_price_multiplier: float = 1.0) -> Dict[str, Any]:
        """
        Run optimization with scenario parameters.
        
        Returns:
        --------
        dict with optimization results
        """
        # Create modified functions
        if port_delay_days > 0:
            evaluate_leg_scenario = self._create_evaluate_leg_with_port_delay(
                port_delay_days, self.china_ports
            )
        else:
            evaluate_leg_scenario = self.evaluate_leg_fn
        
        if bunker_price_multiplier != 1.0:
            get_bunker_price_scenario = self._create_bunker_price_fn_with_multiplier(
                bunker_price_multiplier
            )
        else:
            get_bunker_price_scenario = self.get_bunker_price_fn
        
        # Regenerate arcs with scenario parameters
        all_cargoes = pd.concat([
            self.cargill_cargoes_processed.assign(cargo_type='committed', cargo_idx_in_type=range(len(self.cargill_cargoes_processed))),
            self.market_cargoes_processed.assign(cargo_type='market', cargo_idx_in_type=range(len(self.market_cargoes_processed)))
        ], ignore_index=True).reset_index(drop=True)
        
        cargill_arcs_scenario = []
        market_arcs_scenario = []
        
        # Generate Cargill vessel arcs
        for v_idx, vessel in self.cargill_vessels_processed.iterrows():
            vessel_name = vessel['vessel_name']
            start_port = vessel['current_position_port']
            start_time = vessel['estimated_time_of_departure']
            
            # Arcs from Start -> any cargo
            for c_idx, cargo in all_cargoes.iterrows():
                leg_data = evaluate_leg_scenario(
                    start_port, start_time, vessel, cargo,
                    self.distance_lookup, get_bunker_price_scenario, self.get_market_freight_rate_fn
                )
                
                if leg_data is not None:
                    cargill_arcs_scenario.append({
                        'vessel_idx': v_idx,
                        'vessel_name': vessel_name,
                        'from_node': 'START',
                        'from_port': start_port,
                        'from_time': start_time,
                        'to_node': cargo['cargo_id'],
                        'to_cargo_idx': c_idx,
                        'cargo_type': cargo['cargo_type'],
                        'leg_data': leg_data,
                        'profit': leg_data['profit'],
                        'end_port': leg_data['end_port'],
                        'end_time': leg_data['end_time']
                    })
            
            # Arcs from cargo -> cargo (chaining)
            for c1_idx, cargo1 in all_cargoes.iterrows():
                leg1_data = evaluate_leg_scenario(
                    start_port, start_time, vessel, cargo1,
                    self.distance_lookup, get_bunker_price_scenario, self.get_market_freight_rate_fn
                )
                
                if leg1_data is not None:
                    end_port_c1 = leg1_data['end_port']
                    end_time_c1 = leg1_data['end_time']
                    
                    for c2_idx, cargo2 in all_cargoes.iterrows():
                        if c1_idx == c2_idx:
                            continue
                        
                        leg2_data = evaluate_leg_scenario(
                            end_port_c1, end_time_c1, vessel, cargo2,
                            self.distance_lookup, get_bunker_price_scenario, self.get_market_freight_rate_fn
                        )
                        
                        if leg2_data is not None:
                            cargill_arcs_scenario.append({
                                'vessel_idx': v_idx,
                                'vessel_name': vessel_name,
                                'from_node': cargo1['cargo_id'],
                                'from_port': end_port_c1,
                                'from_time': end_time_c1,
                                'to_node': cargo2['cargo_id'],
                                'to_cargo_idx': c2_idx,
                                'cargo_type': cargo2['cargo_type'],
                                'leg_data': leg2_data,
                                'profit': leg2_data['profit'],
                                'end_port': leg2_data['end_port'],
                                'end_time': leg2_data['end_time']
                            })
        
        # Generate Market vessel arcs
        for m_idx, vessel in self.market_vessels_processed.iterrows():
            vessel_name = vessel['vessel_name']
            start_port = vessel['current_position_port']
            start_time = vessel['estimated_time_of_departure']
            
            for c_idx, cargo in self.cargill_cargoes_processed.iterrows():
                leg_data = evaluate_leg_scenario(
                    start_port, start_time, vessel, cargo,
                    self.distance_lookup, get_bunker_price_scenario, self.get_market_freight_rate_fn
                )
                
                if leg_data is not None:
                    market_arcs_scenario.append({
                        'vessel_idx': m_idx,
                        'vessel_name': vessel_name,
                        'from_node': 'START',
                        'from_port': start_port,
                        'from_time': start_time,
                        'to_node': cargo['cargo_id'],
                        'to_cargo_idx': c_idx,
                        'cargo_type': 'committed',
                        'leg_data': leg_data,
                        'profit': leg_data['profit'],
                        'end_port': leg_data['end_port'],
                        'end_time': leg_data['end_time']
                    })
        
        # Build CP-SAT model
        model = cp_model.CpModel()
        
        # Decision variables
        cargill_arc_vars = {}
        for i, arc in enumerate(cargill_arcs_scenario):
            var_name = f"cargill_arc_{arc['vessel_idx']}_{arc['from_node']}_{arc['to_node']}"
            cargill_arc_vars[i] = model.NewBoolVar(var_name)
        
        market_arc_vars = {}
        for i, arc in enumerate(market_arcs_scenario):
            var_name = f"market_arc_{arc['vessel_idx']}_{arc['to_node']}"
            market_arc_vars[i] = model.NewBoolVar(var_name)
        
        # Constraints
        # 1. Each committed cargo assigned exactly once
        for c_idx, cargo in self.cargill_cargoes_processed.iterrows():
            cargo_id = cargo['cargo_id']
            cargill_arcs_for_cargo = [
                i for i, arc in enumerate(cargill_arcs_scenario)
                if arc['to_node'] == cargo_id and arc['cargo_type'] == 'committed'
            ]
            market_arcs_for_cargo = [
                i for i, arc in enumerate(market_arcs_scenario)
                if arc['to_node'] == cargo_id
            ]
            terms = []
            for arc_idx in cargill_arcs_for_cargo:
                terms.append(cargill_arc_vars[arc_idx])
            for arc_idx in market_arcs_for_cargo:
                terms.append(market_arc_vars[arc_idx])
            if len(terms) > 0:
                model.Add(sum(terms) == 1)
        
        # 2. Market cargoes optional (at most once)
        market_cargo_ids = set()
        for arc in cargill_arcs_scenario:
            if arc['cargo_type'] == 'market':
                market_cargo_ids.add(arc['to_node'])
        for cargo_id in market_cargo_ids:
            arcs_for_cargo = [
                i for i, arc in enumerate(cargill_arcs_scenario)
                if arc['to_node'] == cargo_id and arc['cargo_type'] == 'market'
            ]
            if len(arcs_for_cargo) > 0:
                model.Add(sum(cargill_arc_vars[i] for i in arcs_for_cargo) <= 1)
        
        # 3. Flow conservation
        all_cargo_nodes = set()
        for arc in cargill_arcs_scenario:
            if arc['from_node'] != 'START':
                all_cargo_nodes.add(arc['from_node'])
            all_cargo_nodes.add(arc['to_node'])
        
        for v_idx in self.cargill_vessels_processed.index:
            for cargo_node in all_cargo_nodes:
                inflow_arcs = [
                    i for i, arc in enumerate(cargill_arcs_scenario)
                    if arc['vessel_idx'] == v_idx and arc['to_node'] == cargo_node
                ]
                outflow_arcs = [
                    i for i, arc in enumerate(cargill_arcs_scenario)
                    if arc['vessel_idx'] == v_idx and arc['from_node'] == cargo_node
                ]
                if len(outflow_arcs) > 0 and len(inflow_arcs) > 0:
                    inflow_sum = sum(cargill_arc_vars[i] for i in inflow_arcs)
                    outflow_sum = sum(cargill_arc_vars[i] for i in outflow_arcs)
                    model.Add(outflow_sum <= inflow_sum)
        
        # 4. Each vessel starts at most once
        for v_idx in self.cargill_vessels_processed.index:
            start_arcs = [
                i for i, arc in enumerate(cargill_arcs_scenario)
                if arc['vessel_idx'] == v_idx and arc['from_node'] == 'START'
            ]
            if len(start_arcs) > 0:
                model.Add(sum(cargill_arc_vars[i] for i in start_arcs) <= 1)
        
        # 5. Market vessels at most once
        for m_idx in self.market_vessels_processed.index:
            vessel_arcs = [
                i for i, arc in enumerate(market_arcs_scenario)
                if arc['vessel_idx'] == m_idx
            ]
            if len(vessel_arcs) > 0:
                model.Add(sum(market_arc_vars[i] for i in vessel_arcs) <= 1)
        
        # Objective
        objective_terms = []
        for i, arc in enumerate(cargill_arcs_scenario):
            profit_scaled = int(round(arc['profit'] * 100))
            objective_terms.append(cargill_arc_vars[i] * profit_scaled)
        for i, arc in enumerate(market_arcs_scenario):
            profit_scaled = int(round(arc['profit'] * 100))
            objective_terms.append(market_arc_vars[i] * profit_scaled)
        
        model.Maximize(sum(objective_terms))
        
        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 300.0
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Extract selected arcs
            selected_cargill_arcs = []
            selected_market_arcs = []
            
            for i, arc in enumerate(cargill_arcs_scenario):
                if solver.Value(cargill_arc_vars[i]) == 1:
                    selected_cargill_arcs.append(arc)
            
            for i, arc in enumerate(market_arcs_scenario):
                if solver.Value(market_arc_vars[i]) == 1:
                    selected_market_arcs.append(arc)
            
            total_profit = solver.ObjectiveValue() / 100.0
            
            # Get assignment signature for comparison
            assignment_signature = self._get_assignment_signature(
                selected_cargill_arcs, selected_market_arcs
            )
            
            return {
                'status': status,
                'total_profit': total_profit,
                'selected_cargill_arcs': selected_cargill_arcs,
                'selected_market_arcs': selected_market_arcs,
                'assignment_signature': assignment_signature,
                'objective_value': solver.ObjectiveValue()
            }
        else:
            return {
                'status': status,
                'total_profit': None,
                'selected_cargill_arcs': [],
                'selected_market_arcs': [],
                'assignment_signature': None,
                'objective_value': None
            }
    
    def _get_assignment_signature(self, cargill_arcs: List[Dict], market_arcs: List[Dict]) -> str:
        """Get a signature string representing the assignment pattern."""
        assignments = []
        for arc in cargill_arcs:
            assignments.append(f"{arc['vessel_name']}->{arc['to_node']}")
        for arc in market_arcs:
            assignments.append(f"{arc['vessel_name']}->{arc['to_node']}")
        return "|".join(sorted(assignments))
    
    def analyze_port_delay_scenario(self, 
                                   max_delay_days: float = 50.0,
                                   step_size: float = 1.0,
                                   tolerance: float = 0.5) -> Dict[str, Any]:
        """
        Analyze Scenario 1: Port Delay in China.
        
        Incrementally increases port delays at China ports and finds threshold
        where optimal solution changes.
        
        Parameters:
        -----------
        max_delay_days : float
            Maximum delay to test (days)
        step_size : float
            Initial step size for binary search (days)
        tolerance : float
            Tolerance for threshold detection (days)
        
        Returns:
        --------
        dict with analysis results
        """
        print("=" * 80)
        print("SCENARIO 1: PORT DELAY IN CHINA")
        print("=" * 80)
        print()
        print(f"China ports identified: {', '.join(self.china_ports)}")
        print(f"Base portfolio profit: ${self.base_portfolio_profit:,.2f}")
        print()
        
        # Get base assignment signature
        base_signature = self._get_assignment_signature(
            self.base_optimal_arcs, []
        )
        print(f"Base assignment signature: {base_signature}")
        print()
        
        # Binary search for threshold
        low = 0.0
        high = max_delay_days
        threshold = None
        test_results = []
        
        print("Binary search for threshold delay...")
        print("-" * 80)
        
        while high - low > tolerance:
            mid = (low + high) / 2.0
            
            print(f"Testing delay: {mid:.2f} days...", end=" ")
            
            result = self._run_optimization_with_scenario(
                port_delay_days=mid,
                bunker_price_multiplier=1.0
            )
            
            if result['status'] not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                print("INFEASIBLE")
                high = mid
                continue
            
            current_signature = result['assignment_signature']
            current_profit = result['total_profit']
            
            if current_signature == base_signature:
                print(f"Same assignment (profit: ${current_profit:,.2f})")
                low = mid
                test_results.append({
                    'delay_days': mid,
                    'profit': current_profit,
                    'assignment_changed': False,
                    'signature': current_signature
                })
            else:
                print(f"Assignment CHANGED (profit: ${current_profit:,.2f})")
                threshold = mid
                high = mid
                test_results.append({
                    'delay_days': mid,
                    'profit': current_profit,
                    'assignment_changed': True,
                    'signature': current_signature
                })
        
        if threshold is None:
            threshold = high
        
        # Get detailed results at threshold
        threshold_result = self._run_optimization_with_scenario(
            port_delay_days=threshold,
            bunker_price_multiplier=1.0
        )
        
        # Get next-best solution (if different)
        next_best_result = None
        if threshold_result['assignment_signature'] != base_signature:
            next_best_result = threshold_result
        
        print()
        print("-" * 80)
        print(f"✓ Threshold found: {threshold:.2f} days")
        print()
        
        return {
            'scenario': 'Port Delay in China',
            'threshold_delay_days': threshold,
            'base_profit': self.base_portfolio_profit,
            'threshold_profit': threshold_result['total_profit'] if threshold_result['total_profit'] else None,
            'base_signature': base_signature,
            'threshold_signature': threshold_result['assignment_signature'],
            'assignment_changed': threshold_result['assignment_signature'] != base_signature,
            'test_results': test_results,
            'china_ports': self.china_ports
        }
    
    def analyze_bunker_price_scenario(self,
                                     max_multiplier: float = 3.0,
                                     step_size: float = 0.1,
                                     tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Analyze Scenario 2: Bunker Price Increase (VLSFO).
        
        Incrementally increases VLSFO prices and finds threshold where
        optimal solution changes.
        
        Parameters:
        -----------
        max_multiplier : float
            Maximum price multiplier to test (e.g., 3.0 = 300% of base)
        step_size : float
            Initial step size for binary search
        tolerance : float
            Tolerance for threshold detection
        
        Returns:
        --------
        dict with analysis results
        """
        print("=" * 80)
        print("SCENARIO 2: BUNKER PRICE INCREASE (VLSFO)")
        print("=" * 80)
        print()
        print(f"Base portfolio profit: ${self.base_portfolio_profit:,.2f}")
        print()
        
        # Get base assignment signature
        base_signature = self._get_assignment_signature(
            self.base_optimal_arcs, []
        )
        print(f"Base assignment signature: {base_signature}")
        print()
        
        # Binary search for threshold
        low = 1.0
        high = max_multiplier
        threshold = None
        test_results = []
        
        print("Binary search for threshold price multiplier...")
        print("-" * 80)
        
        while high - low > tolerance:
            mid = (low + high) / 2.0
            price_increase_pct = (mid - 1.0) * 100
            
            print(f"Testing multiplier: {mid:.3f}x ({price_increase_pct:.1f}% increase)...", end=" ")
            
            result = self._run_optimization_with_scenario(
                port_delay_days=0.0,
                bunker_price_multiplier=mid
            )
            
            if result['status'] not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
                print("INFEASIBLE")
                high = mid
                continue
            
            current_signature = result['assignment_signature']
            current_profit = result['total_profit']
            
            if current_signature == base_signature:
                print(f"Same assignment (profit: ${current_profit:,.2f})")
                low = mid
                test_results.append({
                    'multiplier': mid,
                    'price_increase_pct': price_increase_pct,
                    'profit': current_profit,
                    'assignment_changed': False,
                    'signature': current_signature
                })
            else:
                print(f"Assignment CHANGED (profit: ${current_profit:,.2f})")
                threshold = mid
                high = mid
                test_results.append({
                    'multiplier': mid,
                    'price_increase_pct': price_increase_pct,
                    'profit': current_profit,
                    'assignment_changed': True,
                    'signature': current_signature
                })
        
        if threshold is None:
            threshold = high
        
        # Get detailed results at threshold
        threshold_result = self._run_optimization_with_scenario(
            port_delay_days=0.0,
            bunker_price_multiplier=threshold
        )
        
        threshold_pct = (threshold - 1.0) * 100
        
        print()
        print("-" * 80)
        print(f"✓ Threshold found: {threshold:.3f}x ({threshold_pct:.1f}% increase)")
        print()
        
        return {
            'scenario': 'Bunker Price Increase',
            'threshold_multiplier': threshold,
            'threshold_price_increase_pct': threshold_pct,
            'base_profit': self.base_portfolio_profit,
            'threshold_profit': threshold_result['total_profit'] if threshold_result['total_profit'] else None,
            'base_signature': base_signature,
            'threshold_signature': threshold_result['assignment_signature'],
            'assignment_changed': threshold_result['assignment_signature'] != base_signature,
            'test_results': test_results
        }
    
    def generate_scenario_report(self, 
                                port_delay_results: Dict[str, Any],
                                bunker_price_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive scenario analysis report.
        
        Returns:
        --------
        dict with complete scenario analysis report
        """
        report = {
            'analysis_date': datetime.now().isoformat(),
            'base_case': {
                'portfolio_profit': self.base_portfolio_profit,
                'assignment_signature': self._get_assignment_signature(
                    self.base_optimal_arcs, []
                )
            },
            'scenario_1_port_delay': port_delay_results,
            'scenario_2_bunker_price': bunker_price_results,
            'economic_insights': self._generate_economic_insights(
                port_delay_results, bunker_price_results
            )
        }
        
        return report
    
    def _generate_economic_insights(self,
                                   port_delay_results: Dict[str, Any],
                                   bunker_price_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate economic intuition for scenario results."""
        insights = {
            'port_delay_insights': [],
            'bunker_price_insights': []
        }
        
        # Port delay insights
        if port_delay_results.get('assignment_changed'):
            threshold = port_delay_results['threshold_delay_days']
            insights['port_delay_insights'].append(
                f"At {threshold:.2f} days additional delay, the optimal solution changes. "
                f"This indicates that China port delays significantly impact voyage economics, "
                f"particularly for routes with Chinese discharge ports."
            )
            insights['port_delay_insights'].append(
                f"The threshold suggests that current optimal voyages are sensitive to port delays "
                f"beyond {threshold:.1f} days, at which point alternative routes or vessel assignments "
                f"become more profitable."
            )
        else:
            insights['port_delay_insights'].append(
                f"The optimal solution remains robust up to the maximum tested delay, "
                f"indicating strong resilience to port delays in China."
            )
        
        # Bunker price insights
        if bunker_price_results.get('assignment_changed'):
            threshold_pct = bunker_price_results['threshold_price_increase_pct']
            insights['bunker_price_insights'].append(
                f"At {threshold_pct:.1f}% fuel price increase, the optimal solution changes. "
                f"This threshold indicates the point where fuel costs become the dominant factor "
                f"in voyage selection."
            )
            insights['bunker_price_insights'].append(
                f"Higher fuel prices favor shorter-distance routes and more fuel-efficient vessels. "
                f"The threshold of {threshold_pct:.1f}% suggests that current optimal voyages are "
                f"balanced between fuel costs and other factors (freight rates, hire costs)."
            )
        else:
            insights['bunker_price_insights'].append(
                f"The optimal solution remains robust up to the maximum tested fuel price increase, "
                f"indicating that fuel costs are not the primary driver of the current optimal solution."
            )
        
        return insights

