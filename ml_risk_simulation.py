"""
Machine Learning-Based Risk Simulation Module
==============================================
Simulates operational risks for Capesize vessel voyages using probabilistic models.

Risk Categories:
1. Adverse Weather Delays - Simulates weather-related voyage delays
2. Port Congestion - Models waiting time variability at ports
3. Waiting Time Variability - Uncertainty in laycan arrival timing
4. Voyage Uncertainty - Overall voyage duration variability

All risk models output probability distributions that adjust:
- Expected voyage duration
- Demurrage exposure
- Fuel consumption
- Effective revenue

Assumptions:
- Risk models use historical patterns and industry benchmarks
- Probabilistic outputs are integrated into deterministic optimization
- Risk adjustments preserve economic logic while adding uncertainty
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
import json


class MLRiskSimulator:
    """
    Machine Learning-based risk simulation for voyage operations.
    
    This module conceptually integrates ML techniques to simulate operational risks.
    In a production system, these would be trained on historical voyage data,
    weather patterns, port congestion records, and vessel performance data.
    
    For this implementation, we use probabilistic models based on:
    - Industry benchmarks for weather delays
    - Historical port congestion patterns
    - Seasonal variations
    - Route-specific risk factors
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the risk simulator.
        
        Parameters:
        -----------
        random_seed : int
            Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Risk model parameters (conceptually derived from ML training)
        # In production, these would be learned from historical data
        self._initialize_risk_parameters()
    
    def _initialize_risk_parameters(self):
        """
        Initialize risk model parameters based on industry benchmarks and evidence.
        
        DATA SOURCES & EVIDENCE:
        ------------------------
        These parameters are based on industry-standard benchmarks and should be
        replaced with actual historical voyage data when available.
        
        Weather Delays:
        - Industry studies show 2-5% of voyage time lost to weather (conservative: 1-3%)
        - North Atlantic routes: 3-8% delay risk (winter months higher)
        - Pacific routes: 1-4% delay risk
        - Source: Maritime weather routing studies, industry reports
        
        Port Congestion:
        - Major ports (China, Singapore): 20-30% congestion probability
        - Average delay when congested: 1.5-3 days (industry average: 2 days)
        - Source: Port performance reports, shipping industry statistics
        
        Voyage Duration Uncertainty:
        - Industry standard: 5-10% coefficient of variation
        - Conservative estimate: 8% CV
        - Source: Vessel performance analysis, voyage planning studies
        
        Demurrage:
        - Standard demurrage rates: $20,000-$30,000/day for Capesize
        - Demurrage occurrence: 10-20% of voyages (conservative: 15%)
        - Source: Charter party terms, industry benchmarks
        """
        
        # Weather delay parameters (days)
        # EVIDENCE: Industry studies show 1-3% of voyage time lost to weather on average
        # For a 30-day voyage, this translates to 0.3-0.9 days average delay
        # Using conservative 0.4 days base with lognormal distribution
        self.weather_params = {
            'base_delay_mean': 0.4,  # Base expected delay (evidence: 1-3% of voyage time)
            'base_delay_std': 0.8,   # Standard deviation (evidence: delays are right-skewed)
            'seasonal_factor': {     # Seasonal multipliers (evidence: winter months higher risk)
                1: 1.4,  # January (winter storms - North Atlantic/Europe routes)
                2: 1.3,  # February
                3: 1.1,  # March
                4: 0.9,  # April
                5: 0.8,  # May
                6: 0.7,  # June (summer - lower weather risk)
                7: 0.7,  # July
                8: 0.8,  # August
                9: 0.9,  # September
                10: 1.0, # October
                11: 1.2, # November (winter approaching)
                12: 1.3, # December
            },
            'route_risk_factors': {  # Route-specific risk multipliers (evidence: route studies)
                'transatlantic': 1.3,  # Higher weather risk (North Atlantic)
                'transpacific': 1.1,   # Moderate risk
                'asia_europe': 1.0,    # Baseline
                'asia_africa': 0.9,    # Lower risk (tropical routes)
                'asia_australia': 0.8, # Lower risk
                'default': 1.0,
            }
        }
        
        # Port congestion parameters
        # EVIDENCE: Major ports show 20-30% congestion probability
        # Average delay when congested: 1.5-3 days (using 2.0 days as industry average)
        self.congestion_params = {
            'base_congestion_prob': 0.22,  # 22% chance (evidence: industry average 20-30%)
            'congestion_delay_mean': 2.0,   # Mean delay (evidence: industry average 1.5-3 days)
            'congestion_delay_std': 1.2,     # Standard deviation
            'port_congestion_risk': {        # Port-specific (evidence: port performance data)
                'QINGDAO': 0.28,           # Major Chinese port - higher congestion
                'SHANGHAI': 0.32,          # Major hub - highest congestion
                'SINGAPORE': 0.24,         # Major hub - moderate congestion
                'ROTTERDAM': 0.22,         # European hub - moderate
                'PORT HEDLAND': 0.12,      # Export terminal - lower congestion
                'ITAGUAI': 0.18,           # Brazilian port - moderate
                'KAMSAR ANCHORAGE': 0.10,  # Anchor port - lower congestion
                'LIANYUNGANG': 0.26,       # Chinese port - higher congestion
                'FANGCHENG': 0.24,         # Chinese port - moderate congestion
                'default': 0.20,           # Default for other ports
            }
        }
        
        # Waiting time variability (laycan arrival uncertainty)
        # EVIDENCE: Vessel arrival timing shows ±1-2 days uncertainty
        # Early arrivals: 25-35% of cases (using 30%)
        # Late arrivals: 15-25% of cases (using 20%)
        self.waiting_params = {
            'early_arrival_prob': 0.30,      # 30% chance (evidence: 25-35% range)
            'early_arrival_mean': 1.2,      # Mean early arrival (evidence: 1-1.5 days)
            'late_arrival_prob': 0.20,       # 20% chance (evidence: 15-25% range)
            'late_arrival_mean': 1.8,        # Mean late arrival (evidence: 1.5-2.5 days)
            'arrival_uncertainty_std': 0.7,  # Standard deviation (evidence: ±1 day typical)
        }
        
        # Voyage duration uncertainty
        # EVIDENCE: Industry studies show 5-10% coefficient of variation
        # Using 7% as conservative middle estimate
        self.voyage_uncertainty_params = {
            'duration_variability': 0.07,  # 7% CV (evidence: 5-10% industry range)
            'speed_variation': 0.04,       # 4% speed variation (evidence: 3-5% typical)
        }
        
        # Demurrage exposure parameters
        # EVIDENCE: Standard demurrage rates $20,000-$30,000/day for Capesize
        # Demurrage occurrence: 10-20% of voyages (using 15% as middle estimate)
        self.demurrage_params = {
            'base_demurrage_rate': 25000,  # USD per day (evidence: $20k-$30k industry standard)
            'demurrage_prob': 0.15,        # 15% chance (evidence: 10-20% industry range)
            'demurrage_days_mean': 1.4,    # Mean demurrage days (evidence: 1-2 days typical)
            'demurrage_days_std': 0.9,     # Standard deviation
        }
    
    def simulate_weather_delay(
        self,
        voyage_date: datetime,
        route_type: str = 'default',
        distance_nm: float = 0.0
    ) -> Dict[str, float]:
        """
        Simulate adverse weather delays for a voyage.
        
        Parameters:
        -----------
        voyage_date : datetime
            Expected voyage start date
        route_type : str
            Route category (transatlantic, transpacific, etc.)
        distance_nm : float
            Voyage distance in nautical miles
        
        Returns:
        --------
        dict : {
            'delay_days': float,           # Expected delay in days
            'delay_p10': float,           # 10th percentile (optimistic)
            'delay_p50': float,           # 50th percentile (median)
            'delay_p90': float,           # 90th percentile (pessimistic)
            'delay_std': float,           # Standard deviation
            'risk_factor': float          # Overall risk multiplier
        }
        """
        month = voyage_date.month
        seasonal_factor = self.weather_params['seasonal_factor'].get(
            month, 1.0
        )
        
        route_factor = self.weather_params['route_risk_factors'].get(
            route_type, self.weather_params['route_risk_factors']['default']
        )
        
        # Distance factor: longer voyages have higher weather risk
        distance_factor = 1.0 + (distance_nm / 10000.0) * 0.1
        
        # Combined risk factor
        risk_factor = seasonal_factor * route_factor * distance_factor
        
        # Generate delay distribution (lognormal)
        mean_delay = self.weather_params['base_delay_mean'] * risk_factor
        std_delay = self.weather_params['base_delay_std'] * risk_factor
        
        # Sample from lognormal distribution
        delay_samples = np.random.lognormal(
            mean=np.log(mean_delay + 0.1),
            sigma=std_delay,
            size=10000
        )
        
        # Calculate percentiles
        delay_p10 = np.percentile(delay_samples, 10)
        delay_p50 = np.percentile(delay_samples, 50)
        delay_p90 = np.percentile(delay_samples, 90)
        
        return {
            'delay_days': delay_p50,  # Expected (median) delay
            'delay_p10': delay_p10,
            'delay_p50': delay_p50,
            'delay_p90': delay_p90,
            'delay_std': np.std(delay_samples),
            'risk_factor': risk_factor
        }
    
    def simulate_port_congestion(
        self,
        port_name: str,
        arrival_date: datetime
    ) -> Dict[str, float]:
        """
        Simulate port congestion delays.
        
        Parameters:
        -----------
        port_name : str
            Port name
        arrival_date : datetime
            Expected arrival date
        
        Returns:
        --------
        dict : {
            'congestion_delay_days': float,  # Expected congestion delay
            'congestion_prob': float,        # Probability of congestion
            'delay_p10': float,
            'delay_p50': float,
            'delay_p90': float
        }
        """
        # Get port-specific congestion risk
        port_risk = self.congestion_params['port_congestion_risk'].get(
            port_name.upper(),
            self.congestion_params['port_congestion_risk']['default']
        )
        
        # Seasonal variation (higher congestion in certain months)
        month = arrival_date.month
        seasonal_congestion = 1.0
        if month in [3, 4, 5, 9, 10, 11]:  # Peak shipping seasons
            seasonal_congestion = 1.2
        
        adjusted_prob = port_risk * seasonal_congestion
        adjusted_prob = min(adjusted_prob, 0.5)  # Cap at 50%
        
        # Simulate congestion delay if congestion occurs
        if np.random.random() < adjusted_prob:
            delay_mean = self.congestion_params['congestion_delay_mean']
            delay_std = self.congestion_params['congestion_delay_std']
            
            delay_samples = np.random.lognormal(
                mean=np.log(delay_mean + 0.1),
                sigma=delay_std,
                size=10000
            )
            
            delay_p50 = np.percentile(delay_samples, 50)
            delay_p10 = np.percentile(delay_samples, 10)
            delay_p90 = np.percentile(delay_samples, 90)
        else:
            delay_p50 = 0.0
            delay_p10 = 0.0
            delay_p90 = 0.0
        
        return {
            'congestion_delay_days': delay_p50,
            'congestion_prob': adjusted_prob,
            'delay_p10': delay_p10,
            'delay_p50': delay_p50,
            'delay_p90': delay_p90
        }
    
    def simulate_waiting_time_variability(
        self,
        expected_arrival: datetime,
        laycan_start: datetime,
        laycan_end: datetime
    ) -> Dict[str, float]:
        """
        Simulate waiting time variability at load port.
        
        Parameters:
        -----------
        expected_arrival : datetime
            Expected arrival date at load port
        laycan_start : datetime
            Laycan window start
        laycan_end : datetime
            Laycan window end
        
        Returns:
        --------
        dict : {
            'waiting_days': float,          # Expected waiting time
            'early_arrival_days': float,    # Days before laycan (if early)
            'late_arrival_days': float,      # Days after laycan (if late)
            'laycan_breach_prob': float,    # Probability of missing laycan
            'waiting_p10': float,
            'waiting_p50': float,
            'waiting_p90': float
        }
        """
        # Calculate deterministic waiting time
        if expected_arrival < laycan_start:
            base_waiting = (laycan_start - expected_arrival).days
        else:
            base_waiting = 0.0
        
        # Add variability
        arrival_uncertainty = np.random.normal(
            loc=0.0,
            scale=self.waiting_params['arrival_uncertainty_std']
        )
        
        # Early arrival scenario
        if np.random.random() < self.waiting_params['early_arrival_prob']:
            early_days = np.random.exponential(
                self.waiting_params['early_arrival_mean']
            )
            base_waiting = max(0, base_waiting - early_days)
        
        # Late arrival scenario
        late_arrival = False
        late_days = 0.0
        if np.random.random() < self.waiting_params['late_arrival_prob']:
            late_days = np.random.exponential(
                self.waiting_params['late_arrival_mean']
            )
            late_arrival = True
        
        # Calculate laycan breach probability
        adjusted_arrival = expected_arrival + timedelta(days=arrival_uncertainty + late_days)
        laycan_breach_prob = 1.0 if adjusted_arrival > laycan_end else 0.0
        
        # Generate waiting time distribution
        waiting_samples = []
        for _ in range(10000):
            sample_arrival = expected_arrival + timedelta(
                days=np.random.normal(0, self.waiting_params['arrival_uncertainty_std'])
            )
            if sample_arrival < laycan_start:
                waiting = (laycan_start - sample_arrival).days
            else:
                waiting = 0.0
            waiting_samples.append(waiting)
        
        waiting_p10 = np.percentile(waiting_samples, 10)
        waiting_p50 = np.percentile(waiting_samples, 50)
        waiting_p90 = np.percentile(waiting_samples, 90)
        
        return {
            'waiting_days': waiting_p50,
            'early_arrival_days': max(0, base_waiting - waiting_p50) if base_waiting > 0 else 0,
            'late_arrival_days': late_days if late_arrival else 0.0,
            'laycan_breach_prob': laycan_breach_prob,
            'waiting_p10': waiting_p10,
            'waiting_p50': waiting_p50,
            'waiting_p90': waiting_p90
        }
    
    def simulate_voyage_uncertainty(
        self,
        base_duration_days: float,
        distance_nm: float
    ) -> Dict[str, float]:
        """
        Simulate overall voyage duration uncertainty.
        
        Parameters:
        -----------
        base_duration_days : float
            Base voyage duration (deterministic calculation)
        distance_nm : float
            Total voyage distance
        
        Returns:
        --------
        dict : {
            'adjusted_duration_days': float,  # Risk-adjusted duration
            'duration_p10': float,
            'duration_p50': float,
            'duration_p90': float,
            'duration_std': float
        }
        """
        # Apply variability
        cv = self.voyage_uncertainty_params['duration_variability']
        std = base_duration_days * cv
        
        # Generate duration distribution
        duration_samples = np.random.normal(
            loc=base_duration_days,
            scale=std,
            size=10000
        )
        
        # Ensure non-negative
        duration_samples = np.maximum(duration_samples, base_duration_days * 0.9)
        
        duration_p10 = np.percentile(duration_samples, 10)
        duration_p50 = np.percentile(duration_samples, 50)
        duration_p90 = np.percentile(duration_samples, 90)
        
        return {
            'adjusted_duration_days': duration_p50,
            'duration_p10': duration_p10,
            'duration_p50': duration_p50,
            'duration_p90': duration_p90,
            'duration_std': np.std(duration_samples)
        }
    
    def simulate_demurrage_exposure(
        self,
        voyage_duration_days: float,
        port_days: float
    ) -> Dict[str, float]:
        """
        Simulate demurrage exposure risk.
        
        Parameters:
        -----------
        voyage_duration_days : float
            Total voyage duration
        port_days : float
            Days spent in port
        
        Returns:
        --------
        dict : {
            'demurrage_days': float,        # Expected demurrage days
            'demurrage_cost_usd': float,     # Expected demurrage cost
            'demurrage_prob': float,         # Probability of demurrage
            'demurrage_p10': float,
            'demurrage_p50': float,
            'demurrage_p90': float
        }
        """
        # Demurrage risk increases with port time
        port_time_factor = min(port_days / 10.0, 2.0)  # Cap at 2x
        adjusted_prob = self.demurrage_params['demurrage_prob'] * port_time_factor
        adjusted_prob = min(adjusted_prob, 0.4)  # Cap at 40%
        
        if np.random.random() < adjusted_prob:
            demurrage_days_mean = self.demurrage_params['demurrage_days_mean']
            demurrage_days_std = self.demurrage_params['demurrage_days_std']
            
            demurrage_samples = np.random.lognormal(
                mean=np.log(demurrage_days_mean + 0.1),
                sigma=demurrage_days_std,
                size=10000
            )
            
            demurrage_p50 = np.percentile(demurrage_samples, 50)
            demurrage_p10 = np.percentile(demurrage_samples, 10)
            demurrage_p90 = np.percentile(demurrage_samples, 90)
        else:
            demurrage_p50 = 0.0
            demurrage_p10 = 0.0
            demurrage_p90 = 0.0
        
        demurrage_cost = demurrage_p50 * self.demurrage_params['base_demurrage_rate']
        
        return {
            'demurrage_days': demurrage_p50,
            'demurrage_cost_usd': demurrage_cost,
            'demurrage_prob': adjusted_prob,
            'demurrage_p10': demurrage_p10,
            'demurrage_p50': demurrage_p50,
            'demurrage_p90': demurrage_p90
        }
    
    def simulate_fuel_consumption_adjustment(
        self,
        base_fuel_consumption_mt: float,
        weather_delay_days: float,
        voyage_duration_days: float
    ) -> Dict[str, float]:
        """
        Adjust fuel consumption based on risk factors.
        
        Parameters:
        -----------
        base_fuel_consumption_mt : float
            Base fuel consumption (deterministic)
        weather_delay_days : float
            Weather-related delay
        voyage_duration_days : float
            Total voyage duration
        
        Returns:
        --------
        dict : {
            'adjusted_fuel_mt': float,      # Risk-adjusted fuel consumption
            'fuel_adjustment_pct': float,   # Percentage adjustment
            'fuel_p10': float,
            'fuel_p50': float,
            'fuel_p90': float
        }
        """
        # Weather delays increase fuel consumption (idling, slower speeds)
        weather_factor = 1.0 + (weather_delay_days / voyage_duration_days) * 0.15
        
        # Speed variation affects consumption
        speed_variation = self.voyage_uncertainty_params['speed_variation']
        speed_factor = 1.0 + np.random.normal(0, speed_variation)
        
        # Combined adjustment
        fuel_adjustment = weather_factor * speed_factor
        
        adjusted_fuel = base_fuel_consumption_mt * fuel_adjustment
        
        # Generate distribution
        fuel_samples = np.random.normal(
            loc=adjusted_fuel,
            scale=adjusted_fuel * 0.05,  # 5% variability
            size=10000
        )
        
        fuel_p10 = np.percentile(fuel_samples, 10)
        fuel_p50 = np.percentile(fuel_samples, 50)
        fuel_p90 = np.percentile(fuel_samples, 90)
        
        return {
            'adjusted_fuel_mt': fuel_p50,
            'fuel_adjustment_pct': (fuel_adjustment - 1.0) * 100,
            'fuel_p10': fuel_p10,
            'fuel_p50': fuel_p50,
            'fuel_p90': fuel_p90
        }
    
    def simulate_comprehensive_risk(
        self,
        voyage_date: datetime,
        load_port: str,
        discharge_port: str,
        base_duration_days: float,
        base_fuel_mt: float,
        ballast_distance_nm: float,
        laden_distance_nm: float,
        laycan_start: datetime,
        laycan_end: datetime,
        port_days: float,
        route_type: str = 'default'
    ) -> Dict[str, any]:
        """
        Comprehensive risk simulation combining all risk factors.
        
        Returns a complete risk profile for a voyage.
        """
        # Simulate all risk components
        weather_risk = self.simulate_weather_delay(
            voyage_date, route_type, ballast_distance_nm + laden_distance_nm
        )
        
        load_congestion = self.simulate_port_congestion(
            load_port, voyage_date + timedelta(days=base_duration_days * 0.3)
        )
        
        discharge_congestion = self.simulate_port_congestion(
            discharge_port, voyage_date + timedelta(days=base_duration_days * 0.8)
        )
        
        expected_arrival = voyage_date + timedelta(days=base_duration_days * 0.3)
        waiting_risk = self.simulate_waiting_time_variability(
            expected_arrival, laycan_start, laycan_end
        )
        
        voyage_uncertainty = self.simulate_voyage_uncertainty(
            base_duration_days, ballast_distance_nm + laden_distance_nm
        )
        
        demurrage_risk = self.simulate_demurrage_exposure(
            base_duration_days, port_days
        )
        
        fuel_adjustment = self.simulate_fuel_consumption_adjustment(
            base_fuel_mt, weather_risk['delay_days'], base_duration_days
        )
        
        # Aggregate total delay
        total_delay = (
            weather_risk['delay_days'] +
            load_congestion['congestion_delay_days'] +
            discharge_congestion['congestion_delay_days'] +
            waiting_risk['waiting_days']
        )
        
        # Adjusted voyage duration
        adjusted_duration = voyage_uncertainty['adjusted_duration_days'] + total_delay
        
        return {
            'weather_risk': weather_risk,
            'load_congestion': load_congestion,
            'discharge_congestion': discharge_congestion,
            'waiting_risk': waiting_risk,
            'voyage_uncertainty': voyage_uncertainty,
            'demurrage_risk': demurrage_risk,
            'fuel_adjustment': fuel_adjustment,
            'total_delay_days': total_delay,
            'adjusted_duration_days': adjusted_duration,
            'risk_summary': {
                'total_delay_p10': (
                    weather_risk['delay_p10'] +
                    load_congestion['delay_p10'] +
                    discharge_congestion['delay_p10'] +
                    waiting_risk['waiting_p10']
                ),
                'total_delay_p50': total_delay,
                'total_delay_p90': (
                    weather_risk['delay_p90'] +
                    load_congestion['delay_p90'] +
                    discharge_congestion['delay_p90'] +
                    waiting_risk['waiting_p90']
                ),
            }
        }
    
    def calculate_risk_adjusted_profit(
        self,
        base_profit: float,
        base_fuel_cost: float,
        base_hire_cost: float,
        base_duration_days: float,
        risk_profile: Dict[str, any],
        hire_rate_per_day: float
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted profit from base leg economics and risk profile.
        
        This method integrates risk simulation results into the economic calculation
        to produce a risk-adjusted profit that accounts for operational uncertainties.
        
        Parameters:
        -----------
        base_profit : float
            Base (deterministic) profit
        base_fuel_cost : float
            Base fuel cost
        base_hire_cost : float
            Base hire cost
        base_duration_days : float
            Base voyage duration
        risk_profile : dict
            Risk profile from simulate_comprehensive_risk()
        hire_rate_per_day : float
            Vessel hire rate per day
        
        Returns:
        --------
        dict : {
            'risk_adjusted_profit': float,
            'risk_adjusted_fuel_cost': float,
            'risk_adjusted_hire_cost': float,
            'risk_adjusted_duration_days': float,
            'total_risk_cost': float,
            'risk_impact': float
        }
        """
        # Risk-adjusted fuel cost
        fuel_adjustment_pct = risk_profile['fuel_adjustment']['fuel_adjustment_pct'] / 100.0
        risk_adjusted_fuel_cost = base_fuel_cost * (1.0 + fuel_adjustment_pct)
        
        # Risk-adjusted duration
        risk_adjusted_duration = risk_profile['adjusted_duration_days']
        
        # Additional hire cost due to delays
        additional_duration = risk_adjusted_duration - base_duration_days
        additional_hire_cost = additional_duration * hire_rate_per_day
        
        # Demurrage cost
        demurrage_cost = risk_profile['demurrage_risk']['demurrage_cost_usd']
        
        # Total risk costs
        total_risk_cost = (
            (risk_adjusted_fuel_cost - base_fuel_cost) +  # Additional fuel
            additional_hire_cost +                        # Additional hire
            demurrage_cost                                # Demurrage
        )
        
        # Risk-adjusted profit
        risk_adjusted_profit = base_profit - total_risk_cost
        
        # Risk impact (negative means profit reduction)
        risk_impact = -total_risk_cost
        
        return {
            'risk_adjusted_profit': risk_adjusted_profit,
            'risk_adjusted_fuel_cost': risk_adjusted_fuel_cost,
            'risk_adjusted_hire_cost': base_hire_cost + additional_hire_cost,
            'risk_adjusted_duration_days': risk_adjusted_duration,
            'total_risk_cost': total_risk_cost,
            'risk_impact': risk_impact,
            'additional_fuel_cost': risk_adjusted_fuel_cost - base_fuel_cost,
            'additional_hire_cost': additional_hire_cost,
            'demurrage_cost': demurrage_cost
        }

