"""
Test script to verify ML risk simulation integration and update portfolio files.

This script:
1. Tests ML risk simulation modules
2. Applies risk simulation to existing assignments
3. Updates portfolio summary with risk-adjusted results
4. Validates risk-adjusted calculations
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# Import ML modules
try:
    from ml_risk_simulation import MLRiskSimulator
    from explainability import VoyageExplainability
    ML_ENABLED = True
    print("✓ ML modules imported successfully")
except ImportError as e:
    print(f"✗ Error importing ML modules: {e}")
    ML_ENABLED = False
    exit(1)

# Load existing assignments
assignments_file = Path("multileg_assignments.csv")
if not assignments_file.exists():
    print(f"✗ Error: {assignments_file} not found")
    exit(1)

assignments_df = pd.read_csv(assignments_file)
print(f"✓ Loaded {len(assignments_df)} assignments from {assignments_file}")

# Load vessel and cargo data
data_dir = Path("data")
cargill_vessels = pd.read_csv(data_dir / "Cargill_Capesize_Vessels.csv")
market_vessels = pd.read_csv(data_dir / "Market_Vessels_Formatted.csv")
cargill_cargoes = pd.read_csv(data_dir / "Cargill_Committed_Cargoes_Structured.csv")
market_cargoes = pd.read_csv(data_dir / "Market_Cargoes_Structured.csv")

# Process dates
cargill_vessels['ETD'] = pd.to_datetime(cargill_vessels['ETD'])
market_vessels['ETD'] = pd.to_datetime(market_vessels['ETD'])
cargill_cargoes['Laycan_Start'] = pd.to_datetime(cargill_cargoes['Laycan_Start'])
cargill_cargoes['Laycan_End'] = pd.to_datetime(cargill_cargoes['Laycan_End'])
market_cargoes['Laycan_Start'] = pd.to_datetime(market_cargoes['Laycan_Start'])
market_cargoes['Laycan_End'] = pd.to_datetime(market_cargoes['Laycan_End'])

# Initialize risk simulator
risk_simulator = MLRiskSimulator(random_seed=42)
explainability = VoyageExplainability()
print("✓ Risk simulator and explainability engine initialized")

# Apply risk simulation
print("\n" + "="*80)
print("APPLYING ML RISK SIMULATION")
print("="*80)

risk_adjusted_assignments = []
risk_profiles = {}

for idx, assignment in assignments_df.iterrows():
    vessel_name = assignment['Vessel_Name']
    cargo_id = assignment['Cargo_ID']
    load_port = assignment['Load_Port']
    discharge_port = assignment['Discharge_Port']
    
    # Get vessel data
    if assignment['Vessel_Type'] == 'Cargill':
        vessel_row = cargill_vessels[cargill_vessels['Vessel Name'] == vessel_name].iloc[0]
        etd = vessel_row['ETD']
    else:
        vessel_row = market_vessels[market_vessels['Vessel Name'] == vessel_name].iloc[0]
        etd = vessel_row['ETD']
    
    # Get cargo data
    if cargo_id.startswith('CARGILL_'):
        cargo_row = cargill_cargoes[cargill_cargoes.index == int(cargo_id.split('_')[1]) - 1].iloc[0]
        laycan_start = cargo_row['Laycan_Start']
        laycan_end = cargo_row['Laycan_End']
    else:
        cargo_row = market_cargoes[market_cargoes.index == int(cargo_id.split('_')[1]) - 1].iloc[0]
        laycan_start = cargo_row['Laycan_Start']
        laycan_end = cargo_row['Laycan_End']
    
    # Get base parameters
    base_duration = assignment['Leg_Days']
    base_fuel_mt = assignment['Fuel_Cost'] / 500  # Rough estimate
    ballast_dist = assignment['Ballast_Distance']
    laden_dist = assignment['Laden_Distance']
    port_days = assignment['Days_Port']
    
    # Determine route type
    route_type = 'default'
    if 'BRAZIL' in load_port.upper() or 'BRAZIL' in discharge_port.upper():
        if 'CHINA' in discharge_port.upper():
            route_type = 'transpacific'
    elif 'AUSTRALIA' in load_port.upper() and 'CHINA' in discharge_port.upper():
        route_type = 'asia_australia'
    elif 'AFRICA' in load_port.upper() or 'KAMSAR' in load_port.upper():
        route_type = 'asia_africa'
    
    # Simulate comprehensive risk
    risk_profile = risk_simulator.simulate_comprehensive_risk(
        voyage_date=etd,
        load_port=load_port,
        discharge_port=discharge_port,
        base_duration_days=base_duration,
        base_fuel_mt=base_fuel_mt,
        ballast_distance_nm=ballast_dist,
        laden_distance_nm=laden_dist,
        laycan_start=laycan_start,
        laycan_end=laycan_end,
        port_days=port_days,
        route_type=route_type
    )
    
    risk_profiles[f"{vessel_name}_{cargo_id}"] = risk_profile
    
    # Create risk-adjusted row
    risk_adj_row = assignment.copy()
    
    # Add risk metrics
    risk_adj_row['Risk_Adjusted_Duration_Days'] = risk_profile['adjusted_duration_days']
    risk_adj_row['Total_Risk_Delay_Days'] = risk_profile['total_delay_days']
    risk_adj_row['Weather_Delay_Days'] = risk_profile['weather_risk']['delay_days']
    risk_adj_row['Congestion_Delay_Days'] = (
        risk_profile['load_congestion']['congestion_delay_days'] +
        risk_profile['discharge_congestion']['congestion_delay_days']
    )
    risk_adj_row['Waiting_Days_Risk'] = risk_profile['waiting_risk']['waiting_days']
    risk_adj_row['Laycan_Breach_Prob'] = risk_profile['waiting_risk']['laycan_breach_prob']
    risk_adj_row['Demurrage_Cost_USD'] = risk_profile['demurrage_risk']['demurrage_cost_usd']
    risk_adj_row['Demurrage_Prob'] = risk_profile['demurrage_risk']['demurrage_prob']
    risk_adj_row['Fuel_Adjustment_Pct'] = risk_profile['fuel_adjustment']['fuel_adjustment_pct']
    risk_adj_row['Risk_Adjusted_Fuel_Cost'] = (
        assignment['Fuel_Cost'] * (1 + risk_profile['fuel_adjustment']['fuel_adjustment_pct'] / 100)
    )
    
    # Calculate risk-adjusted profit
    additional_costs = (
        risk_adj_row['Demurrage_Cost_USD'] +
        (risk_adj_row['Risk_Adjusted_Fuel_Cost'] - assignment['Fuel_Cost'])
    )
    risk_adj_row['Risk_Adjusted_Profit'] = assignment['Leg_Profit'] - additional_costs
    risk_adj_row['Risk_Adjusted_TCE'] = (
        risk_adj_row['Risk_Adjusted_Profit'] / risk_adj_row['Risk_Adjusted_Duration_Days']
        if risk_adj_row['Risk_Adjusted_Duration_Days'] > 0 else 0
    )
    
    risk_adjusted_assignments.append(risk_adj_row)

# Create DataFrame
risk_adjusted_df = pd.DataFrame(risk_adjusted_assignments)

# Save risk-adjusted assignments
risk_adjusted_df.to_csv('multileg_assignments_risk_adjusted.csv', index=False)
print(f"\n✓ Saved risk-adjusted assignments to: multileg_assignments_risk_adjusted.csv")

# Print summary
print("\n" + "="*80)
print("RISK SIMULATION SUMMARY")
print("="*80)
print(f"Base Portfolio Profit: ${assignments_df['Leg_Profit'].sum():,.2f}")
print(f"Risk-Adjusted Profit: ${risk_adjusted_df['Risk_Adjusted_Profit'].sum():,.2f}")
print(f"Risk Impact: ${risk_adjusted_df['Risk_Adjusted_Profit'].sum() - assignments_df['Leg_Profit'].sum():,.2f}")
print(f"\nAverage Total Delay: {risk_adjusted_df['Total_Risk_Delay_Days'].mean():.2f} days")
print(f"Average Weather Delay: {risk_adjusted_df['Weather_Delay_Days'].mean():.2f} days")
print(f"Average Congestion Delay: {risk_adjusted_df['Congestion_Delay_Days'].mean():.2f} days")
print(f"Total Demurrage Exposure: ${risk_adjusted_df['Demurrage_Cost_USD'].sum():,.2f}")
print(f"Average Fuel Adjustment: {risk_adjusted_df['Fuel_Adjustment_Pct'].mean():.2f}%")

# Update portfolio summary
print("\n" + "="*80)
print("UPDATING PORTFOLIO SUMMARY")
print("="*80)

portfolio_summary = {
    "total_vessels": len(assignments_df['Vessel_Name'].unique()),
    "cargill_vessels": len(assignments_df[assignments_df['Vessel_Type'] == 'Cargill']['Vessel_Name'].unique()),
    "market_vessels": len(assignments_df[assignments_df['Vessel_Type'] == 'Market']['Vessel_Name'].unique()),
    "total_cargoes": len(assignments_df),
    "committed_cargoes": len(assignments_df[assignments_df['Cargo_Type'] == 'Committed']),
    "market_cargoes": len(assignments_df[assignments_df['Cargo_Type'] == 'Market']),
    "base_portfolio_profit": float(assignments_df['Leg_Profit'].sum()),
    "risk_adjusted_portfolio_profit": float(risk_adjusted_df['Risk_Adjusted_Profit'].sum()),
    "risk_impact": float(risk_adjusted_df['Risk_Adjusted_Profit'].sum() - assignments_df['Leg_Profit'].sum()),
    "average_tce": float(assignments_df['TCE_Leg'].mean()),
    "risk_adjusted_average_tce": float(risk_adjusted_df['Risk_Adjusted_TCE'].mean()),
    "total_risk_delay_days": float(risk_adjusted_df['Total_Risk_Delay_Days'].sum()),
    "total_demurrage_exposure": float(risk_adjusted_df['Demurrage_Cost_USD'].sum()),
    "average_fuel_adjustment_pct": float(risk_adjusted_df['Fuel_Adjustment_Pct'].mean()),
    "assignments": [],
    "risk_metrics": {
        "average_weather_delay_days": float(risk_adjusted_df['Weather_Delay_Days'].mean()),
        "average_congestion_delay_days": float(risk_adjusted_df['Congestion_Delay_Days'].mean()),
        "average_waiting_days_risk": float(risk_adjusted_df['Waiting_Days_Risk'].mean()),
        "average_laycan_breach_prob": float(risk_adjusted_df['Laycan_Breach_Prob'].mean()),
        "average_demurrage_prob": float(risk_adjusted_df['Demurrage_Prob'].mean()),
    }
}

# Add assignment details
for idx, assignment in assignments_df.iterrows():
    risk_adj = risk_adjusted_df.iloc[idx]
    portfolio_summary["assignments"].append({
        "vessel": assignment['Vessel_Name'],
        "fleet": assignment['Vessel_Type'],
        "cargo": assignment['Cargo_ID'],
        "route": f"{assignment['Load_Port']} → {assignment['Discharge_Port']}",
        "base_profit": float(assignment['Leg_Profit']),
        "risk_adjusted_profit": float(risk_adj['Risk_Adjusted_Profit']),
        "base_tce": float(assignment['TCE_Leg']),
        "risk_adjusted_tce": float(risk_adj['Risk_Adjusted_TCE']),
        "voyage_days": float(assignment['Leg_Days']),
        "risk_adjusted_days": float(risk_adj['Risk_Adjusted_Duration_Days']),
        "total_delay_days": float(risk_adj['Total_Risk_Delay_Days']),
        "demurrage_cost": float(risk_adj['Demurrage_Cost_USD']),
        "fuel_adjustment_pct": float(risk_adj['Fuel_Adjustment_Pct']),
    })

# Save updated portfolio summary
output_dir = Path("processed")
output_dir.mkdir(exist_ok=True)
with open(output_dir / "portfolio_summary_risk_adjusted.json", "w") as f:
    json.dump(portfolio_summary, f, indent=2, default=str)

print(f"✓ Updated portfolio summary saved to: {output_dir / 'portfolio_summary_risk_adjusted.json'}")

# Validate calculations
print("\n" + "="*80)
print("VALIDATION")
print("="*80)

errors = []

# Check profit calculation
for idx in range(len(assignments_df)):
    base_profit = assignments_df.iloc[idx]['Leg_Profit']
    risk_adj_profit = risk_adjusted_df.iloc[idx]['Risk_Adjusted_Profit']
    demurrage = risk_adjusted_df.iloc[idx]['Demurrage_Cost_USD']
    fuel_adj = risk_adjusted_df.iloc[idx]['Risk_Adjusted_Fuel_Cost'] - assignments_df.iloc[idx]['Fuel_Cost']
    expected_profit = base_profit - demurrage - fuel_adj
    
    if abs(risk_adj_profit - expected_profit) > 0.01:
        errors.append(f"Profit mismatch for {assignments_df.iloc[idx]['Vessel_Name']} → {assignments_df.iloc[idx]['Cargo_ID']}: "
                     f"expected {expected_profit:.2f}, got {risk_adj_profit:.2f}")

# Check TCE calculation
for idx in range(len(risk_adjusted_df)):
    profit = risk_adjusted_df.iloc[idx]['Risk_Adjusted_Profit']
    days = risk_adjusted_df.iloc[idx]['Risk_Adjusted_Duration_Days']
    tce = risk_adjusted_df.iloc[idx]['Risk_Adjusted_TCE']
    if days > 0:
        expected_tce = profit / days
        if abs(tce - expected_tce) > 0.01:
            errors.append(f"TCE mismatch for {risk_adjusted_df.iloc[idx]['Vessel_Name']} → {risk_adjusted_df.iloc[idx]['Cargo_ID']}: "
                         f"expected {expected_tce:.2f}, got {tce:.2f}")

if errors:
    print("✗ Validation errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("✓ All validations passed!")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)

