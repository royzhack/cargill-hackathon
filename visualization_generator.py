"""
Enhanced Visualization Generator
=================================
Creates professional, report-ready visualizations for the chatbot.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
from io import BytesIO
import json

# Professional styling
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'font.family': 'sans-serif',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_portfolio_data():
    """Load portfolio data from JSON files. Uses risk_adjusted file as source of truth."""
    risk_path = Path('processed/portfolio_summary_risk_adjusted.json')
    
    portfolio = {}
    risk_adjusted = {}
    
    try:
        if risk_path.exists():
            with open(risk_path) as f:
                risk_adjusted = json.load(f)
            
            # Extract base data from risk_adjusted structure
            portfolio = {
                'total_portfolio_profit': risk_adjusted.get('base_portfolio_profit', 0),
                'assignments': []
            }
            # Convert risk-adjusted assignments to base format
            for assignment in risk_adjusted.get('assignments', []):
                base_assignment = {
                    'Vessel_Name': assignment.get('vessel', 'Unknown'),
                    'Cargo_ID': assignment.get('cargo', assignment.get('route', 'Unknown')),
                    'Leg_Profit': assignment.get('base_profit', 0),
                    'TCE_Leg': assignment.get('base_tce', 0),
                    'Leg_Days': assignment.get('voyage_days', 0),
                    'Cargo_Type': 'Market' if 'MARKET' in str(assignment.get('cargo', '')).upper() else 'Committed'
                }
                portfolio['assignments'].append(base_assignment)
    except Exception as e:
        print(f"Warning: Could not load risk-adjusted data: {e}")
    
    return portfolio, risk_adjusted

def create_portfolio_profit_chart():
    """Create enhanced portfolio profit breakdown chart."""
    portfolio, risk_adjusted = load_portfolio_data()
    
    if not portfolio or 'assignments' not in portfolio:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Portfolio Profit Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    assignments = portfolio['assignments']
    
    # 1. Profit by Vessel (Top Left)
    vessel_profits = {}
    for assignment in assignments:
        vessel = assignment.get('Vessel_Name', 'Unknown')
        profit = assignment.get('Leg_Profit', 0)
        if vessel in vessel_profits:
            vessel_profits[vessel] += profit
        else:
            vessel_profits[vessel] = profit
    
    vessels = list(vessel_profits.keys())
    profits = [vessel_profits[v] for v in vessels]
    colors = plt.cm.Set3(np.linspace(0, 1, len(vessels)))
    
    bars1 = axes[0, 0].bar(vessels, profits, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('Profit (USD)', fontweight='bold')
    axes[0, 0].set_title('Profit by Vessel', fontweight='bold', fontsize=13)
    axes[0, 0].ticklabel_format(style='plain', axis='y')
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'${height/1e6:.2f}M',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. Profit by Cargo Type (Top Right)
    committed_profit = sum(a.get('Leg_Profit', 0) for a in assignments 
                          if a.get('Cargo_Type') == 'Committed')
    market_profit = sum(a.get('Leg_Profit', 0) for a in assignments 
                       if a.get('Cargo_Type') == 'Market')
    
    cargo_types = ['Committed\nCargoes', 'Market\nCargoes']
    cargo_profits = [committed_profit, market_profit]
    cargo_colors = ['#2ca02c', '#ff7f0e']
    
    bars2 = axes[0, 1].bar(cargo_types, cargo_profits, color=cargo_colors, 
                           edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('Total Profit (USD)', fontweight='bold')
    axes[0, 1].set_title('Profit by Cargo Type', fontweight='bold', fontsize=13)
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar in bars2:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'${height/1e6:.2f}M',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. TCE Comparison (Bottom Left)
    tces = [a.get('TCE_Leg', 0) for a in assignments]
    vessel_names = [a.get('Vessel_Name', 'Unknown') for a in assignments]
    
    bars3 = axes[1, 0].bar(range(len(tces)), tces, color=plt.cm.viridis(np.linspace(0, 1, len(tces))),
                           edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('TCE (USD/day)', fontweight='bold')
    axes[1, 0].set_title('Time Charter Equivalent by Assignment', fontweight='bold', fontsize=13)
    axes[1, 0].set_xticks(range(len(vessel_names)))
    axes[1, 0].set_xticklabels([v[:10] for v in vessel_names], rotation=45, ha='right')
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')
    
    for i, (bar, tce) in enumerate(zip(bars3, tces)):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'${tce:,.0f}',
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 4. Risk Impact (Bottom Right)
    if risk_adjusted and 'assignments' in risk_adjusted:
        base_profits = [a.get('base_profit', 0) for a in risk_adjusted['assignments']]
        risk_profits = [a.get('risk_adjusted_profit', 0) for a in risk_adjusted['assignments']]
        risk_impact = [r - b for r, b in zip(risk_profits, base_profits)]
        
        x = np.arange(len(risk_impact))
        width = 0.35
        
        bars4a = axes[1, 1].bar(x - width/2, base_profits, width, label='Base Profit',
                               color='#1f77b4', edgecolor='black', linewidth=1)
        bars4b = axes[1, 1].bar(x + width/2, risk_profits, width, label='Risk-Adjusted',
                               color='#ff7f0e', edgecolor='black', linewidth=1)
        
        axes[1, 1].set_ylabel('Profit (USD)', fontweight='bold')
        axes[1, 1].set_title('Base vs Risk-Adjusted Profit', fontweight='bold', fontsize=13)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f"Leg {i+1}" for i in range(len(risk_impact))], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3, linestyle='--')
        axes[1, 1].ticklabel_format(style='plain', axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'Risk-adjusted data not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, color='gray')
        axes[1, 1].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = Path('diagrams/portfolio_profit_breakdown.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

def create_voyage_comparison_chart(vessel_names=None):
    """Create voyage comparison chart for specific vessels."""
    portfolio, _ = load_portfolio_data()
    
    if not portfolio or 'assignments' not in portfolio:
        return None
    
    assignments = portfolio['assignments']
    
    if vessel_names:
        assignments = [a for a in assignments if a.get('Vessel_Name') in vessel_names]
    
    if not assignments:
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Voyage Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    labels = [f"{a.get('Vessel_Name', 'Unknown')}\n→ {a.get('Cargo_ID', 'Unknown')}" 
              for a in assignments]
    profits = [a.get('Leg_Profit', 0) for a in assignments]
    tces = [a.get('TCE_Leg', 0) for a in assignments]
    days = [a.get('Leg_Days', 0) for a in assignments]
    
    # Profit comparison
    bars1 = axes[0].bar(range(len(profits)), profits, 
                       color=plt.cm.viridis(np.linspace(0, 1, len(profits))),
                       edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Profit (USD)', fontweight='bold')
    axes[0].set_title('Voyage Profit', fontweight='bold')
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels([l.replace(' → ', '\n→ ') for l in labels], 
                           rotation=45, ha='right', fontsize=9)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].ticklabel_format(style='plain', axis='y')
    
    for bar, profit in zip(bars1, profits):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                   f'${profit/1e6:.2f}M',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # TCE comparison
    bars2 = axes[1].bar(range(len(tces)), tces,
                       color=plt.cm.plasma(np.linspace(0, 1, len(tces))),
                       edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('TCE (USD/day)', fontweight='bold')
    axes[1].set_title('Time Charter Equivalent', fontweight='bold')
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels([l.replace(' → ', '\n→ ') for l in labels],
                           rotation=45, ha='right', fontsize=9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, tce in zip(bars2, tces):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                   f'${tce:,.0f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Voyage duration
    bars3 = axes[2].bar(range(len(days)), days,
                       color=plt.cm.coolwarm(np.linspace(0, 1, len(days))),
                       edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Days', fontweight='bold')
    axes[2].set_title('Voyage Duration', fontweight='bold')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels([l.replace(' → ', '\n→ ') for l in labels],
                           rotation=45, ha='right', fontsize=9)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--')
    
    for bar, day in zip(bars3, days):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                   f'{day:.1f}d',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = Path('diagrams/voyage_comparison.png')
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path

if __name__ == '__main__':
    # Generate enhanced visualizations
    print("Generating enhanced visualizations...")
    create_portfolio_profit_chart()
    create_voyage_comparison_chart()
    print("✓ Enhanced visualizations created")
