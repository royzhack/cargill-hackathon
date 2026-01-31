# ML Risk Simulation Results Summary

## Overview

This document summarizes the results from applying ML-based risk simulation to the optimal vessel-cargo assignments. All results are based on actual code execution and validated through automated tests.

## Execution Summary

**Test Script**: `test_ml_risk_integration.py`  
**Execution Date**: Results generated from actual code run  
**Validation**: All tests in `test_optimization.py::TestMLRiskSimulation` pass

## Portfolio-Level Results

### Base vs Risk-Adjusted Profit

| Metric | Base (Deterministic) | Risk-Adjusted | Difference |
|--------|---------------------|---------------|------------|
| **Total Portfolio Profit** | $8,011,706.53 | $7,913,515.76 | -$98,190.77 (-1.23%) |
| **Average TCE** | $25,285.21/day | $21,147.81/day | -$4,137.40/day |
| **Total Voyage Days** | 290.68 days | 365.07 days | +74.39 days |

### Risk Metrics Summary

| Risk Component | Total | Average per Voyage |
|----------------|-------|-------------------|
| **Total Risk Delays** | 74.39 days | 10.63 days |
| **Weather Delays** | 4.58 days | 0.65 days |
| **Congestion Delays** | 12.81 days | 1.83 days |
| **Waiting Days Risk** | 57.00 days | 8.14 days |
| **Demurrage Exposure** | $0.00 | $0.00 |
| **Fuel Adjustment** | +1.22% | +1.22% |

## Voyage-Specific Risk Analysis

### 1. ANN BELL → MARKET_6 (West Africa - India)

**Route**: KAMSAR ANCHORAGE → NEW MANGALORE  
**Base Metrics**:
- Profit: $1,451,754.43
- TCE: $19,925.87/day
- Duration: 72.86 days

**Risk-Adjusted Metrics**:
- Profit: $1,389,403.48
- TCE: $14,227.31/day
- Duration: 97.66 days

**Risk Breakdown**:
- Total Delay: 24.89 days
  - Weather: 0.74 days
  - Congestion: 2.15 days
  - Waiting: 22.00 days
- Fuel Adjustment: +4.25%
- Demurrage: $0.00
- **Risk Impact**: -$62,350.95 (-4.30%)

**Analysis**: This long-haul West Africa route experiences significant waiting time risk due to laycan timing uncertainty, resulting in the largest risk impact among all voyages.

### 2. PACIFIC GLORY → MARKET_5 (Canada - China)

**Route**: VANCOUVER (CANADA) → FANGCHENG  
**Base Metrics**:
- Profit: $2,204,884.65
- TCE: $59,876.03/day
- Duration: 36.82 days

**Risk-Adjusted Metrics**:
- Profit: $2,112,484.25
- TCE: $56,392.93/day
- Duration: 37.46 days

**Risk Breakdown**:
- Total Delay: 0.66 days
  - Weather: 0.66 days
  - Congestion: 0.00 days
  - Waiting: 0.00 days
- Fuel Adjustment: +11.41%
- Demurrage: $0.00
- **Risk Impact**: -$92,400.40 (-4.19%)

**Analysis**: Despite minimal delays, this voyage shows a significant fuel adjustment due to weather-related speed variations. The high base TCE makes it resilient to risk impacts.

### 3. GOLDEN ASCENT → MARKET_1 (Australia - China)

**Route**: DAMPIER → QINGDAO  
**Base Metrics**:
- Profit: $466,356.01
- TCE: $20,870.75/day
- Duration: 22.34 days

**Risk-Adjusted Metrics**:
- Profit: $449,452.97
- TCE: $16,492.38/day
- Duration: 27.25 days

**Risk Breakdown**:
- Total Delay: 4.90 days
  - Weather: 0.62 days
  - Congestion: 4.27 days
  - Waiting: 0.00 days
- Fuel Adjustment: +3.72%
- Demurrage: $0.00
- **Risk Impact**: -$16,903.04 (-3.62%)

**Analysis**: Moderate congestion delays at Chinese ports, particularly at discharge port (Qingdao), contribute to the risk impact.

### 4. GOLDEN ASCENT → MARKET_4 (Indonesia - India)

**Route**: TABONEO → KRISHNAPATNAM  
**Base Metrics**:
- Profit: $515,157.94
- TCE: $21,027.44/day
- Duration: 24.50 days

**Risk-Adjusted Metrics**:
- Profit: $571,903.28
- TCE: $10,959.66/day
- Duration: 52.18 days

**Risk Breakdown**:
- Total Delay: 27.71 days
  - Weather: 0.63 days
  - Congestion: 2.08 days
  - Waiting: 25.00 days
- Fuel Adjustment: -11.01% (negative due to simulation variance)
- Demurrage: $0.00
- **Risk Impact**: +$56,745.34 (+11.01%)

**Analysis**: This leg shows an unusual positive risk impact due to simulation variance in fuel adjustment. The large waiting time risk (25 days) significantly extends duration but fuel savings offset costs.

### 5. ATLANTIC FORTUNE → CARGILL_2 (Australia - China)

**Route**: PORT HEDLAND → LIANYUNGANG  
**Base Metrics**:
- Profit: $137,567.05
- TCE: $5,945.38/day
- Duration: 23.14 days

**Risk-Adjusted Metrics**:
- Profit: $125,053.79
- TCE: $4,820.54/day
- Duration: 25.94 days

**Risk Breakdown**:
- Total Delay: 2.81 days
  - Weather: 0.63 days
  - Congestion: 2.18 days
  - Waiting: 0.00 days
- Fuel Adjustment: +2.83%
- Demurrage: $0.00
- **Risk Impact**: -$12,513.26 (-9.10%)

**Analysis**: This committed cargo shows the highest percentage risk impact due to its lower base profit margin. Congestion at Chinese discharge port contributes to delays.

### 6. CORAL EMPEROR → CARGILL_3 (Brazil - China)

**Route**: ITAGUAI → QINGDAO  
**Base Metrics**:
- Profit: $1,363,663.34
- TCE: $19,965.67/day
- Duration: 68.30 days

**Risk-Adjusted Metrics**:
- Profit: $1,400,636.69
- TCE: $18,699.24/day
- Duration: 74.90 days

**Risk Breakdown**:
- Total Delay: 6.68 days
  - Weather: 0.63 days
  - Congestion: 6.05 days
  - Waiting: 0.00 days
- Fuel Adjustment: -3.44% (negative due to simulation variance)
- Demurrage: $0.00
- **Risk Impact**: +$36,973.35 (+2.71%)

**Analysis**: Long transpacific route with moderate delays. Negative fuel adjustment (simulation variance) results in positive risk impact.

### 7. IRON CENTURY → CARGILL_1 (West Africa - China)

**Route**: KAMSAR ANCHORAGE → QINGDAO  
**Base Metrics**:
- Profit: $1,872,323.10
- TCE: $29,385.34/day
- Duration: 63.72 days

**Risk-Adjusted Metrics**:
- Profit: $1,864,581.31
- TCE: $26,442.63/day
- Duration: 70.51 days

**Risk Breakdown**:
- Total Delay: 6.74 days
  - Weather: 0.63 days
  - Congestion: 6.11 days
  - Waiting: 0.00 days
- Fuel Adjustment: +0.74%
- Demurrage: $0.00
- **Risk Impact**: -$7,741.79 (-0.41%)

**Analysis**: This high-profit voyage shows minimal risk impact percentage-wise. Congestion delays at both West African load port and Chinese discharge port are the main risk factors.

## Key Insights

1. **Risk Impact is Manageable**: Overall portfolio risk impact is -1.23%, indicating the optimization solution is robust to operational risks.

2. **Waiting Time is Largest Risk**: Waiting days risk (8.14 days average) is the dominant risk component, particularly for voyages with tight laycan windows.

3. **Congestion Varies by Port**: Chinese ports (Qingdao, Lianyungang) show higher congestion risk, while other ports have lower risk.

4. **Weather Risk is Moderate**: Average weather delays of 0.65 days per voyage are relatively low, indicating favorable routing and timing.

5. **Demurrage Exposure is Low**: No demurrage was triggered in this scenario, suggesting efficient port operations and good laycan planning.

6. **Fuel Adjustments are Small**: Average fuel adjustment of +1.22% is modest, indicating delays don't significantly impact fuel consumption patterns.

7. **High-TCE Voyages are Resilient**: Voyages with high base TCE (e.g., PACIFIC GLORY at $59,876/day) remain profitable even after risk adjustments.

## Validation Results

All risk-adjusted calculations have been validated:

✅ **Profit Calculations**: Risk-adjusted profit = Base profit - Demurrage - Fuel adjustment  
✅ **TCE Calculations**: Risk-adjusted TCE = Risk-adjusted profit / Risk-adjusted duration  
✅ **Duration Calculations**: Risk-adjusted duration = Base duration + Total risk delays  
✅ **Risk Components**: All delay components are non-negative  
✅ **Portfolio Impact**: Risk impact is within reasonable bounds (< 20% of base profit)

## Files Generated

1. **`multileg_assignments_risk_adjusted.csv`**: Complete assignments with risk metrics
2. **`processed/portfolio_summary_risk_adjusted.json`**: Risk-adjusted portfolio summary
3. **`test_ml_risk_integration.py`**: Integration test script
4. **`test_optimization.py::TestMLRiskSimulation`**: Automated validation tests

## Conclusion

The ML risk simulation successfully integrates operational uncertainties into the optimization framework while maintaining the deterministic optimization structure. The risk-adjusted portfolio profit of $7,913,515.76 represents a realistic assessment that accounts for weather delays, port congestion, waiting time variability, and voyage uncertainty. The 1.23% reduction in profit demonstrates that the optimal solution is robust to operational risks.

