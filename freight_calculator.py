"""
Cargill Freight Calculator & Voyage Optimizer
==============================================
Core engine for the Cargill Ocean Transportation Data-thon 2026.

Calculates TCE (Time Charter Equivalent) for every feasible vessel-cargo
combination, recommends optimal portfolio assignment, and runs scenario
analyses for China port delays and bunker price increases.

TCE = (Net Revenue - Voyage Costs) / Total Voyage Days
Voyage Profit = (TCE - Hire Rate) × Total Voyage Days   [Cargill vessels]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from itertools import product
import json

# Import ML risk simulation and explainability modules
try:
    from ml_risk_simulation import MLRiskSimulator
    from explainability import VoyageExplainability, FeatureImportance, SensitivityResult
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False
    print("Warning: ML risk simulation modules not available. Running in deterministic mode.")

BASE = Path(__file__).parent / "data"
PROCESSED = Path(__file__).parent / "processed"
PROCESSED.mkdir(exist_ok=True)

# ── Port-to-bunkering-location mapping ────────────────────────────────────
# Maps each project port to the nearest bunkering hub from the forward curve
PORT_TO_BUNKER = {
    "Qingdao_China": "Qingdao",
    "Caofeidian_China": "Qingdao",
    "Tianjin_China": "Qingdao",
    "Fangcheng_China": "Qingdao",
    "Xiamen_China": "Shanghai",
    "Jingtang_China": "Qingdao",
    "Lianyungang_China": "Shanghai",
    "Dampier_Australia": "Singapore",
    "Port Hedland_Australia": "Singapore",
    "Ponta da Madeira_Brazil": "Gibraltar",
    "Tubarao_Brazil": "Gibraltar",
    "Itaguaí_Brazil": "Gibraltar",
    "Saldanha Bay_South Africa": "Durban",
    "Kamsar_Guinea": "Gibraltar",
    "Vancouver_Canada": "Singapore",
    "Gwangyang_South Korea": "Qingdao",
    "Teluk Rubiah_Malaysia": "Singapore",
    "Taboneo_Indonesia": "Singapore",
    "Map Ta Phut_Thailand": "Singapore",
    "Rotterdam_Netherlands": "Rotterdam",
    "Port Talbot_Wales": "Rotterdam",
    "Vizag_India": "Singapore",
    "Mangalore_India": "Fujairah",
    "Paradip_India": "Singapore",
    "Kandla_India": "Fujairah",
    "Mundra_India": "Fujairah",
    "Krishnapatnam_India": "Singapore",
    "Jubail_Saudi Arabia": "Fujairah",
}

# Chinese ports for scenario analysis (port delay)
CHINA_PORTS = [
    "Qingdao_China", "Caofeidian_China", "Tianjin_China",
    "Fangcheng_China", "Xiamen_China", "Jingtang_China",
    "Lianyungang_China",
]

# MGO consumption at sea (mt/day) by vessel — from PDF specifications
# These are the "+ X mt MGO" values listed alongside VLSFO consumption
VESSEL_MGO_AT_SEA = {
    "ANN BELL": 2.0, "OCEAN HORIZON": 1.8, "PACIFIC GLORY": 1.9,
    "GOLDEN ASCENT": 2.0, "ATLANTIC FORTUNE": 2.0, "PACIFIC VANGUARD": 1.9,
    "CORAL EMPEROR": 2.0, "EVEREST OCEAN": 1.8, "POLARIS SPIRIT": 1.9,
    "IRON CENTURY": 2.1, "MOUNTAIN TRADER": 2.0, "NAVIS PRIDE": 1.8,
    "AURORA SKY": 2.0, "ZENITH GLORY": 1.9, "TITAN LEGACY": 2.0,
}


class FreightCalculator:
    """
    Evaluates every vessel-cargo combination and recommends optimal
    portfolio assignments to maximize total profit.
    """

    def __init__(self, enable_ml_risks=True):
        """
        Initialize freight calculator.
        
        Parameters:
        -----------
        enable_ml_risks : bool
            Enable ML-based risk simulation (default: True)
        """
        self.enable_ml_risks = enable_ml_risks and ML_ENABLED
        self._load_data()
        
        # Initialize ML risk simulator and explainability engine
        if self.enable_ml_risks:
            self.risk_simulator = MLRiskSimulator()
            self.explainability = VoyageExplainability()
        else:
            self.risk_simulator = None
            self.explainability = None

    # ── Data Loading ──────────────────────────────────────────────────────

    def _load_data(self):
        # Distance matrix
        self.dist_matrix = pd.read_csv(
            PROCESSED / "distance_matrix.csv", index_col=0
        )

        # Vessels
        self.vessels = pd.read_csv(PROCESSED / "vessel_specs.csv")
        self.vessels["ETD"] = pd.to_datetime(self.vessels["ETD"], errors="coerce")

        # Cargoes
        self.cargoes = pd.read_csv(PROCESSED / "cargo_requirements.csv")
        self.cargoes["Laycan_Start"] = pd.to_datetime(
            self.cargoes["Laycan_Start"], errors="coerce"
        )
        self.cargoes["Laycan_End"] = pd.to_datetime(
            self.cargoes["Laycan_End"], errors="coerce"
        )

        # Bunker forward curves (actual data from PDF)
        self.bunker_curves = pd.read_csv(BASE / "bunker_forward_curves.csv")

        # Baltic FFA rates
        self.ffa = pd.read_csv(BASE / "baltic_ffa_rates.csv")

        # Build bunker lookup: (location, fuel_type) -> {month: price}
        self._build_bunker_lookup()

    def _build_bunker_lookup(self):
        """Parse bunker forward curve into a fast lookup dict."""
        self.bunker_prices = {}
        month_cols = [c for c in self.bunker_curves.columns
                      if c not in ("Location", "Fuel_Type")]

        for _, row in self.bunker_curves.iterrows():
            loc = row["Location"]
            fuel = row["Fuel_Type"]
            self.bunker_prices[(loc, fuel)] = {}
            for col in month_cols:
                self.bunker_prices[(loc, fuel)][col] = row[col]

    def get_bunker_price(self, port_name, fuel_type, voyage_date):
        """
        Get bunker price (USD/MT) for a port, fuel type, and approximate date.
        Maps the port to the nearest bunkering hub, then looks up the
        monthly forward curve price.
        """
        bunker_loc = PORT_TO_BUNKER.get(port_name, "Singapore")
        key = (bunker_loc, fuel_type)

        if key not in self.bunker_prices:
            # Fallback to Singapore
            key = ("Singapore", fuel_type)

        prices = self.bunker_prices.get(key, {})

        # Determine month column from date
        if isinstance(voyage_date, str):
            voyage_date = pd.to_datetime(voyage_date)

        month_map = {
            2: "Feb-26", 3: "Mar-26", 4: "Apr-26", 5: "May-26",
            6: "Jun-26", 7: "Jul-26", 8: "Aug-26", 9: "Sep-26",
            10: "Oct-26", 11: "Nov-26", 12: "Dec-26",
        }

        if voyage_date.year == 2026:
            col = month_map.get(voyage_date.month, "Mar-26")
        else:
            col = "Cal-27"

        return prices.get(col, prices.get("Mar-26", 500))

    def get_ffa_rate(self, route_key, month="Mar-26"):
        """Get FFA benchmark rate for a route."""
        row = self.ffa[self.ffa["Route"] == route_key]
        if row.empty:
            return None
        val = row.iloc[0].get(month)
        if pd.isna(val) or val == "":
            return None
        return float(val)

    # ── Voyage Calculation ────────────────────────────────────────────────

    def calculate_voyage(
        self,
        vessel_idx,
        cargo_idx,
        speed_mode="warranted",
        extra_china_delay_days=0,
        bunker_price_multiplier=1.0,
        freight_rate_override=None,
        apply_ml_risks=True,
    ):
        """
        Calculate full voyage economics for one vessel-cargo combination.

        Returns a dict with all revenue/cost components and TCE.
        """
        v = self.vessels.iloc[vessel_idx]
        c = self.cargoes.iloc[cargo_idx]

        vessel_name = v["Vessel Name"]
        fleet = v["Fleet"]
        current_port = str(v["Current Position / Status"]).strip()
        load_port = str(c["Load_Port"]).strip()
        discharge_port = str(c["Discharge_Port"]).strip()

        # ── Speed & consumption parameters ────────────────────────────────
        if speed_mode == "economical":
            speed_laden = v["Economical Speed Laden (kn)"]
            speed_ballast = v["Economical Speed Ballast (kn)"]
            cons_laden = v["Economical Sea Consumption Laden (mt/day)"]
            cons_ballast = v["Economical Sea Consumption Ballast (mt/day)"]
        else:
            speed_laden = v["Warranted Speed Laden (kn)"]
            speed_ballast = v["Warranted Speed Ballast (kn)"]
            cons_laden = v["Warranted Sea Consumption Laden (mt/day)"]
            cons_ballast = v["Warranted Sea Consumption Ballast (mt/day)"]

        cons_port_idle = v["Port Consumption Idle (mt/day)"]
        cons_port_working = v["Port Consumption Working (mt/day)"]
        mgo_sea_per_day = VESSEL_MGO_AT_SEA.get(vessel_name, 2.0)
        hire_rate = v.get("Hire Rate (USD/day)", np.nan)
        dwt = v["DWT (MT)"]
        etd = v["ETD"]

        # ── Distances ─────────────────────────────────────────────────────
        ballast_dist = self._get_distance(current_port, load_port)
        laden_dist = self._get_distance(load_port, discharge_port)

        if ballast_dist is None or laden_dist is None:
            return None  # Cannot calculate — missing distance

        # ── Sailing days ──────────────────────────────────────────────────
        ballast_days = ballast_dist / (speed_ballast * 24) if speed_ballast else 0
        laden_days = laden_dist / (speed_laden * 24) if speed_laden else 0

        # ── Port days ─────────────────────────────────────────────────────
        cargo_qty = c["Quantity_MT"]
        load_rate = c.get("Loading_Rate_MT_per_day", np.nan)
        discharge_rate = c.get("Discharge_Rate_MT_per_day", np.nan)

        if pd.isna(load_rate) or load_rate == 0:
            load_turn = c.get("Load_Turn_Size_MT", 50000)
            load_time = c.get("Load_Turn_Time_hr", 12)
            load_rate = load_turn / (load_time / 24) if load_time > 0 else 100000

        if pd.isna(discharge_rate) or discharge_rate == 0:
            disc_turn = c.get("Discharge_Turn_Size_MT", 30000)
            disc_time = c.get("Discharge_Turn_Time_hr", 24)
            discharge_rate = disc_turn / (disc_time / 24) if disc_time > 0 else 50000

        loading_days = cargo_qty / load_rate
        discharge_days = cargo_qty / discharge_rate

        # Apply China port delay
        if extra_china_delay_days > 0:
            if load_port in CHINA_PORTS:
                loading_days += extra_china_delay_days
            if discharge_port in CHINA_PORTS:
                discharge_days += extra_china_delay_days

        total_port_days = loading_days + discharge_days
        total_voyage_days = ballast_days + laden_days + total_port_days

        # ── Laycan feasibility ────────────────────────────────────────────
        if pd.notna(etd) and pd.notna(c.get("Laycan_End")):
            arrival_at_load = etd + timedelta(days=ballast_days)
            laycan_end = c["Laycan_End"]
            laycan_feasible = arrival_at_load <= laycan_end + timedelta(days=2)
        else:
            arrival_at_load = None
            laycan_feasible = True  # Assume feasible if dates unknown

        # ── Bunker costs ──────────────────────────────────────────────────
        # Determine voyage midpoint date for price lookup
        if pd.notna(etd):
            voyage_mid = etd + timedelta(days=total_voyage_days / 2)
        else:
            voyage_mid = datetime(2026, 3, 15)

        vlsfo_price_load = self.get_bunker_price(
            load_port, "VLSFO", voyage_mid
        ) * bunker_price_multiplier
        vlsfo_price_disch = self.get_bunker_price(
            discharge_port, "VLSFO", voyage_mid
        ) * bunker_price_multiplier
        mgo_price = self.get_bunker_price(
            load_port, "MGO", voyage_mid
        ) * bunker_price_multiplier

        # Average VLSFO price across voyage
        vlsfo_avg = (vlsfo_price_load + vlsfo_price_disch) / 2

        # Bunker consumption
        ballast_vlsfo = ballast_days * cons_ballast
        ballast_mgo = ballast_days * mgo_sea_per_day
        laden_vlsfo = laden_days * cons_laden
        laden_mgo = laden_days * mgo_sea_per_day
        port_vlsfo = loading_days * cons_port_working + discharge_days * cons_port_working

        total_vlsfo = ballast_vlsfo + laden_vlsfo + port_vlsfo
        total_mgo = ballast_mgo + laden_mgo

        bunker_cost = total_vlsfo * vlsfo_avg + total_mgo * mgo_price

        # ── Port costs ────────────────────────────────────────────────────
        source = c.get("Source", "")
        total_port_cost = c.get("Port_Cost_Total_USD", 0)
        if pd.isna(total_port_cost):
            total_port_cost = 0

        # ── Revenue ───────────────────────────────────────────────────────
        commission_pct = c.get("Commission_Percent", 3.75)
        if pd.isna(commission_pct):
            commission_pct = 3.75

        if freight_rate_override is not None:
            freight_rate = freight_rate_override
        else:
            freight_rate = c.get("Freight_Rate_USD_PMT", np.nan)

        if pd.isna(freight_rate):
            # Estimate from FFA for market cargoes
            freight_rate = self._estimate_freight_rate(c)

        gross_revenue = freight_rate * cargo_qty
        commission = gross_revenue * (commission_pct / 100)
        net_revenue = gross_revenue - commission

        # ── Total voyage costs ────────────────────────────────────────────
        total_voyage_costs = bunker_cost + total_port_cost

        # ── TCE ───────────────────────────────────────────────────────────
        if total_voyage_days > 0:
            tce = (net_revenue - total_voyage_costs) / total_voyage_days
        else:
            tce = 0

        # ── ML Risk Simulation & Adjustments ───────────────────────────────
        risk_profile = None
        risk_adjusted_days = total_voyage_days
        risk_adjusted_fuel = total_vlsfo + total_mgo
        risk_adjusted_bunker_cost = bunker_cost
        demurrage_cost = 0.0
        risk_adjusted_profit = total_profit if pd.notna(total_profit) else 0.0
        
        if self.enable_ml_risks and apply_ml_risks:
            # Determine route type for risk simulation
            route_type = self._classify_route(load_port, discharge_port)
            
            # Simulate comprehensive risk
            risk_profile = self.risk_simulator.simulate_comprehensive_risk(
                voyage_date=etd if pd.notna(etd) else datetime(2026, 3, 15),
                load_port=load_port,
                discharge_port=discharge_port,
                base_duration_days=total_voyage_days,
                base_fuel_mt=risk_adjusted_fuel,
                ballast_distance_nm=ballast_dist,
                laden_distance_nm=laden_dist,
                laycan_start=c.get("Laycan_Start") if pd.notna(c.get("Laycan_Start")) else datetime(2026, 3, 1),
                laycan_end=c.get("Laycan_End") if pd.notna(c.get("Laycan_End")) else datetime(2026, 4, 30),
                port_days=total_port_days,
                route_type=route_type
            )
            
            # Apply risk adjustments
            risk_adjusted_days = risk_profile['adjusted_duration_days']
            
            # Adjust fuel consumption
            fuel_adj = risk_profile['fuel_adjustment']
            risk_adjusted_fuel = fuel_adj['adjusted_fuel_mt']
            risk_adjusted_bunker_cost = risk_adjusted_fuel * vlsfo_avg * 0.9  # Approximate
            
            # Demurrage cost
            demurrage_cost = risk_profile['demurrage_risk']['demurrage_cost_usd']
            
            # Recalculate profit with risk adjustments
            if pd.notna(hire_rate) and hire_rate > 0:
                # Additional costs: demurrage + extra fuel
                additional_costs = demurrage_cost + (risk_adjusted_bunker_cost - bunker_cost)
                risk_adjusted_revenue = net_revenue
                risk_adjusted_costs = total_voyage_costs + additional_costs
                risk_adjusted_tce = (risk_adjusted_revenue - risk_adjusted_costs) / risk_adjusted_days if risk_adjusted_days > 0 else 0
                risk_adjusted_daily_profit = risk_adjusted_tce - hire_rate
                risk_adjusted_profit = risk_adjusted_daily_profit * risk_adjusted_days
            else:
                risk_adjusted_profit = np.nan
        
        # ── Profit (for Cargill vessels) ──────────────────────────────────
        if pd.notna(hire_rate) and hire_rate > 0:
            daily_profit = tce - hire_rate
            total_profit = daily_profit * total_voyage_days
        else:
            daily_profit = np.nan
            total_profit = np.nan

        result = {
            "Vessel": vessel_name,
            "Fleet": fleet,
            "DWT": dwt,
            "Hire_Rate": hire_rate,
            "Current_Port": current_port,
            "ETD": str(etd.date()) if pd.notna(etd) else "",
            "Cargo_Index": cargo_idx,
            "Route": c.get("Route", ""),
            "Source": source,
            "Customer": c.get("Customer", ""),
            "Commodity": c.get("Commodity", ""),
            "Load_Port": load_port,
            "Discharge_Port": discharge_port,
            "Cargo_Quantity_MT": cargo_qty,
            "Freight_Rate_USD_PMT": round(freight_rate, 3),
            "Commission_Pct": commission_pct,
            "Speed_Mode": speed_mode,
            # Distances
            "Ballast_Distance_NM": round(ballast_dist, 1),
            "Laden_Distance_NM": round(laden_dist, 1),
            # Time
            "Ballast_Days": round(ballast_days, 2),
            "Laden_Days": round(laden_days, 2),
            "Loading_Days": round(loading_days, 2),
            "Discharge_Days": round(discharge_days, 2),
            "Extra_China_Delay": extra_china_delay_days,
            "Total_Voyage_Days": round(total_voyage_days, 2),
            # Bunker
            "VLSFO_Price_Avg": round(vlsfo_avg, 1),
            "MGO_Price": round(mgo_price, 1),
            "Total_VLSFO_MT": round(total_vlsfo, 1),
            "Total_MGO_MT": round(total_mgo, 1),
            "Bunker_Cost_USD": round(bunker_cost, 0),
            # Costs
            "Port_Cost_USD": round(total_port_cost, 0),
            "Total_Voyage_Cost_USD": round(total_voyage_costs, 0),
            # Revenue
            "Gross_Revenue_USD": round(gross_revenue, 0),
            "Commission_USD": round(commission, 0),
            "Net_Revenue_USD": round(net_revenue, 0),
            # Results
            "TCE_USD_per_day": round(tce, 0),
            "Daily_Profit_USD": round(daily_profit, 0) if pd.notna(daily_profit) else None,
            "Total_Voyage_Profit_USD": round(total_profit, 0) if pd.notna(total_profit) else None,
            "Laycan_Feasible": laycan_feasible,
            "Arrival_At_Load": str(arrival_at_load.date()) if arrival_at_load and pd.notna(arrival_at_load) else "",
        }
        
        # Add ML risk-adjusted metrics if available
        if risk_profile is not None:
            result.update({
                "Risk_Adjusted_Duration_Days": round(risk_adjusted_days, 2),
                "Risk_Adjusted_Fuel_MT": round(risk_adjusted_fuel, 1),
                "Risk_Adjusted_Bunker_Cost_USD": round(risk_adjusted_bunker_cost, 0),
                "Demurrage_Cost_USD": round(demurrage_cost, 0),
                "Risk_Adjusted_Profit_USD": round(risk_adjusted_profit, 0) if pd.notna(risk_adjusted_profit) else None,
                "Total_Risk_Delay_Days": round(risk_profile['total_delay_days'], 2),
                "Weather_Delay_Days": round(risk_profile['weather_risk']['delay_days'], 2),
                "Congestion_Delay_Days": round(
                    risk_profile['load_congestion']['congestion_delay_days'] +
                    risk_profile['discharge_congestion']['congestion_delay_days'], 2
                ),
                "Waiting_Days_Risk": round(risk_profile['waiting_risk']['waiting_days'], 2),
                "Laycan_Breach_Prob": round(risk_profile['waiting_risk']['laycan_breach_prob'], 3),
            })
        
        return result
    
    def _classify_route(self, load_port: str, discharge_port: str) -> str:
        """
        Classify route type for risk simulation.
        
        Parameters:
        -----------
        load_port : str
            Load port name
        discharge_port : str
            Discharge port name
        
        Returns:
        --------
        str : Route classification
        """
        # Simple classification based on port regions
        asia_ports = ['QINGDAO', 'SHANGHAI', 'GWANGYANG', 'XIAMEN', 'FANGCHENG', 'TIANJIN', 'CAOFEIDIAN', 'JINGTANG', 'LIANYUNGANG']
        europe_ports = ['ROTTERDAM', 'PORT TALBOT']
        americas_ports = ['VANCOUVER', 'ITAGUAI', 'TUBARAO', 'PONTA DA MADEIRA']
        africa_ports = ['KAMSAR', 'SALDANHA']
        australia_ports = ['DAMPIER', 'PORT HEDLAND']
        
        load_asia = any(p in load_port.upper() for p in asia_ports)
        load_europe = any(p in load_port.upper() for p in europe_ports)
        load_americas = any(p in load_port.upper() for p in americas_ports)
        
        disch_asia = any(p in discharge_port.upper() for p in asia_ports)
        disch_europe = any(p in discharge_port.upper() for p in europe_ports)
        disch_americas = any(p in discharge_port.upper() for p in americas_ports)
        
        if (load_europe or load_americas) and disch_asia:
            return 'transpacific'
        elif load_europe and disch_americas:
            return 'transatlantic'
        elif (load_asia or load_europe) and (disch_asia or disch_europe):
            return 'asia_europe'
        elif load_asia and any(p in discharge_port.upper() for p in africa_ports):
            return 'asia_africa'
        elif any(p in load_port.upper() for p in australia_ports) and disch_asia:
            return 'asia_australia'
        else:
            return 'default'

    def _get_distance(self, port_a, port_b):
        """Get distance between two ports from the distance matrix."""
        if port_a == port_b:
            return 0.0
        try:
            d = self.dist_matrix.loc[port_a, port_b]
            if pd.notna(d):
                return d
        except KeyError:
            pass
        return None

    def _estimate_freight_rate(self, cargo):
        """
        Estimate freight rate for market cargoes using FFA benchmarks.

        Calibrated from two FFA anchor points:
          C5 (Port Hedland → Qingdao): ~3,493 NM at $8.717/MT (Mar-26)
          C3 (Tubarao → Qingdao):      ~10,480 NM at $20.908/MT (Mar-26)

        Linear model: freight_rate = 2.622 + 0.001745 × laden_distance_NM
        """
        route = str(cargo.get("Route", "")).lower()
        load_port = str(cargo.get("Load_Port", ""))
        discharge_port = str(cargo.get("Discharge_Port", ""))

        # Determine voyage timing for FFA lookup
        laycan = cargo.get("Laycan_Start")
        if pd.notna(laycan):
            month = laycan.month
            if month == 3:
                ffa_col = "Mar-26"
            elif month == 4:
                ffa_col = "Apr-26"
            else:
                ffa_col = "Q2-26"
        else:
            ffa_col = "Mar-26"

        # Direct FFA route matching
        if "brazil" in route and "china" in route:
            rate = self.get_ffa_rate("C3_Tubarao_Qingdao", ffa_col)
            if rate:
                return rate

        if "australia" in route and "china" in route and "korea" not in route:
            rate = self.get_ffa_rate("C5_WestAustralia_Qingdao", ffa_col)
            if rate:
                return rate

        # Distance-calibrated estimation for all other routes
        # Anchor points (Mar-26 FFA):
        #   C5: dist=3493 NM, rate=$8.717/MT
        #   C3: dist=10480 NM, rate=$20.908/MT
        # Linear fit: rate = a + b × dist
        #   b = (20.908 - 8.717) / (10480 - 3493) = 0.001745
        #   a = 8.717 - 0.001745 × 3493 = 2.622
        laden_dist = self._get_distance(load_port, discharge_port)
        if laden_dist and laden_dist > 0:
            # Apply FFA time adjustment: scale the base rate by the
            # ratio of the relevant period's FFA to the Mar-26 baseline
            c5_mar = self.get_ffa_rate("C5_WestAustralia_Qingdao", "Mar-26") or 8.717
            c5_period = self.get_ffa_rate("C5_WestAustralia_Qingdao", ffa_col) or c5_mar
            time_adj = c5_period / c5_mar if c5_mar > 0 else 1.0

            base_rate = 2.622 + 0.001745 * laden_dist
            estimated = base_rate * time_adj
            return round(max(estimated, 4.0), 3)

        return 10.0

    # ── All Combinations ──────────────────────────────────────────────────

    def calculate_all_combinations(
        self,
        speed_mode="warranted",
        extra_china_delay_days=0,
        bunker_price_multiplier=1.0,
    ):
        """
        Calculate TCE for every vessel-cargo combination.
        Returns a DataFrame with all results.
        """
        results = []
        for v_idx in range(len(self.vessels)):
            for c_idx in range(len(self.cargoes)):
                voyage = self.calculate_voyage(
                    v_idx, c_idx,
                    speed_mode=speed_mode,
                    extra_china_delay_days=extra_china_delay_days,
                    bunker_price_multiplier=bunker_price_multiplier,
                )
                if voyage is not None:
                    results.append(voyage)

        df = pd.DataFrame(results)
        return df

    # ── Portfolio Optimization ────────────────────────────────────────────

    def optimize_portfolio(
        self,
        speed_mode="warranted",
        extra_china_delay_days=0,
        bunker_price_multiplier=1.0,
    ):
        """
        Find the optimal vessel-cargo assignment that maximizes total
        portfolio profit.

        Constraints:
          - Each Cargill committed cargo MUST be carried (by Cargill or market vessel)
          - Each vessel can carry at most one cargo at a time
          - Laycan feasibility must be met
          - Cargill vessels should be prioritized for committed cargoes

        Returns: (assignments_df, all_combinations_df)
        """
        all_combos = self.calculate_all_combinations(
            speed_mode=speed_mode,
            extra_china_delay_days=extra_china_delay_days,
            bunker_price_multiplier=bunker_price_multiplier,
        )

        if all_combos.empty:
            return pd.DataFrame(), all_combos

        # Filter to feasible voyages
        feasible = all_combos[all_combos["Laycan_Feasible"]].copy()

        # Split into Cargill vessels and market vessels
        cargill_vessels = feasible[feasible["Fleet"] == "Cargill"]
        market_vessels = feasible[feasible["Fleet"] == "Market"]

        # Committed cargoes (must be assigned)
        committed_indices = set(
            self.cargoes[self.cargoes["Source"] == "Cargill_Committed"].index.tolist()
        )
        market_cargo_indices = set(
            self.cargoes[self.cargoes["Source"] == "Market"].index.tolist()
        )

        # Greedy optimization: assign vessels to cargoes by highest TCE
        assignments = []
        assigned_vessels = set()
        assigned_cargoes = set()

        # Phase 1: Assign Cargill vessels to committed cargoes first (by highest TCE)
        committed_combos = cargill_vessels[
            cargill_vessels["Cargo_Index"].isin(committed_indices)
        ].sort_values("TCE_USD_per_day", ascending=False)

        for _, row in committed_combos.iterrows():
            vessel = row["Vessel"]
            cargo_idx = row["Cargo_Index"]
            if vessel not in assigned_vessels and cargo_idx not in assigned_cargoes:
                assignments.append(row.to_dict())
                assigned_vessels.add(vessel)
                assigned_cargoes.add(cargo_idx)

        # Phase 2: Any unassigned committed cargoes → best market vessel
        unassigned_committed = committed_indices - assigned_cargoes
        if unassigned_committed:
            for c_idx in unassigned_committed:
                market_options = market_vessels[
                    (market_vessels["Cargo_Index"] == c_idx)
                    & (~market_vessels["Vessel"].isin(assigned_vessels))
                ].sort_values("TCE_USD_per_day", ascending=False)

                if not market_options.empty:
                    best = market_options.iloc[0]
                    assignments.append(best.to_dict())
                    assigned_vessels.add(best["Vessel"])
                    assigned_cargoes.add(c_idx)

        # Phase 3: Assign remaining Cargill vessels to best market cargoes
        remaining_cargill = [
            v for v in self.vessels[self.vessels["Fleet"] == "Cargill"]["Vessel Name"]
            if v not in assigned_vessels
        ]
        remaining_market_cargoes = market_cargo_indices - assigned_cargoes

        for vessel in remaining_cargill:
            options = feasible[
                (feasible["Vessel"] == vessel)
                & (feasible["Cargo_Index"].isin(remaining_market_cargoes))
            ].sort_values("TCE_USD_per_day", ascending=False)

            if not options.empty:
                best = options.iloc[0]
                if best["TCE_USD_per_day"] > best.get("Hire_Rate", 0):
                    assignments.append(best.to_dict())
                    assigned_vessels.add(vessel)
                    assigned_cargoes.add(best["Cargo_Index"])
                    remaining_market_cargoes.discard(best["Cargo_Index"])

        assignments_df = pd.DataFrame(assignments)
        return assignments_df, all_combos

    # ── Scenario Analysis ─────────────────────────────────────────────────

    def scenario_china_delay(self, base_assignments, max_delay=60):
        """
        Scenario 1: Find the number of additional China port delay days
        that makes the current recommendation no longer optimal.

        Tests every delay value from 0 to max_delay.
        """
        results = []
        base_vessels = {
            row["Cargo_Index"]: row["Vessel"]
            for _, row in base_assignments.iterrows()
        }

        first_change_delay = None

        for delay in range(0, max_delay + 1):
            new_assignments, _ = self.optimize_portfolio(
                extra_china_delay_days=delay
            )

            if new_assignments.empty:
                continue

            new_vessels = {
                row["Cargo_Index"]: row["Vessel"]
                for _, row in new_assignments.iterrows()
            }

            total_profit = new_assignments["Total_Voyage_Profit_USD"].sum()
            avg_tce = new_assignments["TCE_USD_per_day"].mean()

            changed = any(
                base_vessels.get(k) != new_vessels.get(k)
                for k in base_vessels
            )

            results.append({
                "China_Delay_Days": delay,
                "Total_Portfolio_Profit": round(total_profit, 0),
                "Avg_TCE": round(avg_tce, 0),
                "Assignment_Changed": changed,
                "Assignments": json.dumps(
                    {str(k): v for k, v in new_vessels.items()}
                ),
            })

            if changed and delay > 0 and first_change_delay is None:
                first_change_delay = delay

            # Continue a few more after finding threshold for context
            if first_change_delay and delay >= first_change_delay + 5:
                break

        return pd.DataFrame(results)

    def scenario_bunker_increase(self, base_assignments, max_pct=300, step=5):
        """
        Scenario 2: Find the VLSFO price increase (%) at which the current
        recommendation becomes less profitable than the next best option.

        Tests from 0% to max_pct in step increments.
        """
        results = []
        base_vessels = {
            row["Cargo_Index"]: row["Vessel"]
            for _, row in base_assignments.iterrows()
        }

        first_change_pct = None

        for pct in range(0, max_pct + 1, step):
            multiplier = 1.0 + pct / 100.0
            new_assignments, _ = self.optimize_portfolio(
                bunker_price_multiplier=multiplier
            )

            if new_assignments.empty:
                continue

            new_vessels = {
                row["Cargo_Index"]: row["Vessel"]
                for _, row in new_assignments.iterrows()
            }

            total_profit = new_assignments["Total_Voyage_Profit_USD"].sum()
            avg_tce = new_assignments["TCE_USD_per_day"].mean()

            changed = any(
                base_vessels.get(k) != new_vessels.get(k)
                for k in base_vessels
            )

            results.append({
                "Bunker_Increase_Pct": pct,
                "Bunker_Multiplier": round(multiplier, 2),
                "Total_Portfolio_Profit": round(total_profit, 0),
                "Avg_TCE": round(avg_tce, 0),
                "Assignment_Changed": changed,
                "Assignments": json.dumps(
                    {str(k): v for k, v in new_vessels.items()}
                ),
            })

            if changed and pct > 0 and first_change_pct is None:
                first_change_pct = pct

            if first_change_pct and pct >= first_change_pct + 20:
                break

        return pd.DataFrame(results)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Run full analysis
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("CARGILL FREIGHT CALCULATOR & VOYAGE OPTIMIZER")
    print("=" * 70)
    print()

    calc = FreightCalculator()

    # ── 1. Calculate all vessel-cargo TCE combinations ─────────────────
    print("[1] Calculating all vessel-cargo TCE combinations...")
    all_combos = calc.calculate_all_combinations()
    all_combos.to_csv(PROCESSED / "all_voyage_combinations.csv", index=False)
    print(f"    {len(all_combos)} combinations calculated")
    print()

    # ── 2. Find optimal portfolio assignment ───────────────────────────
    print("[2] Optimizing portfolio assignments...")
    assignments, _ = calc.optimize_portfolio()
    assignments.to_csv(PROCESSED / "optimal_assignments.csv", index=False)

    print(f"\n    OPTIMAL VESSEL-CARGO ASSIGNMENTS:")
    print("    " + "=" * 66)
    for _, a in assignments.iterrows():
        profit_str = (
            f"${a['Total_Voyage_Profit_USD']:,.0f}"
            if pd.notna(a.get("Total_Voyage_Profit_USD"))
            else "N/A"
        )
        print(
            f"    {a['Vessel']:20s} → {a['Route']:40s}"
        )
        print(
            f"      TCE: ${a['TCE_USD_per_day']:,.0f}/day | "
            f"Profit: {profit_str} | "
            f"Days: {a['Total_Voyage_Days']:.1f}"
        )
    print()

    total_profit = assignments["Total_Voyage_Profit_USD"].sum()
    avg_tce = assignments["TCE_USD_per_day"].mean()
    print(f"    Total Portfolio Profit: ${total_profit:,.0f}")
    print(f"    Average TCE: ${avg_tce:,.0f}/day")
    print()

    # ── 3. Scenario Analysis: China Port Delay ─────────────────────────
    print("[3] Scenario: China port delay analysis...")
    delay_results = calc.scenario_china_delay(assignments)
    delay_results.to_csv(PROCESSED / "scenario_china_delay.csv", index=False)

    delay_threshold = delay_results[
        (delay_results["Assignment_Changed"]) & (delay_results["China_Delay_Days"] > 0)
    ]
    if not delay_threshold.empty:
        days = delay_threshold.iloc[0]["China_Delay_Days"]
        print(f"    Assignment change threshold: {days} days of extra delay")
    else:
        max_tested = delay_results["China_Delay_Days"].max()
        print(f"    Vessel assignment stable through {max_tested} days of extra delay")

    # Find when portfolio profit turns negative
    negative_rows = delay_results[delay_results["Total_Portfolio_Profit"] < 0]
    if not negative_rows.empty:
        break_even_days = negative_rows.iloc[0]["China_Delay_Days"]
        print(f"    Portfolio break-even threshold: {break_even_days} days")
        print(f"    (Profit goes negative at {break_even_days}+ days of China delay)")
    print()

    # ── 4. Scenario Analysis: Bunker Price Increase ────────────────────
    print("[4] Scenario: Bunker price increase analysis...")
    bunker_results = calc.scenario_bunker_increase(assignments)
    bunker_results.to_csv(PROCESSED / "scenario_bunker_increase.csv", index=False)

    bunker_threshold = bunker_results[
        (bunker_results["Assignment_Changed"]) & (bunker_results["Bunker_Increase_Pct"] > 0)
    ]
    if not bunker_threshold.empty:
        pct = bunker_threshold.iloc[0]["Bunker_Increase_Pct"]
        new_assign = bunker_threshold.iloc[0]["Assignments"]
        print(f"    Assignment change threshold: +{pct}% bunker price increase")
        print(f"    New assignments: {new_assign}")
    else:
        max_tested = bunker_results["Bunker_Increase_Pct"].max()
        print(f"    Recommendation stable through +{max_tested}% bunker price increase")

    # Find when portfolio profit turns negative under bunker increase
    neg_bunker = bunker_results[bunker_results["Total_Portfolio_Profit"] < 0]
    if not neg_bunker.empty:
        break_pct = neg_bunker.iloc[0]["Bunker_Increase_Pct"]
        print(f"    Portfolio break-even threshold: +{break_pct}% bunker increase")
    print()

    # ── 5. Summary for chatbot ─────────────────────────────────────────
    print("[5] Generating summary for chatbot integration...")
    summary = {
        "total_vessels": len(calc.vessels),
        "cargill_vessels": len(calc.vessels[calc.vessels["Fleet"] == "Cargill"]),
        "market_vessels": len(calc.vessels[calc.vessels["Fleet"] == "Market"]),
        "total_cargoes": len(calc.cargoes),
        "committed_cargoes": len(calc.cargoes[calc.cargoes["Source"] == "Cargill_Committed"]),
        "market_cargoes": len(calc.cargoes[calc.cargoes["Source"] == "Market"]),
        "total_combinations": len(all_combos),
        "assignments": [],
        "total_portfolio_profit": round(total_profit, 0),
        "average_tce": round(avg_tce, 0),
    }

    for _, a in assignments.iterrows():
        summary["assignments"].append({
            "vessel": a["Vessel"],
            "fleet": a["Fleet"],
            "route": a["Route"],
            "commodity": a["Commodity"],
            "load_port": a["Load_Port"],
            "discharge_port": a["Discharge_Port"],
            "tce": round(a["TCE_USD_per_day"], 0),
            "total_profit": round(a["Total_Voyage_Profit_USD"], 0) if pd.notna(a.get("Total_Voyage_Profit_USD")) else None,
            "voyage_days": round(a["Total_Voyage_Days"], 1),
            "freight_rate": a["Freight_Rate_USD_PMT"],
        })

    with open(PROCESSED / "portfolio_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"    Summary saved to {PROCESSED / 'portfolio_summary.json'}")

    print()
    print("=" * 70)
    print("FREIGHT CALCULATOR COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
