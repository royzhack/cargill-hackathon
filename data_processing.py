"""
Cargill Capesize Shipping Data Processing Pipeline
===================================================
Produces:
  1. Master distance matrix between all load/discharge ports
  2. Unified vessel specifications table
  3. Unified cargo requirements table (with laycan, load/discharge rates)
  4. Bunker price lookup stub (pending forward curve data)
  5. Ballast & laden sailing time matrices per vessel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from math import radians, sin, cos, asin, sqrt

BASE = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "processed"
OUTPUT.mkdir(exist_ok=True)


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 3440.065 * asin(sqrt(a))  # Earth radius in nm


# ── Port name mapping ─────────────────────────────────────────────────────
# Maps port_locations.csv names → Port Distances.csv ALL-CAPS names
PORT_NAME_MAP = {
    "Dampier_Australia": "DAMPIER",
    "Ponta da Madeira_Brazil": "PONTA DA MADEIRA",
    "Saldanha Bay_South Africa": "SALDANHA BAY",
    "Taboneo_Indonesia": "TABONEO",
    "Vancouver_Canada": "VANCOUVER (CANADA)",
    "Kamsar_Guinea": "PORT KAMSAR",
    "Port Hedland_Australia": "PORT HEDLAND",
    "Tubarao_Brazil": "TUBARAO",
    "Qingdao_China": "QINGDAO",
    "Caofeidian_China": "CAOFEIDIAN",
    "Tianjin_China": "TIANJIN",
    "Vizag_India": "VIZAG",
    "Fangcheng_China": "FANGCHENG",
    "Mangalore_India": "MANGALORE",
    "Gwangyang_South Korea": "KWANGYANG",
    "Teluk Rubiah_Malaysia": "TELUK RUBIAH",
    "Paradip_India": "PARADIP",
    "Map Ta Phut_Thailand": "MAP TA PHUT",
    "Rotterdam_Netherlands": "ROTTERDAM",
    "Xiamen_China": "XIAMEN",
    "Kandla_India": "KANDLA",
    "Port Talbot_Wales": "PORT TALBOT",
    "Mundra_India": "MUNDRA",
    "Jubail_Saudi Arabia": "JUBAIL",
    "Itaguaí_Brazil": "ITAGUAI",
    "Jingtang_China": "JINGTANG",
    "Lianyungang_China": "LIANYUNGANG",
    "Krishnapatnam_India": "KRISHNAPATNAM",
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. MASTER DISTANCE MATRIX
# ═══════════════════════════════════════════════════════════════════════════

def build_distance_matrix():
    """
    Build a symmetric NxN distance matrix for all project ports.

    Uses Port Distances.csv for known routes. For missing pairs, estimates
    sailing distance as haversine × 1.40 (typical ocean routing factor
    accounting for land masses, straits, and shipping lanes).
    """
    ROUTING_FACTOR = 1.40

    port_locs = pd.read_csv(BASE / "port_locations.csv")
    port_names = [str(n).strip().strip('"') for n in port_locs["port_name"]]

    # Build lat/lon lookup from port_locations.csv
    port_coords = {}
    for _, row in port_locs.iterrows():
        name = str(row["port_name"]).strip().strip('"')
        port_coords[name] = (row["latitude"], row["longitude"])

    dist_raw = pd.read_csv(BASE / "Port Distances.csv")

    # Build lookup: (FROM_UPPER, TO_UPPER) -> distance
    dist_lookup = {}
    for _, row in dist_raw.iterrows():
        key = (str(row["PORT_NAME_FROM"]).strip(), str(row["PORT_NAME_TO"]).strip())
        dist_lookup[key] = row["DISTANCE"]

    # Map our port names to distance-file names
    dist_names = {p: PORT_NAME_MAP.get(p, p.split("_")[0].upper()) for p in port_names}

    n = len(port_names)
    matrix = pd.DataFrame(np.nan, index=port_names, columns=port_names)
    source = pd.DataFrame("", index=port_names, columns=port_names)

    exact_count = 0
    estimated_count = 0

    for i, p1 in enumerate(port_names):
        matrix.iloc[i, i] = 0.0
        source.iloc[i, i] = "self"
        for j, p2 in enumerate(port_names):
            if i >= j:
                continue
            d1, d2 = dist_names[p1], dist_names[p2]
            dist = dist_lookup.get((d1, d2)) or dist_lookup.get((d2, d1))
            if dist is not None:
                matrix.iloc[i, j] = dist
                matrix.iloc[j, i] = dist
                source.iloc[i, j] = "exact"
                source.iloc[j, i] = "exact"
                exact_count += 1
            else:
                # Fallback: haversine × routing factor
                c1, c2 = port_coords.get(p1), port_coords.get(p2)
                if c1 and c2:
                    gc = haversine_nm(c1[0], c1[1], c2[0], c2[1])
                    est = round(gc * ROUTING_FACTOR, 2)
                    matrix.iloc[i, j] = est
                    matrix.iloc[j, i] = est
                    source.iloc[i, j] = "estimated"
                    source.iloc[j, i] = "estimated"
                    estimated_count += 1

    out_path = OUTPUT / "distance_matrix.csv"
    matrix.to_csv(out_path)

    source_path = OUTPUT / "distance_matrix_source.csv"
    source.to_csv(source_path)

    total_pairs = n * (n - 1) // 2
    print(f"[1] Distance matrix ({n}x{n}) saved to {out_path}")
    print(f"    {exact_count} exact (from Port Distances.csv)")
    print(f"    {estimated_count} estimated (haversine x {ROUTING_FACTOR})")
    print(f"    {total_pairs - exact_count - estimated_count} still missing")
    print(f"    Source flags saved to {source_path}")

    return matrix


# ═══════════════════════════════════════════════════════════════════════════
# 2. VESSEL SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_vessel_specs():
    """Combine Cargill + Market vessels into one structured table."""
    cargill = pd.read_csv(BASE / "Cargill_Capesize_Vessels.csv")
    market = pd.read_csv(BASE / "Market_Vessels_Formatted.csv")

    cargill["Fleet"] = "Cargill"
    market["Fleet"] = "Market"

    # Standardize columns — Cargill has Hire Rate, Market does not
    if "Hire Rate (USD/day)" not in market.columns:
        market["Hire Rate (USD/day)"] = np.nan

    # Common columns to keep
    cols = [
        "Fleet",
        "Vessel Name",
        "DWT (MT)",
        "Hire Rate (USD/day)",
        "Warranted Speed Laden (kn)",
        "Warranted Speed Ballast (kn)",
        "Warranted Sea Consumption Laden (mt/day)",
        "Warranted Sea Consumption Ballast (mt/day)",
        "Economical Speed Laden (kn)",
        "Economical Speed Ballast (kn)",
        "Economical Sea Consumption Laden (mt/day)",
        "Economical Sea Consumption Ballast (mt/day)",
        "Port Consumption Idle (mt/day)",
        "Port Consumption Working (mt/day)",
        "Current Position / Status",
        "Current Position / Status_Latitude",
        "Current Position / Status_Longitude",
        "ETD",
        "Bunker Remaining VLSFO (mt)",
        "Bunker Remaining MGO (mt)",
    ]

    # Only keep columns that exist in both
    cols_c = [c for c in cols if c in cargill.columns]
    cols_m = [c for c in cols if c in market.columns]

    vessels = pd.concat([cargill[cols_c], market[cols_m]], ignore_index=True)
    vessels["ETD"] = pd.to_datetime(vessels["ETD"], errors="coerce")

    out_path = OUTPUT / "vessel_specs.csv"
    vessels.to_csv(out_path, index=False)
    print(f"[2] Vessel specs ({len(vessels)} vessels) saved to {out_path}")
    return vessels


# ═══════════════════════════════════════════════════════════════════════════
# 3. CARGO REQUIREMENTS
# ═══════════════════════════════════════════════════════════════════════════

def build_cargo_requirements():
    """Combine Cargill committed + Market cargoes into one structured table."""
    cargill = pd.read_csv(BASE / "Cargill_Committed_Cargoes_Structured.csv")
    market = pd.read_csv(BASE / "Market_Cargoes_Structured.csv")

    cargill["Source"] = "Cargill_Committed"
    market["Source"] = "Market"

    # Normalize column names between the two files
    # Cargill uses Load_Port_Primary / Discharge_Port_Primary
    # Market uses Load_Port / Discharge_Port
    rename_cargill = {
        "Load_Port_Primary": "Load_Port",
        "Load_Port_Primary_Latitude": "Load_Port_Latitude",
        "Load_Port_Primary_Longitude": "Load_Port_Longitude",
        "Discharge_Port_Primary": "Discharge_Port",
        "Discharge_Port_Primary_Latitude": "Discharge_Port_Latitude",
        "Discharge_Port_Primary_Longitude": "Discharge_Port_Longitude",
        "Load_Parcel_Size_MT": "Load_Turn_Size_MT",
        "Load_Turn_Time_Hours": "Load_Turn_Time_hr",
        "Discharge_Parcel_Size_MT": "Discharge_Turn_Size_MT",
        "Discharge_Turn_Time_Hours": "Discharge_Turn_Time_hr",
        "Freight_Rate_USD_PMT": "Freight_Rate_USD_PMT",
    }
    cargill = cargill.rename(columns=rename_cargill)

    # Standardize common columns
    common_cols = [
        "Source",
        "Route",
        "Customer",
        "Commodity",
        "Quantity_MT",
        "Quantity_Tolerance",
        "Laycan_Start",
        "Laycan_End",
        "Load_Port",
        "Load_Port_Latitude",
        "Load_Port_Longitude",
        "Load_Turn_Size_MT",
        "Load_Turn_Time_hr",
        "Discharge_Port",
        "Discharge_Port_Latitude",
        "Discharge_Port_Longitude",
        "Discharge_Turn_Size_MT",
        "Discharge_Turn_Time_hr",
        "Commission_Percent",
    ]

    # Add freight rate and port cost columns
    if "Freight_Rate_USD_PMT" in cargill.columns:
        common_cols.append("Freight_Rate_USD_PMT")

    # Normalize port costs: compute Port_Cost_Total_USD for both sources
    # Cargill has Port_Cost_USD (single total)
    # Market has Port_Cost_Load_USD + Port_Cost_Discharge_USD
    if "Port_Cost_USD" in cargill.columns:
        cargill["Port_Cost_Total_USD"] = cargill["Port_Cost_USD"].fillna(0)
    else:
        cargill["Port_Cost_Total_USD"] = 0

    load_cost = market.get("Port_Cost_Load_USD", pd.Series(dtype=float)).fillna(0)
    disch_cost = market.get("Port_Cost_Discharge_USD", pd.Series(dtype=float)).fillna(0)
    market["Port_Cost_Total_USD"] = load_cost + disch_cost

    common_cols.append("Port_Cost_Total_USD")

    # Keep only existing columns
    cols_c = [c for c in common_cols if c in cargill.columns]
    cols_m = [c for c in common_cols if c in market.columns]

    cargoes = pd.concat([cargill[cols_c], market[cols_m]], ignore_index=True)

    # Parse dates
    for col in ["Laycan_Start", "Laycan_End"]:
        if col in cargoes.columns:
            cargoes[col] = pd.to_datetime(cargoes[col], errors="coerce")

    # Derived: laycan window in days
    if "Laycan_Start" in cargoes.columns and "Laycan_End" in cargoes.columns:
        cargoes["Laycan_Window_Days"] = (
            cargoes["Laycan_End"] - cargoes["Laycan_Start"]
        ).dt.days

    # Derived: loading rate (MT/day) and discharge rate (MT/day)
    cargoes["Loading_Rate_MT_per_day"] = (
        cargoes["Load_Turn_Size_MT"] / (cargoes["Load_Turn_Time_hr"] / 24)
    ).round(0)
    cargoes["Discharge_Rate_MT_per_day"] = (
        cargoes["Discharge_Turn_Size_MT"] / (cargoes["Discharge_Turn_Time_hr"] / 24)
    ).round(0)

    # Derived: estimated loading days and discharge days for full cargo
    cargoes["Est_Loading_Days"] = (
        cargoes["Quantity_MT"] / cargoes["Loading_Rate_MT_per_day"]
    ).round(2)
    cargoes["Est_Discharge_Days"] = (
        cargoes["Quantity_MT"] / cargoes["Discharge_Rate_MT_per_day"]
    ).round(2)

    out_path = OUTPUT / "cargo_requirements.csv"
    cargoes.to_csv(out_path, index=False)
    print(f"[3] Cargo requirements ({len(cargoes)} cargoes) saved to {out_path}")
    return cargoes


# ═══════════════════════════════════════════════════════════════════════════
# 4. BUNKER PRICE LOOKUP (Stub — awaiting forward curve data)
# ═══════════════════════════════════════════════════════════════════════════

def build_bunker_lookup():
    """
    Create a bunker price lookup structure.

    The user mentioned "Bunker forward curves (18 locations x 2 fuel types)"
    but this data has not been provided yet. This function creates the
    expected schema and populates it with industry-average placeholder
    values so downstream calculations can proceed.

    When actual data arrives, replace bunker_forward_curves.csv.
    """
    # 18 major bunkering ports (representative set)
    bunkering_ports = [
        "Singapore", "Fujairah", "Rotterdam", "Houston", "Busan",
        "Hong Kong", "Shanghai", "Zhoushan", "Colombo", "Durban",
        "Gibraltar", "Panama", "Piraeus", "Jeddah", "Mumbai",
        "Kaohsiung", "Algeciras", "Port Louis",
    ]

    fuel_types = ["VLSFO", "MGO"]

    # Generate monthly dates for 6-month forward curve
    base_date = datetime(2026, 3, 1)
    dates = [base_date + timedelta(days=30 * i) for i in range(6)]

    rows = []
    for port in bunkering_ports:
        for fuel in fuel_types:
            for dt in dates:
                # Placeholder prices (USD/mt) — replace with actual forward curves
                if fuel == "VLSFO":
                    price = 580.0  # Approximate VLSFO spot
                else:
                    price = 780.0  # Approximate MGO spot
                rows.append({
                    "Port": port,
                    "Fuel_Type": fuel,
                    "Date": dt.strftime("%Y-%m-%d"),
                    "Price_USD_per_MT": price,
                    "Source": "PLACEHOLDER",
                })

    bunker_df = pd.DataFrame(rows)
    out_path = OUTPUT / "bunker_forward_curves.csv"
    bunker_df.to_csv(out_path, index=False)
    print(f"[4] Bunker lookup stub ({len(bunker_df)} rows) saved to {out_path}")
    print("    ⚠ Using PLACEHOLDER prices — replace with actual forward curve data")
    return bunker_df


def lookup_bunker_price(port: str, fuel_type: str, date: str, bunker_df: pd.DataFrame):
    """Look up bunker price for a given port, fuel type, and date."""
    mask = (
        (bunker_df["Port"].str.upper() == port.upper())
        & (bunker_df["Fuel_Type"] == fuel_type)
    )
    matches = bunker_df[mask].copy()
    if matches.empty:
        return None
    matches["Date"] = pd.to_datetime(matches["Date"])
    target = pd.to_datetime(date)
    matches["delta"] = (matches["Date"] - target).abs()
    return matches.loc[matches["delta"].idxmin(), "Price_USD_per_MT"]


# ═══════════════════════════════════════════════════════════════════════════
# 5. SAILING TIME CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_sailing_times(distance_matrix: pd.DataFrame, vessels: pd.DataFrame):
    """
    Calculate ballast and laden sailing times for each vessel between all port pairs.

    Sailing time (days) = distance (nm) / (speed (kn) × 24)

    Produces:
      - sailing_times_laden_warranted.csv   (vessel × port-pair)
      - sailing_times_ballast_warranted.csv
      - sailing_times_laden_economical.csv
      - sailing_times_ballast_economical.csv
    """
    ports = list(distance_matrix.index)

    configs = {
        "laden_warranted": "Warranted Speed Laden (kn)",
        "ballast_warranted": "Warranted Speed Ballast (kn)",
        "laden_economical": "Economical Speed Laden (kn)",
        "ballast_economical": "Economical Speed Ballast (kn)",
    }

    all_results = {}

    for label, speed_col in configs.items():
        rows = []
        for _, v in vessels.iterrows():
            name = v["Vessel Name"]
            speed = v.get(speed_col, np.nan)
            if pd.isna(speed) or speed == 0:
                continue

            for i, p1 in enumerate(ports):
                for j, p2 in enumerate(ports):
                    if i >= j:
                        continue
                    dist = distance_matrix.iloc[i, j]
                    if pd.isna(dist):
                        continue
                    time_days = round(dist / (speed * 24), 2)
                    rows.append({
                        "Vessel": name,
                        "From_Port": p1,
                        "To_Port": p2,
                        "Distance_NM": dist,
                        "Speed_kn": speed,
                        "Sailing_Days": time_days,
                    })

        df = pd.DataFrame(rows)
        out_path = OUTPUT / f"sailing_times_{label}.csv"
        df.to_csv(out_path, index=False)
        all_results[label] = df
        print(f"[5] Sailing times ({label}): {len(df)} rows saved to {out_path}")

    # Also create a summary: for each vessel, sailing time from current position
    # to each load port (ballast voyage to pick up cargo)
    summary_rows = []
    for _, v in vessels.iterrows():
        name = v["Vessel Name"]
        current_port = str(v.get("Current Position / Status", "")).strip()
        speed_b = v.get("Warranted Speed Ballast (kn)", np.nan)
        speed_l = v.get("Warranted Speed Laden (kn)", np.nan)

        if current_port not in ports:
            continue

        for port in ports:
            if port == current_port:
                continue
            dist = distance_matrix.loc[current_port, port]
            if pd.isna(dist):
                continue

            ballast_days = round(dist / (speed_b * 24), 2) if speed_b else np.nan
            laden_days = round(dist / (speed_l * 24), 2) if speed_l else np.nan

            summary_rows.append({
                "Vessel": name,
                "Current_Port": current_port,
                "Destination_Port": port,
                "Distance_NM": dist,
                "Ballast_Days_Warranted": ballast_days,
                "Laden_Days_Warranted": laden_days,
            })

    summary = pd.DataFrame(summary_rows)
    out_path = OUTPUT / "vessel_to_port_sailing_times.csv"
    summary.to_csv(out_path, index=False)
    print(f"[5] Vessel-to-port sailing summary: {len(summary)} rows saved to {out_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("CARGILL DATA PROCESSING PIPELINE")
    print("=" * 70)
    print()

    dist_matrix = build_distance_matrix()
    print()

    vessels = build_vessel_specs()
    print()

    cargoes = build_cargo_requirements()
    print()

    bunkers = build_bunker_lookup()
    print()

    sailing = build_sailing_times(dist_matrix, vessels)
    print()

    print("=" * 70)
    print("PIPELINE COMPLETE — All outputs in ./processed/")
    print("=" * 70)
