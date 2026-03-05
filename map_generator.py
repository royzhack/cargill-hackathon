"""
Map Visualization Generator
Creates interactive maps showing vessel routes and cargo locations
"""

import folium
from folium import plugins
import pandas as pd
import json
from pathlib import Path
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def create_vessel_routes_map():
    """
    Create an interactive map showing all vessel routes, cargo locations, and ports.
    Returns path to generated HTML file.
    """
    try:
        # Load portfolio data - use risk_adjusted file as source of truth
        portfolio_path = Path('processed/portfolio_summary_risk_adjusted.json')
        if not portfolio_path.exists():
            return None
        
        with open(portfolio_path) as f:
            risk_data = json.load(f)
        
        if 'assignments' not in risk_data or len(risk_data['assignments']) == 0:
            return None
        
        # Convert risk-adjusted assignments to expected format
        assignments = []
        for assignment in risk_data['assignments']:
            # Parse route to get ports
            route = assignment.get('route', '')
            load_port = assignment.get('load_port', '')
            discharge_port = assignment.get('discharge_port', '')
            
            if not load_port or not discharge_port:
                if '→' in route or '->' in route:
                    parts = route.replace('→', '->').split('->')
                    if len(parts) == 2:
                        load_port = parts[0].strip()
                        discharge_port = parts[1].strip()
            
            converted_assignment = {
                'Vessel_Name': assignment.get('vessel', 'Unknown'),
                'Cargo_ID': assignment.get('cargo', assignment.get('route', 'Unknown')),
                'Load_Port': load_port,
                'Discharge_Port': discharge_port,
                'Leg_Profit': assignment.get('base_profit', 0),
                'TCE_Leg': assignment.get('base_tce', 0),
                'Leg_Days': assignment.get('voyage_days', 0),
                'Vessel_Type': assignment.get('fleet', 'Unknown'),
                'Cargo_Type': 'Market' if 'MARKET' in str(assignment.get('cargo', '')).upper() else 'Committed'
            }
            assignments.append(converted_assignment)
        
        # Load port locations
        port_locations_path = Path('data/port_locations.csv')
        if not port_locations_path.exists():
            return None
        
        port_locations = pd.read_csv(port_locations_path)
        
        # Calculate map center from all ports used
        all_ports = set()
        for assignment in assignments:
            if assignment.get('Load_Port'):
                all_ports.add(assignment.get('Load_Port'))
            if assignment.get('Discharge_Port'):
                all_ports.add(assignment.get('Discharge_Port'))
        
        # Get coordinates for all ports
        port_coords = {}
        for port_name in all_ports:
            port_row = port_locations[port_locations['port_name'] == port_name]
            if len(port_row) > 0:
                port_coords[port_name] = {
                    'lat': port_row.iloc[0]['latitude'],
                    'lon': port_row.iloc[0]['longitude']
                }
        
        if not port_coords:
            return None
        
        # Calculate center
        center_lat = sum(p['lat'] for p in port_coords.values()) / len(port_coords)
        center_lon = sum(p['lon'] for p in port_coords.values()) / len(port_coords)
        
        # Create map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=3,
            tiles='OpenStreetMap'
        )
        
        # Add tile layers
        folium.TileLayer('CartoDB positron').add_to(m)
        folium.TileLayer('CartoDB dark_matter').add_to(m)
        
        # Color palette for vessels
        vessel_names = list(set(a['Vessel_Name'] for a in assignments))
        colors = cm.tab20(range(len(vessel_names)))
        vessel_colors = {}
        for i, vessel_name in enumerate(vessel_names):
            vessel_colors[vessel_name] = mcolors.rgb2hex(colors[i])
        
        # Group assignments by vessel
        vessel_assignments = {}
        for assignment in assignments:
            vessel_name = assignment['Vessel_Name']
            if vessel_name not in vessel_assignments:
                vessel_assignments[vessel_name] = []
            vessel_assignments[vessel_name].append(assignment)
        
        # Sort by leg number for each vessel
        for vessel_name in vessel_assignments:
            vessel_assignments[vessel_name].sort(key=lambda x: x.get('Leg_Number', 0))
        
        # Plot routes for each vessel
        for vessel_name, vessel_legs in vessel_assignments.items():
            color = vessel_colors[vessel_name]
            vessel_type = vessel_legs[0].get('Vessel_Type', 'Unknown')
            
            # Create route coordinates list
            route_coords = []
            
            # Plot each leg
            for leg in vessel_legs:
                load_port = leg.get('Load_Port')
                discharge_port = leg.get('Discharge_Port')
                cargo_id = leg.get('Cargo_ID', 'Unknown')
                cargo_type = leg.get('Cargo_Type', 'Unknown')
                quantity = leg.get('Quantity_MT', 0)
                profit = leg.get('Leg_Profit', 0)
                tce = leg.get('TCE_Leg', 0)
                
                if load_port in port_coords and discharge_port in port_coords:
                    load_coords = port_coords[load_port]
                    disc_coords = port_coords[discharge_port]
                    
                    # Add load port marker
                    marker_color = 'blue' if cargo_type == 'Committed' else 'orange'
                    folium.Marker(
                        [load_coords['lat'], load_coords['lon']],
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: Arial, sans-serif;">
                                <h4 style="margin: 0 0 8px 0; color: #2d333a;">{cargo_id}</h4>
                                <p style="margin: 4px 0;"><strong>Type:</strong> {cargo_type}</p>
                                <p style="margin: 4px 0;"><strong>Port:</strong> {load_port}</p>
                                <p style="margin: 4px 0;"><strong>Quantity:</strong> {quantity:,.0f} MT</p>
                                <p style="margin: 4px 0;"><strong>Vessel:</strong> {vessel_name}</p>
                            </div>
                            """,
                            max_width=250
                        ),
                        tooltip=f"Load: {load_port} ({cargo_id})",
                        icon=folium.Icon(color=marker_color, icon='upload', prefix='fa')
                    ).add_to(m)
                    
                    # Add discharge port marker
                    folium.Marker(
                        [disc_coords['lat'], disc_coords['lon']],
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: Arial, sans-serif;">
                                <h4 style="margin: 0 0 8px 0; color: #2d333a;">{cargo_id}</h4>
                                <p style="margin: 4px 0;"><strong>Type:</strong> {cargo_type}</p>
                                <p style="margin: 4px 0;"><strong>Port:</strong> {discharge_port}</p>
                                <p style="margin: 4px 0;"><strong>Quantity:</strong> {quantity:,.0f} MT</p>
                                <p style="margin: 4px 0;"><strong>Profit:</strong> ${profit:,.0f}</p>
                                <p style="margin: 4px 0;"><strong>TCE:</strong> ${tce:,.0f}/day</p>
                                <p style="margin: 4px 0;"><strong>Vessel:</strong> {vessel_name}</p>
                            </div>
                            """,
                            max_width=250
                        ),
                        tooltip=f"Discharge: {discharge_port} ({cargo_id})",
                        icon=folium.Icon(color='red', icon='download', prefix='fa')
                    ).add_to(m)
                    
                    # Draw route line
                    folium.PolyLine(
                        [[load_coords['lat'], load_coords['lon']], 
                         [disc_coords['lat'], disc_coords['lon']]],
                        color=color,
                        weight=4,
                        opacity=0.7,
                        popup=folium.Popup(
                            f"""
                            <div style="font-family: Arial, sans-serif;">
                                <h4 style="margin: 0 0 8px 0; color: #2d333a;">{vessel_name}</h4>
                                <p style="margin: 4px 0;"><strong>Cargo:</strong> {cargo_id}</p>
                                <p style="margin: 4px 0;"><strong>Route:</strong> {load_port} → {discharge_port}</p>
                                <p style="margin: 4px 0;"><strong>Profit:</strong> ${profit:,.0f}</p>
                                <p style="margin: 4px 0;"><strong>TCE:</strong> ${tce:,.0f}/day</p>
                            </div>
                            """,
                            max_width=250
                        ),
                        tooltip=f"{vessel_name}: {load_port} → {discharge_port}"
                    ).add_to(m)
                    
                    # Add to route coordinates
                    route_coords.append([load_coords['lat'], load_coords['lon']])
                    route_coords.append([disc_coords['lat'], disc_coords['lon']])
            
            # Add vessel route summary (dashed line connecting all legs)
            if len(route_coords) > 1:
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=2,
                    opacity=0.4,
                    dashArray='10, 5',
                    tooltip=f"{vessel_name} Complete Route"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px; border-radius: 8px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <h4 style="margin: 0 0 10px 0; color: #2d333a;">Map Legend</h4>
        <p style="margin: 4px 0;"><i class="fa fa-upload" style="color: blue;"></i> Load Port (Committed)</p>
        <p style="margin: 4px 0;"><i class="fa fa-upload" style="color: orange;"></i> Load Port (Market)</p>
        <p style="margin: 4px 0;"><i class="fa fa-download" style="color: red;"></i> Discharge Port</p>
        <p style="margin: 4px 0;"><span style="display: inline-block; width: 20px; height: 3px; background-color: #10a37f; margin-right: 5px;"></span> Vessel Route</p>
        <p style="margin: 4px 0; font-size: 12px; color: #6e6e80;">Click markers for details</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        output_path = Path('diagrams/vessel_routes_map.html')
        output_path.parent.mkdir(exist_ok=True)
        m.save(str(output_path))
        
        return output_path
        
    except Exception as e:
        print(f"Error creating map: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    result = create_vessel_routes_map()
    if result:
        print(f"Map created: {result}")
    else:
        print("Failed to create map")
