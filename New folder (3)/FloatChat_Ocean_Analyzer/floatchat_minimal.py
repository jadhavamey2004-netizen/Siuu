"""
FloatChat Ocean Analyzer - Ultra Simple Version
Small data chunk, minimal interface, clean visualization only
"""
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path


class SimpleOceanProcessor:
    """Simplified processor with small data chunks for clean visualization."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load a small chunk of data for simple visualization."""
        try:
            measurements_file = Path(self.data_path) / "measurements.parquet"
            trajectory_file = Path(self.data_path) / "trajectory.parquet"
            
            if not measurements_file.exists():
                print(f"File not found: {measurements_file}")
                self.data = pd.DataFrame()
                return
                
            # Load measurements data
            measurements_df = pd.read_parquet(measurements_file)
            
            # Take only a small sample for visualization (every 50th record)
            sample_data = measurements_df.iloc[::50].copy()  # Every 50th record
            
            # Load trajectory data for timestamps
            if trajectory_file.exists():
                trajectory_df = pd.read_parquet(trajectory_file)
                
                # Create simple datetime mapping
                profile_datetime_map = {}
                unique_profiles = sample_data['profile_id'].unique()
                
                for i, profile_id in enumerate(sorted(unique_profiles)):
                    if i < len(trajectory_df):
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_df.iloc[i]['date'])
                    else:
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_df.iloc[-1]['date'])
                
                sample_data['datetime'] = sample_data['profile_id'].map(profile_datetime_map)
            else:
                # Simple synthetic timestamps
                base_date = datetime(2005, 5, 13)
                sample_data['datetime'] = [
                    base_date + timedelta(hours=i*6) for i in range(len(sample_data))
                ]
            
            # Clean data - remove invalid values
            sample_data = sample_data[
                (sample_data['pressure'] > 0) & 
                (sample_data['temperature'] > 0) & (sample_data['temperature'] < 40) &
                (sample_data['salinity'] > 30) & (sample_data['salinity'] < 40)
            ].copy()
            
            # Take only first 100 records for ultra-simple visualization
            self.data = sample_data.head(100)
            
            print(f"✓ Loaded {len(self.data)} sample records for visualization")
            if not self.data.empty:
                print(f"  Temperature: {self.data['temperature'].min():.1f}°C to {self.data['temperature'].max():.1f}°C")
                print(f"  Salinity: {self.data['salinity'].min():.1f} to {self.data['salinity'].max():.1f} PSU")
                print(f"  Pressure: {self.data['pressure'].min():.0f} to {self.data['pressure'].max():.0f} dbar")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
    
    def create_simple_plots(self, title="Ocean Data"):
        """Create simple time series plots."""
        
        if self.data is None or self.data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        # Create simple 3-panel plot
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Temperature (°C)',
                'Salinity (PSU)', 
                'Pressure (dbar)'
            ),
            vertical_spacing=0.1
        )
        
        # Temperature
        fig.add_trace(
            go.Scatter(
                x=self.data['datetime'],
                y=self.data['temperature'],
                mode='lines+markers',
                name='Temperature',
                line=dict(color='red', width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Salinity
        fig.add_trace(
            go.Scatter(
                x=self.data['datetime'],
                y=self.data['salinity'],
                mode='lines+markers',
                name='Salinity',
                line=dict(color='blue', width=3),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        # Pressure
        fig.add_trace(
            go.Scatter(
                x=self.data['datetime'],
                y=self.data['pressure'],
                mode='lines+markers',
                name='Pressure',
                line=dict(color='green', width=3),
                marker=dict(size=6)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=700,
            title_text=title,
            showlegend=False,
            template="plotly_white"
        )
        
        # Clean y-axis
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="PSU", row=2, col=1)
        fig.update_yaxes(title_text="dbar", row=3, col=1)
        
        return fig
    
    def get_simple_summary(self):
        """Get simple data summary."""
        if self.data is None or self.data.empty:
            return {}
        
        return {
            'records': len(self.data),
            'temp_avg': self.data['temperature'].mean(),
            'sal_avg': self.data['salinity'].mean(),
            'pres_avg': self.data['pressure'].mean()
        }


# Initialize app
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

app.title = "FloatChat - Simple Ocean Viewer"

# Initialize processor and load data
processor = SimpleOceanProcessor()
print("🔄 Loading sample ocean data...")
processor.load_data()

# Ultra-simple layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("FloatChat - Simple Ocean Viewer", 
                   className="text-primary text-center mb-4"),
        ], width=12)
    ]),
    
    dbc.Row([
        # Left Panel - Minimal Controls
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("Quick Actions", className="text-primary mb-3"),
                    
                    # Simple action buttons
                    dbc.ButtonGroup([
                        dbc.Button("Show All Data", id="btn-all", color="primary", size="sm"),
                        dbc.Button("Temperature Focus", id="btn-temp", color="danger", size="sm"),
                        dbc.Button("Salinity Focus", id="btn-sal", color="info", size="sm"),
                    ], className="mb-3", vertical=True),
                    
                    html.Hr(),
                    
                    # Data info
                    html.Div(id="data-info"),
                ])
            ])
        ], width=3),
        
        # Right Panel - Clean Visualization
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(id="ocean-plot", style={"height": "650px"})
                    ], type="circle")
                ])
            ])
        ], width=9)
    ], className="mb-4"),
    
    # Simple data table
    dbc.Row([
        dbc.Col([
            html.H6("Sample Data (First 20 records)", className="mb-2"),
            html.Div(id="simple-table")
        ], width=12)
    ])
], fluid=True)


# Show data info
@app.callback(
    Output("data-info", "children"),
    Input("btn-all", "n_clicks"),
    prevent_initial_call=False
)
def show_data_info(_):
    if processor.data is not None and not processor.data.empty:
        summary = processor.get_simple_summary()
        return [
            html.P(f"📊 {summary['records']} data points", className="mb-1"),
            html.P(f"🌡️ Avg Temp: {summary['temp_avg']:.1f}°C", className="mb-1"),
            html.P(f"💧 Avg Salinity: {summary['sal_avg']:.1f} PSU", className="mb-1"),
            html.P(f"📏 Avg Pressure: {summary['pres_avg']:.0f} dbar", className="mb-1"),
        ]
    else:
        return html.P("⚠️ No data loaded", className="text-warning")


# Main visualization callback
@app.callback(
    [Output("ocean-plot", "figure"),
     Output("simple-table", "children")],
    [Input("btn-all", "n_clicks"),
     Input("btn-temp", "n_clicks"),
     Input("btn-sal", "n_clicks")],
    prevent_initial_call=False
)
def update_view(btn_all, btn_temp, btn_sal):
    try:
        # Determine which button was clicked
        ctx = dash.callback_context
        title = "Ocean Float Data - Sample View"
        
        if ctx.triggered:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == "btn-temp":
                title = "Temperature Focus View"
            elif trigger_id == "btn-sal":
                title = "Salinity Focus View"
        
        # Create plot
        fig = processor.create_simple_plots(title)
        
        # Create simple table (first 20 records)
        if processor.data is not None and not processor.data.empty:
            table_data = processor.data.head(20).copy()
            
            # Format datetime for display
            if 'datetime' in table_data.columns:
                table_data['datetime'] = table_data['datetime'].dt.strftime('%m-%d %H:%M')
            
            # Create simple table
            simple_table = dash_table.DataTable(
                data=table_data[['datetime', 'temperature', 'salinity', 'pressure']].to_dict('records'),
                columns=[
                    {"name": "Time", "id": "datetime"},
                    {"name": "Temp (°C)", "id": "temperature", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "Sal (PSU)", "id": "salinity", "type": "numeric", "format": {"specifier": ".1f"}},
                    {"name": "Pres (dbar)", "id": "pressure", "type": "numeric", "format": {"specifier": ".0f"}}
                ],
                style_table={"height": "200px", "overflowY": "auto"},
                style_cell={"textAlign": "center", "fontSize": 12, "padding": "5px"},
                style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
                style_data={"backgroundColor": "#ffffff"}
            )
        else:
            simple_table = html.P("No data to display", className="text-center text-muted")
        
        return fig, simple_table
        
    except Exception as e:
        print(f"Error: {e}")
        
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False, font_size=14
        )
        
        return error_fig, html.P("Error loading data")


if __name__ == "__main__":
    print("🌊 FloatChat Simple Ocean Viewer")
    print("📊 Minimal interface with sample data")
    print("🚀 Starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)