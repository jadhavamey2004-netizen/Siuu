"""
FloatChat Ocean Analyzer - Fixed Version
Uses measurements.parquet for clean data, no time filters, direct responses
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


class OceanDataProcessor:
    """Processes ocean float data using the clean measurements file."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        self.trajectory_data = None
        
    def load_data(self):
        """Load and process ocean data files using the clean measurements data."""
        try:
            # Use the clean measurements.parquet file
            measurements_file = Path(self.data_path) / "measurements.parquet"
            trajectory_file = Path(self.data_path) / "trajectory.parquet"
            
            print(f"Loading measurements from: {measurements_file}")
            
            if not measurements_file.exists():
                print(f"Measurements file not found: {measurements_file}")
                self.data = pd.DataFrame()
                return
                
            # Load measurements data
            measurements_df = pd.read_parquet(measurements_file)
            print(f"Raw measurements loaded: {len(measurements_df)} rows, {len(measurements_df.columns)} columns")
            
            # Load trajectory data for timestamps
            if trajectory_file.exists():
                self.trajectory_data = pd.read_parquet(trajectory_file)
                print(f"Trajectory data loaded: {len(self.trajectory_data)} rows")
                
                # Create a mapping from profile_id to datetime using trajectory
                # Assume each profile corresponds to trajectory points in sequence
                profile_datetime_map = {}
                unique_profiles = measurements_df['profile_id'].unique()
                
                # Simple approach: distribute trajectory timestamps across profiles
                trajectory_dates = self.trajectory_data['date'].values
                for i, profile_id in enumerate(sorted(unique_profiles)):
                    if i < len(trajectory_dates):
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_dates[i])
                    else:
                        # Use the last available date
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_dates[-1])
                
                # Add datetime column based on profile_id
                measurements_df['datetime'] = measurements_df['profile_id'].map(profile_datetime_map)
            else:
                # Create synthetic timestamps if no trajectory data
                print("No trajectory data found, creating synthetic timestamps")
                base_date = datetime(2005, 5, 13)  # Based on the sample data
                measurements_df['datetime'] = [
                    base_date + timedelta(days=i//100) for i in range(len(measurements_df))
                ]
            
            # Clean the data
            self.data = measurements_df.copy()
            
            # Remove any rows with NaN values in critical columns
            self.data = self.data.dropna(subset=['pressure', 'temperature', 'salinity'])
            
            # Remove obviously invalid data (zeros or extreme values)
            self.data = self.data[
                (self.data['pressure'] > 0) & 
                (self.data['temperature'] > -5) & (self.data['temperature'] < 50) &
                (self.data['salinity'] > 0) & (self.data['salinity'] < 50)
            ]
            
            print(f"✓ Processed {len(self.data)} valid records")
            if not self.data.empty:
                print(f"  Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
                print(f"  Temperature range: {self.data['temperature'].min():.2f} to {self.data['temperature'].max():.2f}°C")
                print(f"  Salinity range: {self.data['salinity'].min():.2f} to {self.data['salinity'].max():.2f} PSU")
                print(f"  Pressure range: {self.data['pressure'].min():.1f} to {self.data['pressure'].max():.1f} dbar")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            self.data = pd.DataFrame()
    
    def create_time_series_plots(self, filtered_data, title_suffix=""):
        """Create time series plots for temperature, salinity, and pressure."""
        
        if filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'Temperature over Time {title_suffix}',
                f'Salinity over Time {title_suffix}', 
                f'Pressure over Time {title_suffix}'
            ),
            vertical_spacing=0.08
        )
        
        # Temperature plot
        fig.add_trace(
            go.Scatter(
                x=filtered_data['datetime'],
                y=filtered_data['temperature'],
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Salinity plot
        fig.add_trace(
            go.Scatter(
                x=filtered_data['datetime'],
                y=filtered_data['salinity'],
                mode='lines+markers',
                name='Salinity (PSU)',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Pressure plot (inverted for depth-like appearance)
        fig.add_trace(
            go.Scatter(
                x=filtered_data['datetime'],
                y=filtered_data['pressure'],
                mode='lines+markers',
                name='Pressure (dbar)',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"Ocean Float Time Series Analysis {title_suffix}",
            showlegend=True,
            template="plotly_white"
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Salinity (PSU)", row=2, col=1)
        fig.update_yaxes(title_text="Pressure (dbar)", autorange="reversed", row=3, col=1)
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Date/Time", row=3, col=1)
        
        return fig
    
    def get_data_summary(self, data=None):
        """Get summary statistics for the data."""
        if data is None:
            data = self.data
            
        if data is None or data.empty:
            return {}
        
        summary = {
            'total_records': len(data),
            'date_range': {
                'start': data['datetime'].min(),
                'end': data['datetime'].max()
            } if 'datetime' in data.columns else None,
            'temperature': {
                'min': data['temperature'].min(),
                'max': data['temperature'].max(),
                'mean': data['temperature'].mean()
            } if 'temperature' in data.columns else None,
            'salinity': {
                'min': data['salinity'].min(),
                'max': data['salinity'].max(),
                'mean': data['salinity'].mean()
            } if 'salinity' in data.columns else None,
            'pressure': {
                'min': data['pressure'].min(),
                'max': data['pressure'].max(),
                'mean': data['pressure'].mean()
            } if 'pressure' in data.columns else None
        }
        
        return summary
    
    def search_data(self, query):
        """Process search queries and return relevant data subset - single definitive result."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(), "No data available"
        
        query_lower = query.lower().strip()
        
        # Always return a definitive result based on the query
        if 'temperature' in query_lower or 'temp' in query_lower:
            if 'high' in query_lower or 'warm' in query_lower or 'hot' in query_lower:
                # Return top 25% of temperature readings
                threshold = self.data['temperature'].quantile(0.75)
                result_data = self.data[self.data['temperature'] >= threshold].head(100)
                response_text = f"High temperature readings: {len(result_data)} records ≥{threshold:.2f}°C"
            elif 'low' in query_lower or 'cold' in query_lower or 'cool' in query_lower:
                # Return bottom 25% of temperature readings
                threshold = self.data['temperature'].quantile(0.25)
                result_data = self.data[self.data['temperature'] <= threshold].head(100)
                response_text = f"Low temperature readings: {len(result_data)} records ≤{threshold:.2f}°C"
            else:
                # Return temperature data overview
                result_data = self.data.head(100)
                temp_range = f"{self.data['temperature'].min():.2f}°C to {self.data['temperature'].max():.2f}°C"
                response_text = f"Temperature overview: Range {temp_range}, Mean {self.data['temperature'].mean():.2f}°C"
        
        elif 'salinity' in query_lower or 'salt' in query_lower:
            if 'high' in query_lower:
                threshold = self.data['salinity'].quantile(0.75)
                result_data = self.data[self.data['salinity'] >= threshold].head(100)
                response_text = f"High salinity readings: {len(result_data)} records ≥{threshold:.2f} PSU"
            elif 'low' in query_lower:
                threshold = self.data['salinity'].quantile(0.25)
                result_data = self.data[self.data['salinity'] <= threshold].head(100)
                response_text = f"Low salinity readings: {len(result_data)} records ≤{threshold:.2f} PSU"
            else:
                result_data = self.data.head(100)
                sal_range = f"{self.data['salinity'].min():.2f} to {self.data['salinity'].max():.2f} PSU"
                response_text = f"Salinity overview: Range {sal_range}, Mean {self.data['salinity'].mean():.2f} PSU"
        
        elif 'pressure' in query_lower or 'depth' in query_lower:
            if 'high' in query_lower or 'deep' in query_lower:
                threshold = self.data['pressure'].quantile(0.75)
                result_data = self.data[self.data['pressure'] >= threshold].head(100)
                response_text = f"High pressure/deep readings: {len(result_data)} records ≥{threshold:.1f} dbar"
            elif 'low' in query_lower or 'shallow' in query_lower:
                threshold = self.data['pressure'].quantile(0.25)
                result_data = self.data[self.data['pressure'] <= threshold].head(100)
                response_text = f"Low pressure/shallow readings: {len(result_data)} records ≤{threshold:.1f} dbar"
            else:
                result_data = self.data.head(100)
                pres_range = f"{self.data['pressure'].min():.1f} to {self.data['pressure'].max():.1f} dbar"
                response_text = f"Pressure overview: Range {pres_range}, Mean {self.data['pressure'].mean():.1f} dbar"
        
        elif 'recent' in query_lower or 'latest' in query_lower or 'newest' in query_lower:
            # Return most recent 50 records
            result_data = self.data.nlargest(50, 'datetime') if 'datetime' in self.data.columns else self.data.tail(50)
            response_text = f"Most recent {len(result_data)} measurements from the dataset"
        
        elif 'overview' in query_lower or 'summary' in query_lower or 'all' in query_lower:
            # Return general overview
            result_data = self.data.head(50)
            response_text = f"Dataset overview: {len(self.data)} total records available"
        
        else:
            # Default: return first 50 records as sample
            result_data = self.data.head(50)
            response_text = f"Sample data: Showing {len(result_data)} records from {len(self.data)} total records"
        
        return result_data, response_text


# Initialize the Dash app
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

app.title = "FloatChat Ocean Analyzer"

# Initialize data processor
processor = OceanDataProcessor()

# Load data at startup
print("🔄 Loading ocean data at startup...")
processor.load_data()

# Simplified app layout - removed time range filters
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FloatChat Ocean Analyzer", 
                   className="text-primary text-center mb-4"),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        # Left Panel - Search Only
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Natural Language Search", className="text-primary")),
                dbc.CardBody([
                    # Data Loading Status
                    html.Div(id="data-status", className="mb-3"),
                    
                    # Search Input
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Try: 'high temperature', 'low salinity', 'deep water', 'recent data'",
                            type="text",
                            className="mb-2"
                        ),
                        dbc.Button("Search", id="search-btn", color="primary", size="sm")
                    ], className="mb-3"),
                    
                    # Quick Search Buttons
                    html.Div([
                        dbc.Button("High Temp", id="btn-high-temp", color="outline-danger", size="sm", className="me-2 mb-2"),
                        dbc.Button("Low Temp", id="btn-low-temp", color="outline-info", size="sm", className="me-2 mb-2"),
                        dbc.Button("High Salinity", id="btn-high-sal", color="outline-primary", size="sm", className="me-2 mb-2"),
                        dbc.Button("Deep Water", id="btn-deep", color="outline-success", size="sm", className="me-2 mb-2"),
                        dbc.Button("Recent Data", id="btn-recent", color="outline-warning", size="sm", className="me-2 mb-2"),
                        dbc.Button("Overview", id="btn-overview", color="outline-secondary", size="sm", className="me-2 mb-2"),
                    ], className="mb-3"),
                    
                    # Search Results Text
                    html.Hr(),
                    html.H6("Search Results:", className="mb-2"),
                    html.Div(id="search-results-text", 
                            style={"maxHeight": "300px", "overflowY": "auto"},
                            className="border p-2 bg-light rounded")
                ])
            ])
        ], width=4),
        
        # Right Panel - Visualizations
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Ocean Data Visualization", className="text-primary d-inline"),
                    dbc.Badge("Live Data", color="success", className="ms-2")
                ]),
                dbc.CardBody([
                    # Loading indicator
                    dcc.Loading(
                        id="loading-graphs",
                        children=[
                            # Main time series plot
                            dcc.Graph(id="time-series-plot", 
                                    style={"height": "600px"}),
                        ],
                        type="circle"
                    ),
                    
                    # Data Summary
                    html.Hr(),
                    html.H6("Data Summary", className="mb-2"),
                    html.Div(id="data-summary", className="row")
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    # Data Table for detailed view
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Detailed Data View")),
                dbc.CardBody([
                    html.Div(id="data-table-container")
                ])
            ])
        ], width=12)
    ])
], fluid=True)


# Component to show data status
@app.callback(
    Output("data-status", "children"),
    Input("search-btn", "n_clicks"),  # Dummy input to trigger
    prevent_initial_call=False
)
def show_data_status(_):
    if processor.data is not None and not processor.data.empty:
        return dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"✓ Data loaded: {len(processor.data):,} records",
            html.Br(),
            f"Temperature: {processor.data['temperature'].min():.1f}°C to {processor.data['temperature'].max():.1f}°C",
            html.Br(),
            f"Salinity: {processor.data['salinity'].min():.1f} to {processor.data['salinity'].max():.1f} PSU",
            html.Br(),
            f"Pressure: {processor.data['pressure'].min():.1f} to {processor.data['pressure'].max():.1f} dbar"
        ], color="success")
    else:
        return dbc.Alert([
            html.I(className="fas fa-exclamation-triangle me-2"),
            "⚠ No data loaded. Check data path."
        ], color="warning")


# Main callback for search and visualization
@app.callback(
    [Output("time-series-plot", "figure"),
     Output("data-summary", "children"),
     Output("search-results-text", "children"),
     Output("data-table-container", "children"),
     Output("search-input", "value")],
    [Input("search-btn", "n_clicks"),
     Input("btn-high-temp", "n_clicks"),
     Input("btn-low-temp", "n_clicks"),
     Input("btn-high-sal", "n_clicks"),
     Input("btn-deep", "n_clicks"),
     Input("btn-recent", "n_clicks"),
     Input("btn-overview", "n_clicks")],
    State("search-input", "value"),
    prevent_initial_call=False
)
def update_visualization(search_clicks, btn1, btn2, btn3, btn4, btn5, btn6, search_query):
    try:
        # Determine which button was clicked
        ctx = dash.callback_context
        if not ctx.triggered:
            # Initial load - show overview
            search_query = "overview"
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if trigger_id == "btn-high-temp":
                search_query = "high temperature"
            elif trigger_id == "btn-low-temp":
                search_query = "low temperature"
            elif trigger_id == "btn-high-sal":
                search_query = "high salinity"
            elif trigger_id == "btn-deep":
                search_query = "deep water"
            elif trigger_id == "btn-recent":
                search_query = "recent data"
            elif trigger_id == "btn-overview":
                search_query = "overview"
            # If search button clicked, use the input value
        
        # Default query if none provided
        if not search_query or not search_query.strip():
            search_query = "overview"
        
        # Process search
        if processor.data is not None and not processor.data.empty:
            search_result_data, search_text = processor.search_data(search_query)
            
            # Create visualization
            fig = processor.create_time_series_plots(search_result_data, f"- {search_text}")
            
            # Get summary
            summary = processor.get_data_summary(search_result_data)
            if summary:
                summary_cards = [
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Records", className="card-title text-muted"),
                                html.H4(f"{summary['total_records']:,}", className="text-primary")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg Temperature", className="card-title text-muted"),
                                html.H4(f"{summary['temperature']['mean']:.2f}°C" if summary['temperature'] else "N/A", 
                                        className="text-danger")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg Salinity", className="card-title text-muted"),
                                html.H4(f"{summary['salinity']['mean']:.2f} PSU" if summary['salinity'] else "N/A", 
                                        className="text-info")
                            ])
                        ])
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H6("Avg Pressure", className="card-title text-muted"),
                                html.H4(f"{summary['pressure']['mean']:.1f} dbar" if summary['pressure'] else "N/A", 
                                        className="text-success")
                            ])
                        ])
                    ], width=3)
                ]
            else:
                summary_cards = [html.P("No summary available")]
            
            # Create data table
            if not search_result_data.empty:
                display_cols = ['datetime', 'temperature', 'salinity', 'pressure']
                available_cols = [col for col in display_cols if col in search_result_data.columns]
                
                table_data = search_result_data[available_cols].head(50).copy()
                
                if 'datetime' in table_data.columns:
                    table_data['datetime'] = table_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                columns = []
                for col in available_cols:
                    if col == 'datetime':
                        columns.append({"name": "Date/Time", "id": "datetime"})
                    elif col == 'temperature':
                        columns.append({"name": "Temperature (°C)", "id": "temperature", "type": "numeric", "format": {"specifier": ".2f"}})
                    elif col == 'salinity':
                        columns.append({"name": "Salinity (PSU)", "id": "salinity", "type": "numeric", "format": {"specifier": ".2f"}})
                    elif col == 'pressure':
                        columns.append({"name": "Pressure (dbar)", "id": "pressure", "type": "numeric", "format": {"specifier": ".1f"}})
                
                data_table = dash_table.DataTable(
                    data=table_data.to_dict('records'),
                    columns=columns,
                    style_table={"maxHeight": "300px", "overflowY": "auto"},
                    style_cell={"textAlign": "left", "fontSize": 12},
                    style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"}
                )
            else:
                data_table = html.P("No data available for the current search.", className="text-muted text-center p-3")
        
        else:
            # No data loaded
            fig = go.Figure()
            fig.add_annotation(
                text="Please load ocean data first",
                xref="paper", yref="paper", x=0.5, y=0.5, 
                xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            summary_cards = [html.P("No data available")]
            search_text = "No data loaded"
            data_table = html.P("No data available")
        
        return fig, summary_cards, search_text, data_table, ""
        
    except Exception as e:
        # Error handling
        print(f"Error in callback: {e}")
        import traceback
        traceback.print_exc()
        
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False, font_size=14
        )
        
        return error_fig, [html.P(f"Error: {str(e)}", className="text-danger")], f"Error: {str(e)}", html.P("Error"), ""


if __name__ == "__main__":
    print("🌊 Starting FloatChat Ocean Analyzer (Fixed)...")
    print("📊 Using clean measurements data for visualization...")
    print("🚀 Server starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)