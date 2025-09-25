"""
FloatChat Ocean Analyzer - Simplified Version
Direct visualization with natural language search, no time filters
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
    """Processes ocean float data with simplified approach."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        
    def safe_juld_to_datetime(self, juld_value):
        """Safely convert JULD to datetime with error handling."""
        try:
            if pd.isna(juld_value) or juld_value is None:
                return None
            
            # Handle extremely large values that might cause overflow
            if juld_value > 100000 or juld_value < -100000:
                return None
                
            reference_date = datetime(1950, 1, 1)
            return reference_date + timedelta(days=float(juld_value))
        except (OverflowError, ValueError, TypeError):
            return None
    
    def load_data(self):
        """Load and process ocean data files with robust error handling."""
        try:
            # Try to load the main profile data
            profile_file = Path(self.data_path) / "full_profile_data.parquet"
            
            if not profile_file.exists():
                print(f"Profile file not found: {profile_file}")
                self.data = pd.DataFrame()
                return
                
            print(f"Loading data from: {profile_file}")
            profile_df = pd.read_parquet(profile_file)
            print(f"Raw data loaded: {len(profile_df)} rows, {len(profile_df.columns)} columns")
            
            # Convert JULD to datetime with robust error handling
            datetime_list = []
            juld_values = profile_df['JULD'].values if 'JULD' in profile_df.columns else []
            
            for juld in juld_values:
                dt = self.safe_juld_to_datetime(juld)
                datetime_list.append(dt)
            
            profile_df['datetime'] = datetime_list
            
            # Select and clean relevant columns
            required_cols = ['datetime', 'PRES', 'TEMP', 'PSAL']
            optional_cols = ['CYCLE_NUMBER', 'LATITUDE', 'LONGITUDE', 'PRES_ADJUSTED', 'TEMP_ADJUSTED', 'PSAL_ADJUSTED']
            
            available_cols = ['datetime']
            for col in required_cols[1:]:  # Skip datetime as we just created it
                if col in profile_df.columns:
                    available_cols.append(col)
            
            for col in optional_cols:
                if col in profile_df.columns:
                    available_cols.append(col)
            
            self.data = profile_df[available_cols].copy()
            
            # Remove rows with invalid datetime
            self.data = self.data.dropna(subset=['datetime'])
            
            # Create clean measurement columns
            self.data['pressure'] = (
                self.data.get('PRES_ADJUSTED', self.data.get('PRES', 0))
                if 'PRES_ADJUSTED' in self.data.columns or 'PRES' in self.data.columns 
                else 0
            )
            
            self.data['temperature'] = (
                self.data.get('TEMP_ADJUSTED', self.data.get('TEMP', 0))
                if 'TEMP_ADJUSTED' in self.data.columns or 'TEMP' in self.data.columns 
                else 0
            )
            
            self.data['salinity'] = (
                self.data.get('PSAL_ADJUSTED', self.data.get('PSAL', 0))
                if 'PSAL_ADJUSTED' in self.data.columns or 'PSAL' in self.data.columns 
                else 0
            )
            
            # Remove rows where all measurements are 0 or NaN
            self.data = self.data[
                (self.data['pressure'] != 0) | 
                (self.data['temperature'] != 0) | 
                (self.data['salinity'] != 0)
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
        
        # Pressure plot
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
                # Return all temperature data summary
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
            result_data = self.data.nlargest(50, 'datetime')
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

# Load data at startup (outside of callback)
processor.load_data()

# Create initial data status
def get_initial_data_status():
    try:
        if processor.data is not None and not processor.data.empty:
            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✓ Data loaded: {len(processor.data)} records",
                html.Br(),
                f"Date range: {processor.data['datetime'].min().strftime('%Y-%m-%d')} to {processor.data['datetime'].max().strftime('%Y-%m-%d')}"
            ], color="success")
        else:
            status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "⚠ No data loaded. Check data path or file format."
            ], color="warning")
    except Exception as e:
        status = dbc.Alert([
            html.I(className="fas fa-times-circle me-2"),
            f"✗ Error loading data: {str(e)}"
        ], color="danger")
    
    return status

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
                    html.Div(get_initial_data_status(), className="mb-3"),
                    
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
    ]),
    
], fluid=True)


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
    State("search-input", "value")
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
                table_data = search_result_data[['datetime', 'temperature', 'salinity', 'pressure']].head(50).copy()
                table_data['datetime'] = table_data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                data_table = dash_table.DataTable(
                    data=table_data.to_dict('records'),
                    columns=[
                        {"name": "Date/Time", "id": "datetime"},
                        {"name": "Temperature (°C)", "id": "temperature", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Salinity (PSU)", "id": "salinity", "type": "numeric", "format": {"specifier": ".2f"}},
                        {"name": "Pressure (dbar)", "id": "pressure", "type": "numeric", "format": {"specifier": ".1f"}}
                    ],
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
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, 
            xanchor='center', yanchor='middle',
            showarrow=False, font_size=14
        )
        
        return error_fig, [html.P(f"Error: {str(e)}", className="text-danger")], f"Error: {str(e)}", html.P("Error"), ""


if __name__ == "__main__":
    print("🌊 Starting FloatChat Ocean Analyzer (Simplified)...")
    print("📊 Direct ocean data visualization with natural language search...")
    print("🚀 Server starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)