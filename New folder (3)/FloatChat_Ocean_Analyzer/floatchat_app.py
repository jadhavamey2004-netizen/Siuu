"""
FloatChat Ocean Analyzer - Complete Application
Time-series focused dashboard with JULD conversion and quarterly scaling
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
    """Processes ocean float data with time conversion and filtering."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        self.trajectory_data = None
        
    def juld_to_datetime(self, juld_values):
        """
        Convert JULD (Julian Date) to datetime.
        JULD is days since 1950-01-01 00:00:00 UTC
        """
        reference_date = datetime(1950, 1, 1)
        
        # Handle both single values and arrays
        if isinstance(juld_values, (int, float)):
            if pd.isna(juld_values):
                return None
            return reference_date + timedelta(days=float(juld_values))
        
        # Handle pandas Series/arrays
        datetimes = []
        for juld in juld_values:
            if pd.isna(juld):
                datetimes.append(None)
            else:
                datetimes.append(reference_date + timedelta(days=float(juld)))
        
        return pd.Series(datetimes)
    
    def load_data(self):
        """Load and process all ocean data files."""
        try:
            # Load main profile data
            profile_file = Path(self.data_path) / "full_profile_data.parquet"
            measurements_file = Path(self.data_path) / "measurements.parquet"
            trajectory_file = Path(self.data_path) / "trajectory.parquet"
            
            if profile_file.exists():
                profile_df = pd.read_parquet(profile_file)
                
                # Convert JULD to datetime
                profile_df['datetime'] = self.juld_to_datetime(profile_df['JULD'])
                
                # Select relevant columns
                self.data = profile_df[['datetime', 'CYCLE_NUMBER', 'LATITUDE', 'LONGITUDE', 
                                       'PRES', 'TEMP', 'PSAL', 'PRES_ADJUSTED', 
                                       'TEMP_ADJUSTED', 'PSAL_ADJUSTED']].copy()
                
                # Clean data - remove NaN values
                self.data = self.data.dropna(subset=['datetime'])
                
                # Use adjusted values if available, otherwise use raw values
                self.data['pressure'] = self.data['PRES_ADJUSTED'].fillna(self.data['PRES'])
                self.data['temperature'] = self.data['TEMP_ADJUSTED'].fillna(self.data['TEMP'])
                self.data['salinity'] = self.data['PSAL_ADJUSTED'].fillna(self.data['PSAL'])
                
                # Add time-based filtering columns
                self.data['year'] = self.data['datetime'].dt.year
                self.data['month'] = self.data['datetime'].dt.month
                self.data['quarter'] = self.data['datetime'].dt.quarter
                
                print(f"✓ Loaded {len(self.data)} profile records")
                print(f"  Date range: {self.data['datetime'].min()} to {self.data['datetime'].max()}")
                
            # Load trajectory data
            if trajectory_file.exists():
                self.trajectory_data = pd.read_parquet(trajectory_file)
                self.trajectory_data['date'] = pd.to_datetime(self.trajectory_data['date'])
                print(f"✓ Loaded {len(self.trajectory_data)} trajectory points")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
            
    def filter_by_time_range(self, start_month=1, end_month=12, years=None):
        """
        Filter data by time range (quarterly or custom).
        """
        if self.data is None or self.data.empty:
            return pd.DataFrame()
        
        filtered_data = self.data.copy()
        
        # Filter by months
        filtered_data = filtered_data[
            (filtered_data['month'] >= start_month) & 
            (filtered_data['month'] <= end_month)
        ]
        
        # Filter by years if specified
        if years:
            filtered_data = filtered_data[filtered_data['year'].isin(years)]
            
        return filtered_data
    
    def get_quarterly_data(self, quarter=1, years=None):
        """Get data for a specific quarter."""
        quarter_months = {
            1: (1, 3),   # Jan-Mar
            2: (4, 6),   # Apr-Jun
            3: (7, 9),   # Jul-Sep
            4: (10, 12)  # Oct-Dec
        }
        
        start_month, end_month = quarter_months.get(quarter, (1, 3))
        return self.filter_by_time_range(start_month, end_month, years)
    
    def create_time_series_plots(self, filtered_data, title_suffix=""):
        """Create time series plots for temperature, salinity, and pressure."""
        
        if filtered_data.empty:
            # Return empty plot
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for the selected time range",
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
    
    def get_data_summary(self, filtered_data=None):
        """Get summary statistics for the data."""
        if filtered_data is None:
            filtered_data = self.data
            
        if filtered_data is None or filtered_data.empty:
            return {}
        
        summary = {
            'total_records': len(filtered_data),
            'date_range': {
                'start': filtered_data['datetime'].min(),
                'end': filtered_data['datetime'].max()
            },
            'temperature': {
                'min': filtered_data['temperature'].min(),
                'max': filtered_data['temperature'].max(),
                'mean': filtered_data['temperature'].mean()
            },
            'salinity': {
                'min': filtered_data['salinity'].min(),
                'max': filtered_data['salinity'].max(),
                'mean': filtered_data['salinity'].mean()
            },
            'pressure': {
                'min': filtered_data['pressure'].min(),
                'max': filtered_data['pressure'].max(),
                'mean': filtered_data['pressure'].mean()
            }
        }
        
        return summary
    
    def search_data(self, query, filtered_data=None):
        """Process search queries and return relevant data subset."""
        if filtered_data is None:
            filtered_data = self.data
            
        if filtered_data is None or filtered_data.empty:
            return pd.DataFrame(), "No data available"
        
        query_lower = query.lower()
        result_data = filtered_data.copy()
        response_text = ""
        
        # Temperature queries
        if 'temperature' in query_lower or 'temp' in query_lower:
            if 'high' in query_lower or 'warm' in query_lower:
                threshold = filtered_data['temperature'].quantile(0.75)
                result_data = filtered_data[filtered_data['temperature'] >= threshold]
                response_text = f"Found {len(result_data)} records with high temperatures (≥{threshold:.2f}°C)"
            elif 'low' in query_lower or 'cold' in query_lower:
                threshold = filtered_data['temperature'].quantile(0.25)
                result_data = filtered_data[filtered_data['temperature'] <= threshold]
                response_text = f"Found {len(result_data)} records with low temperatures (≤{threshold:.2f}°C)"
            else:
                response_text = f"Temperature data: {filtered_data['temperature'].min():.2f}°C to {filtered_data['temperature'].max():.2f}°C"
        
        # Salinity queries
        elif 'salinity' in query_lower or 'salt' in query_lower:
            if 'high' in query_lower:
                threshold = filtered_data['salinity'].quantile(0.75)
                result_data = filtered_data[filtered_data['salinity'] >= threshold]
                response_text = f"Found {len(result_data)} records with high salinity (≥{threshold:.2f} PSU)"
            elif 'low' in query_lower:
                threshold = filtered_data['salinity'].quantile(0.25)
                result_data = filtered_data[filtered_data['salinity'] <= threshold]
                response_text = f"Found {len(result_data)} records with low salinity (≤{threshold:.2f} PSU)"
            else:
                response_text = f"Salinity data: {filtered_data['salinity'].min():.2f} to {filtered_data['salinity'].max():.2f} PSU"
        
        # Pressure/Depth queries
        elif 'pressure' in query_lower or 'depth' in query_lower:
            if 'deep' in query_lower or 'high pressure' in query_lower:
                threshold = filtered_data['pressure'].quantile(0.75)
                result_data = filtered_data[filtered_data['pressure'] >= threshold]
                response_text = f"Found {len(result_data)} records at high pressure/depth (≥{threshold:.1f} dbar)"
            elif 'shallow' in query_lower or 'low pressure' in query_lower:
                threshold = filtered_data['pressure'].quantile(0.25)
                result_data = filtered_data[filtered_data['pressure'] <= threshold]
                response_text = f"Found {len(result_data)} records at low pressure/shallow depth (≤{threshold:.1f} dbar)"
            else:
                response_text = f"Pressure data: {filtered_data['pressure'].min():.1f} to {filtered_data['pressure'].max():.1f} dbar"
        
        # Time-based queries
        elif any(word in query_lower for word in ['recent', 'latest', 'newest']):
            result_data = filtered_data.nlargest(100, 'datetime')
            response_text = f"Showing {len(result_data)} most recent measurements"
        
        # Default: return summary
        else:
            result_data = filtered_data.head(50)
            response_text = f"Showing overview of {len(result_data)} records from the dataset"
        
        # Limit result size for performance
        if len(result_data) > 200:
            result_data = result_data.head(200)
            response_text += f" (limited to 200 records for display)"
        
        return result_data, response_text


# Initialize the Dash app
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP],
               suppress_callback_exceptions=True)

app.title = "FloatChat Ocean Analyzer"

# Initialize data processor
processor = OceanDataProcessor()

# App layout with left search panel and right visualization panel
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("FloatChat Ocean Analyzer", 
                   className="text-primary text-center mb-4"),
            html.Hr()
        ], width=12)
    ]),
    
    dbc.Row([
        # Left Panel - Search and Controls
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H4("Search & Controls", className="text-primary")),
                dbc.CardBody([
                    # Data Loading Status
                    html.Div(id="data-status", className="mb-3"),
                    
                    # Search Input
                    html.H5("Natural Language Search", className="mb-2"),
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Ask about temperature, salinity, pressure, or time periods...",
                            type="text",
                            className="mb-2"
                        ),
                        dbc.Button("Search", id="search-btn", color="primary", size="sm")
                    ], className="mb-3"),
                    
                    # Time Range Controls
                    html.Hr(),
                    html.H5("Time Range Filter", className="mb-2"),
                    
                    # Quarter Selection
                    html.Label("Select Quarter:", className="fw-bold mb-1"),
                    dcc.Dropdown(
                        id="quarter-dropdown",
                        options=[
                            {"label": "Q1 (Jan-Mar)", "value": 1},
                            {"label": "Q2 (Apr-Jun)", "value": 2}, 
                            {"label": "Q3 (Jul-Sep)", "value": 3},
                            {"label": "Q4 (Oct-Dec)", "value": 4},
                            {"label": "All Quarters", "value": "all"}
                        ],
                        value="all",
                        className="mb-3"
                    ),
                    
                    # Custom Month Range
                    html.Label("Or Custom Month Range:", className="fw-bold mb-1"),
                    dcc.RangeSlider(
                        id="month-range-slider",
                        min=1, max=12, step=1,
                        marks={i: str(i) for i in range(1, 13)},
                        value=[1, 12],
                        className="mb-3"
                    ),
                    
                    # Year Selection
                    html.Label("Select Years:", className="fw-bold mb-1"),
                    dcc.Dropdown(
                        id="year-dropdown",
                        multi=True,
                        placeholder="Select years (leave empty for all)",
                        className="mb-3"
                    ),
                    
                    # Update Button
                    dbc.Button("Update Visualization", id="update-btn", 
                              color="success", size="lg", className="w-100 mb-3"),
                    
                    # Search Results Text
                    html.Hr(),
                    html.H6("Search Results:", className="mb-2"),
                    html.Div(id="search-results-text", 
                            style={"maxHeight": "200px", "overflowY": "auto"},
                            className="border p-2 bg-light rounded")
                ])
            ])
        ], width=4),
        
        # Right Panel - Visualizations
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H4("Time Series Visualization", className="text-primary d-inline"),
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
    
    # Optional: Data Table for detailed view
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Detailed Data View (Limited to 50 records)")),
                dbc.CardBody([
                    html.Div(id="data-table-container")
                ])
            ])
        ], width=12)
    ]),
    
    # Hidden div to trigger initial data load
    html.Div(id="app-load", style={"display": "none"})
], fluid=True)


# Callback to load data and populate year dropdown on app startup
@app.callback(
    [Output("data-status", "children"),
     Output("year-dropdown", "options")],
    [Input("app-load", "data")],
    prevent_initial_call=False
)
def load_data_on_startup(_):
    try:
        processor.load_data()
        
        if processor.data is not None and not processor.data.empty:
            # Get available years
            available_years = sorted(processor.data['year'].unique())
            year_options = [{"label": str(year), "value": year} for year in available_years]
            
            status = dbc.Alert([
                html.I(className="fas fa-check-circle me-2"),
                f"✓ Data loaded successfully! {len(processor.data)} records available.",
                html.Br(),
                f"Date range: {processor.data['datetime'].min().strftime('%Y-%m-%d')} to {processor.data['datetime'].max().strftime('%Y-%m-%d')}"
            ], color="success")
        else:
            year_options = []
            status = dbc.Alert([
                html.I(className="fas fa-exclamation-triangle me-2"),
                "⚠ No data loaded. Please check data path."
            ], color="warning")
            
    except Exception as e:
        year_options = []
        status = dbc.Alert([
            html.I(className="fas fa-times-circle me-2"),
            f"✗ Error loading data: {str(e)}"
        ], color="danger")
    
    return status, year_options


# Main callback for updating visualizations
@app.callback(
    [Output("time-series-plot", "figure"),
     Output("data-summary", "children"),
     Output("search-results-text", "children"),
     Output("data-table-container", "children")],
    [Input("update-btn", "n_clicks"),
     Input("search-btn", "n_clicks")],
    [State("quarter-dropdown", "value"),
     State("month-range-slider", "value"),
     State("year-dropdown", "value"),
     State("search-input", "value")]
)
def update_visualization(update_clicks, search_clicks, quarter, month_range, selected_years, search_query):
    try:
        # Determine which input triggered the callback
        ctx = dash.callback_context
        title_suffix = ""
        
        if not ctx.triggered:
            # Initial load
            filtered_data = processor.data if processor.data is not None else pd.DataFrame()
            search_result_data = pd.DataFrame()
            search_text = "Enter a search query to explore the data"
        else:
            # Filter data based on time range selection
            if quarter == "all":
                start_month, end_month = month_range
                filtered_data = processor.filter_by_time_range(start_month, end_month, selected_years)
                title_suffix = f"(Months {start_month}-{end_month})"
            else:
                filtered_data = processor.get_quarterly_data(quarter, selected_years)
                title_suffix = f"(Q{quarter})"
            
            # Add year filter info to title
            if selected_years:
                year_text = ", ".join(map(str, selected_years))
                title_suffix += f" - Years: {year_text}"
            
            # Handle search
            search_result_data = pd.DataFrame()
            search_text = "Enter a search query to explore the data"
            
            if search_query and search_query.strip():
                search_result_data, search_text = processor.search_data(search_query, filtered_data)
        
        # Create time series plot
        if filtered_data is not None and not filtered_data.empty:
            fig = processor.create_time_series_plots(filtered_data, title_suffix)
            
            # Get data summary
            summary = processor.get_data_summary(filtered_data)
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
                            html.H4(f"{summary['temperature']['mean']:.2f}°C", className="text-danger")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Avg Salinity", className="card-title text-muted"),
                            html.H4(f"{summary['salinity']['mean']:.2f} PSU", className="text-info")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Avg Pressure", className="card-title text-muted"),
                            html.H4(f"{summary['pressure']['mean']:.1f} dbar", className="text-success")
                        ])
                    ])
                ], width=3)
            ]
        else:
            # Empty data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available. Please load data first.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16
            )
            summary_cards = [html.P("No data available for summary.")]
        
        # Create data table
        if not search_result_data.empty:
            # Prepare data for table display
            table_data = search_result_data[['datetime', 'temperature', 'salinity', 'pressure']].head(50)
            table_data = table_data.copy()
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
            data_table = html.P("No data to display. Use the search function to explore specific data.", 
                               className="text-muted text-center p-3")
        
        return fig, summary_cards, search_text, data_table
        
    except Exception as e:
        # Error handling
        error_fig = go.Figure()
        error_fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font_size=14
        )
        
        error_summary = [html.P(f"Error: {str(e)}", className="text-danger")]
        error_text = f"Error processing request: {str(e)}"
        error_table = html.P("Error loading data table.", className="text-danger")
        
        return error_fig, error_summary, error_text, error_table


if __name__ == "__main__":
    print("🌊 Starting FloatChat Ocean Analyzer...")
    print("📊 Loading ocean float data with JULD conversion...")
    print("🚀 Server starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)