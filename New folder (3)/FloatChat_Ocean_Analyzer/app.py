"""
Main FloatChat Ocean Analyzer Application
Time-series focused dashboard with JULD conversion and quarterly scaling
"""
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from data_processor import OceanDataProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Current path:", sys.path)
    # Fallback - try direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location("data_processor", 
                                                 os.path.join(os.path.dirname(__file__), 'src', 'data_processor.py'))
    data_processor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(data_processor_module)
    OceanDataProcessor = data_processor_module.OceanDataProcessor

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
    ])
], fluid=True)

# Callback to load data and populate year dropdown on app startup
@app.callback(
    [Output("data-status", "children"),
     Output("year-dropdown", "options")],
    [Input("app-load", "data")],  # Dummy input to trigger on load
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
            fig = processor.create_time_series_plots(filtered_data, title_suffix if 'title_suffix' in locals() else "")
            
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

# Hidden div to trigger initial data load
app.layout.children.append(html.Div(id="app-load", style={"display": "none"}))

if __name__ == "__main__":
    print("🌊 Starting FloatChat Ocean Analyzer...")
    print("📊 Loading ocean float data with JULD conversion...")
    print("🚀 Server starting on http://localhost:8050")
    
    app.run_server(debug=True, port=8050)