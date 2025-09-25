"""
FloatChat Ocean Analyzer - Beautiful Chat Interface
Clean chat-based interaction with improved visualizations
"""
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ALL
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import uuid


class ChatOceanProcessor:
    """Chat-based ocean data processor with beautiful visualizations."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        
    def load_data(self):
        """Load sample data for visualization."""
        try:
            measurements_file = Path(self.data_path) / "measurements.parquet"
            trajectory_file = Path(self.data_path) / "trajectory.parquet"
            
            if not measurements_file.exists():
                self.data = pd.DataFrame()
                return
                
            # Load and sample data
            measurements_df = pd.read_parquet(measurements_file)
            sample_data = measurements_df.iloc[::50].copy()  # Every 50th record
            
            # Add timestamps
            if trajectory_file.exists():
                trajectory_df = pd.read_parquet(trajectory_file)
                profile_datetime_map = {}
                unique_profiles = sample_data['profile_id'].unique()
                
                for i, profile_id in enumerate(sorted(unique_profiles)):
                    if i < len(trajectory_df):
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_df.iloc[i]['date'])
                    else:
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_df.iloc[-1]['date'])
                
                sample_data['datetime'] = sample_data['profile_id'].map(profile_datetime_map)
            
            # Clean and limit data
            sample_data = sample_data[
                (sample_data['pressure'] > 0) & 
                (sample_data['temperature'] > 0) & (sample_data['temperature'] < 40) &
                (sample_data['salinity'] > 30) & (sample_data['salinity'] < 40)
            ].copy()
            
            self.data = sample_data.head(100)
            print(f"✓ Loaded {len(self.data)} sample records")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
    
    def create_beautiful_plots(self, filtered_data, query="Ocean Data"):
        """Create beautiful, clean visualizations like in the images."""
        
        if filtered_data is None or filtered_data.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font_size=16, font_color="#666"
            )
            return fig
        
        # Create clean subplots with proper spacing
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'Temperature over Time - {query}',
                f'Salinity over Time - {query}', 
                f'Pressure over Time - {query}'
            ),
            vertical_spacing=0.12,
            row_heights=[0.33, 0.33, 0.34]
        )
        
        # Beautiful color scheme
        colors = {
            'temperature': '#FF4B4B',  # Red
            'salinity': '#1f77b4',     # Blue  
            'pressure': '#2E8B57'      # Green
        }
        
        # Temperature plot - clean line with markers
        fig.add_trace(
            go.Scatter(
                x=filtered_data['datetime'],
                y=filtered_data['temperature'],
                mode='lines+markers',
                name='Temperature (°C)',
                line=dict(color=colors['temperature'], width=2),
                marker=dict(size=4, color=colors['temperature']),
                hovertemplate='<b>Temperature</b><br>%{y:.1f}°C<br>%{x}<extra></extra>'
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
                line=dict(color=colors['salinity'], width=2),
                marker=dict(size=4, color=colors['salinity']),
                hovertemplate='<b>Salinity</b><br>%{y:.2f} PSU<br>%{x}<extra></extra>'
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
                line=dict(color=colors['pressure'], width=2),
                marker=dict(size=4, color=colors['pressure']),
                hovertemplate='<b>Pressure</b><br>%{y:.0f} dbar<br>%{x}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # Beautiful styling
        fig.update_layout(
            height=800,
            title={
                'text': f"Ocean Float Time Series Analysis - {query}",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            showlegend=False,
            template="plotly_white",
            font=dict(size=12),
            margin=dict(t=80, b=60, l=60, r=60)
        )
        
        # Clean axis styling
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1, title_font=dict(color=colors['temperature']))
        fig.update_yaxes(title_text="Salinity (PSU)", row=2, col=1, title_font=dict(color=colors['salinity']))
        fig.update_yaxes(title_text="Pressure (dbar)", row=3, col=1, title_font=dict(color=colors['pressure']))
        
        # Clean x-axis
        fig.update_xaxes(title_text="Date/Time", row=3, col=1, title_font=dict(size=12))
        
        return fig
    
    def process_chat_query(self, query):
        """Process chat queries and return data with response."""
        if self.data is None or self.data.empty:
            return pd.DataFrame(), "No data available. Please check data connection."
        
        query_lower = query.lower().strip()
        
        if 'high temperature' in query_lower or 'warm' in query_lower:
            threshold = self.data['temperature'].quantile(0.75)
            result_data = self.data[self.data['temperature'] >= threshold]
            response = f"Found {len(result_data)} high temperature readings (≥{threshold:.1f}°C)"
            
        elif 'low temperature' in query_lower or 'cold' in query_lower:
            threshold = self.data['temperature'].quantile(0.25)
            result_data = self.data[self.data['temperature'] <= threshold]
            response = f"Found {len(result_data)} low temperature readings (≤{threshold:.1f}°C)"
            
        elif 'high salinity' in query_lower or 'salty' in query_lower:
            threshold = self.data['salinity'].quantile(0.75)
            result_data = self.data[self.data['salinity'] >= threshold]
            response = f"Found {len(result_data)} high salinity readings (≥{threshold:.2f} PSU)"
            
        elif 'low salinity' in query_lower:
            threshold = self.data['salinity'].quantile(0.25)
            result_data = self.data[self.data['salinity'] <= threshold]
            response = f"Found {len(result_data)} low salinity readings (≤{threshold:.2f} PSU)"
            
        elif 'deep' in query_lower or 'high pressure' in query_lower:
            threshold = self.data['pressure'].quantile(0.75)
            result_data = self.data[self.data['pressure'] >= threshold]
            response = f"Found {len(result_data)} deep water readings (≥{threshold:.0f} dbar)"
            
        elif 'shallow' in query_lower or 'low pressure' in query_lower:
            threshold = self.data['pressure'].quantile(0.25)
            result_data = self.data[self.data['pressure'] <= threshold]
            response = f"Found {len(result_data)} shallow water readings (≤{threshold:.0f} dbar)"
            
        elif 'recent' in query_lower or 'latest' in query_lower:
            result_data = self.data.tail(30)
            response = f"Showing the latest {len(result_data)} measurements"
            
        elif 'overview' in query_lower or 'summary' in query_lower or 'all' in query_lower:
            result_data = self.data
            response = f"Complete dataset overview: {len(result_data)} records"
            
        else:
            result_data = self.data.head(50)
            response = f"Sample data: {len(result_data)} records from the ocean dataset"
        
        return result_data, response


# Initialize app
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               suppress_callback_exceptions=True)

app.title = "FloatChat - Ocean Data Assistant"

# Initialize processor
processor = ChatOceanProcessor()
print("🔄 Loading ocean data for chat interface...")
processor.load_data()

# Chat interface layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("🌊 FloatChat", className="text-primary mb-0"),
                html.P("Ocean Data Assistant", className="text-muted small mb-0")
            ], className="text-center py-3")
        ], width=12)
    ], className="mb-3"),
    
    dbc.Row([
        # Left Panel - Chat Interface
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.H5("💬 Chat with Ocean Data", className="mb-0 text-primary")
                ]),
                dbc.CardBody([
                    # Chat messages container
                    html.Div(id="chat-messages", 
                            children=[
                                html.Div([
                                    html.Div([
                                        html.I(className="fas fa-robot me-2"),
                                        "Hi! I'm your ocean data assistant. Ask me about temperature, salinity, pressure, or say 'show overview'."
                                    ], className="p-3 bg-light rounded mb-3 small")
                                ])
                            ],
                            style={"height": "400px", "overflowY": "auto", "border": "1px solid #e9ecef", "padding": "10px"}),
                    
                    # Chat input
                    dbc.InputGroup([
                        dbc.Input(
                            id="chat-input",
                            placeholder="Ask about ocean data...",
                            type="text"
                        ),
                        dbc.Button([
                            html.I(className="fas fa-paper-plane")
                        ], id="send-btn", color="primary")
                    ], className="mt-3"),
                    
                    # Quick suggestions
                    html.Div([
                        html.Small("💡 Try: ", className="text-muted"),
                        dbc.ButtonGroup([
                            dbc.Button("High Temperature", id="quick-temp", size="sm", color="outline-danger"),
                            dbc.Button("High Salinity", id="quick-sal", size="sm", color="outline-primary"),
                            dbc.Button("Deep Water", id="quick-deep", size="sm", color="outline-success"),
                        ], size="sm")
                    ], className="mt-2")
                ])
            ])
        ], width=4),
        
        # Right Panel - Beautiful Visualization
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Loading([
                        dcc.Graph(id="ocean-viz", style={"height": "600px"})
                    ], type="circle")
                ])
            ])
        ], width=8)
    ], className="mb-4"),
    
    # Summary cards
    dbc.Row([
        dbc.Col([
            html.Div(id="summary-cards")
        ], width=12)
    ]),
    
    # Store for chat history
    dcc.Store(id="chat-store", data=[])
], fluid=True, style={"backgroundColor": "#f8f9fa", "minHeight": "100vh", "padding": "20px"})


# Chat functionality
@app.callback(
    [Output("chat-messages", "children"),
     Output("chat-input", "value"),
     Output("chat-store", "data"),
     Output("ocean-viz", "figure"),
     Output("summary-cards", "children")],
    [Input("send-btn", "n_clicks"),
     Input("chat-input", "n_submit"),
     Input("quick-temp", "n_clicks"),
     Input("quick-sal", "n_clicks"),
     Input("quick-deep", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-store", "data")],
    prevent_initial_call=False
)
def handle_chat(send_clicks, input_submit, quick_temp, quick_sal, quick_deep, input_text, chat_history):
    try:
        # Determine what was triggered
        ctx = dash.callback_context
        
        # Initial load
        if not ctx.triggered:
            query = "overview"
            show_user_message = False
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == "quick-temp":
                query = "high temperature"
                show_user_message = True
            elif trigger_id == "quick-sal":
                query = "high salinity"
                show_user_message = True
            elif trigger_id == "quick-deep":
                query = "deep water"
                show_user_message = True
            elif trigger_id in ["send-btn", "chat-input"] and input_text and input_text.strip():
                query = input_text.strip()
                show_user_message = True
            else:
                # No valid input, return current state
                return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        # Process the query
        result_data, response_text = processor.process_chat_query(query)
        
        # Update chat history
        if chat_history is None:
            chat_history = []
        
        new_messages = []
        
        # Add user message if needed
        if show_user_message:
            chat_history.append({"type": "user", "text": query, "id": str(uuid.uuid4())})
        
        # Add bot response
        chat_history.append({"type": "bot", "text": response_text, "id": str(uuid.uuid4())})
        
        # Create chat message components
        chat_components = []
        for msg in chat_history[-10:]:  # Show last 10 messages
            if msg["type"] == "user":
                chat_components.append(
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-user me-2"),
                            msg["text"]
                        ], className="p-2 bg-primary text-white rounded mb-2 small text-end")
                    ], className="text-end")
                )
            else:
                chat_components.append(
                    html.Div([
                        html.Div([
                            html.I(className="fas fa-robot me-2"),
                            msg["text"]
                        ], className="p-2 bg-light rounded mb-2 small")
                    ])
                )
        
        # Create visualization
        fig = processor.create_beautiful_plots(result_data, response_text)
        
        # Create summary cards
        if not result_data.empty:
            summary_cards = dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{len(result_data)}", className="text-primary mb-0"),
                            html.Small("Records", className="text-muted")
                        ], className="text-center py-2")
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{result_data['temperature'].mean():.1f}°C", className="text-danger mb-0"),
                            html.Small("Avg Temperature", className="text-muted")
                        ], className="text-center py-2")
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{result_data['salinity'].mean():.2f} PSU", className="text-info mb-0"),
                            html.Small("Avg Salinity", className="text-muted")
                        ], className="text-center py-2")
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(f"{result_data['pressure'].mean():.0f} dbar", className="text-success mb-0"),
                            html.Small("Avg Pressure", className="text-muted")
                        ], className="text-center py-2")
                    ])
                ], width=3)
            ], className="mt-3")
        else:
            summary_cards = html.Div()
        
        return chat_components, "", chat_history, fig, summary_cards
        
    except Exception as e:
        print(f"Error in chat: {e}")
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# Auto-scroll chat to bottom
app.clientside_callback(
    """
    function(children) {
        setTimeout(function() {
            const chatDiv = document.getElementById('chat-messages');
            if (chatDiv) {
                chatDiv.scrollTop = chatDiv.scrollHeight;
            }
        }, 100);
        return '';
    }
    """,
    Output("chat-messages", "title"),
    Input("chat-messages", "children")
)

if __name__ == "__main__":
    print("🌊 FloatChat - Beautiful Ocean Data Assistant")
    print("💬 Chat interface with clean visualizations")
    print("🚀 Starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)