"""
FloatChat Ocean Dashboard - Modern UI Design
Beautiful dashboard interface inspired by financial/analytics dashboards
"""
import dash
from dash import dcc, html, Input, Output, State, callback, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path


class OceanDashboardProcessor:
    """Ocean data processor for dashboard display."""
    
    def __init__(self, data_path=r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"):
        self.data_path = data_path
        self.data = None
        self.full_data = None  # Store full dataset for filtering
        
    def load_data(self):
        """Load and process ocean data for dashboard."""
        try:
            measurements_file = Path(self.data_path) / "measurements.parquet"
            trajectory_file = Path(self.data_path) / "trajectory.parquet"
            
            if not measurements_file.exists():
                self.data = pd.DataFrame()
                return
                
            # Load measurements
            measurements_df = pd.read_parquet(measurements_file)
            sample_data = measurements_df.iloc[::20].copy()  # Every 20th record for more data
            
            # Add timestamps
            if trajectory_file.exists():
                trajectory_df = pd.read_parquet(trajectory_file)
                profile_datetime_map = {}
                unique_profiles = sample_data['profile_id'].unique()
                
                for i, profile_id in enumerate(sorted(unique_profiles)):
                    if i < len(trajectory_df):
                        profile_datetime_map[profile_id] = pd.to_datetime(trajectory_df.iloc[i]['date'])
                
                sample_data['datetime'] = sample_data['profile_id'].map(profile_datetime_map)
            
            # Clean data
            sample_data = sample_data[
                (sample_data['pressure'] > 0) & 
                (sample_data['temperature'] > 0) & (sample_data['temperature'] < 40) &
                (sample_data['salinity'] > 30) & (sample_data['salinity'] < 40)
            ].copy()
            
            # Use ALL cleaned data instead of limiting
            self.data = sample_data.copy()  # Use all available data
            self.full_data = sample_data.copy()  # Same as data for consistency
            print(f"✓ Dashboard loaded {len(self.data)} ocean records (full dataset)")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = pd.DataFrame()
            self.full_data = pd.DataFrame()
    
    def search_data(self, query_text="", depth_level="all", temp_range=None, sal_range=None):
        """Search and filter ocean data based on query parameters."""
        if self.full_data is None or self.full_data.empty:
            return pd.DataFrame()
        
        filtered_data = self.full_data.copy()
        
        # Apply depth level filtering
        if depth_level != "all":
            if depth_level == "surface":
                filtered_data = filtered_data[filtered_data['pressure'] <= 100]
            elif depth_level == "shallow":
                filtered_data = filtered_data[(filtered_data['pressure'] > 100) & (filtered_data['pressure'] <= 500)]
            elif depth_level == "deep":
                filtered_data = filtered_data[(filtered_data['pressure'] > 500) & (filtered_data['pressure'] <= 1000)]
            elif depth_level == "very_deep":
                filtered_data = filtered_data[filtered_data['pressure'] > 1000]
        
        # Apply temperature range filtering
        if temp_range:
            filtered_data = filtered_data[
                (filtered_data['temperature'] >= temp_range[0]) & 
                (filtered_data['temperature'] <= temp_range[1])
            ]
        
        # Apply salinity range filtering
        if sal_range:
            filtered_data = filtered_data[
                (filtered_data['salinity'] >= sal_range[0]) & 
                (filtered_data['salinity'] <= sal_range[1])
            ]
        
        # Text-based search (search in profile_id or other fields)
        if query_text:
            query_text = query_text.lower()
            # Search in profile_id or create text search
            if 'profile_id' in filtered_data.columns:
                filtered_data = filtered_data[
                    filtered_data['profile_id'].astype(str).str.lower().str.contains(query_text, na=False)
                ]
        
        return filtered_data  # Return ALL filtered results, no limit
    
    def get_kpi_metrics(self):
        """Get KPI metrics for dashboard cards."""
        if self.data is None or self.data.empty:
            return {}
        
        return {
            'avg_temp': self.data['temperature'].mean(),
            'max_temp': self.data['temperature'].max(),
            'avg_salinity': self.data['salinity'].mean(),
            'max_pressure': self.data['pressure'].max(),
            'total_records': len(self.data),
            'temp_trend': self.data['temperature'].iloc[-10:].mean() - self.data['temperature'].iloc[:10].mean(),
            'sal_trend': self.data['salinity'].iloc[-10:].mean() - self.data['salinity'].iloc[:10].mean()
        }
    
    def create_main_chart(self, chart_type="overview", query_context="", selected_param="temperature"):
        """Create optimized main chart with ultra-smooth visualizations."""
        if self.data is None or self.data.empty:
            return go.Figure()
        
        # Sort data by datetime and optimize for smoothest visualization
        if 'datetime' in self.data.columns:
            plot_data = self.data.sort_values('datetime').copy()
            x_axis = plot_data['datetime']
            x_title = "📅 Time Period"
            
            # Advanced sampling for ultra-smooth curves
            if len(plot_data) > 60:
                # Use percentile-based sampling for better distribution
                import numpy as np
                n_points = min(50, len(plot_data) // 2)
                indices = np.linspace(0, len(plot_data) - 1, n_points, dtype=int)
                plot_data = plot_data.iloc[indices]
                x_axis = plot_data['datetime']
                
                # Add smoothing using rolling average
                window_size = max(1, len(plot_data) // 10)
                for col in ['temperature', 'salinity', 'pressure']:
                    if col in plot_data.columns:
                        plot_data[col] = plot_data[col].rolling(window=window_size, center=True, min_periods=1).mean()
        else:
            plot_data = self.data.reset_index()
            x_axis = plot_data.index
            x_title = "📊 Measurement Index"
        
        fig = go.Figure()
        
        # Determine what to show based on selected parameter or query context
        if query_context:
            query_lower = query_context.lower()
            if any(word in query_lower for word in ['temperature', 'temp', 'hot', 'cold', 'warm']):
                selected_param = 'temperature'
            elif any(word in query_lower for word in ['salinity', 'salt', 'salty']):
                selected_param = 'salinity'
            elif any(word in query_lower for word in ['pressure', 'depth', 'deep']):
                selected_param = 'pressure'
            elif any(word in query_lower for word in ['combined', 'complete']):
                selected_param = 'combined'
        
        # Create optimized charts based on selection
        if selected_param == 'temperature':
            # Enhanced temperature visualization
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['temperature'],
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.15)',
                line=dict(
                    color='rgba(239, 68, 68, 1)', 
                    width=3, 
                    shape='spline',
                    smoothing=1.0
                ),
                name='🌡️ Temperature (°C)',
                hovertemplate='<b>%{y:.1f}°C</b><br>%{x|%b %d, %Y}<extra></extra>',
                mode='lines'
            ))
            
            # Add gradient fill for better visual appeal
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['temperature'],
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.05)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            chart_title = "🌡️ Ocean Temperature Trends"
            y_title = "Temperature (°C)"
            
        elif selected_param == 'salinity':
            # Enhanced salinity visualization
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['salinity'],
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.15)',
                line=dict(
                    color='rgba(59, 130, 246, 1)', 
                    width=3, 
                    shape='spline',
                    smoothing=1.0
                ),
                name='🧂 Salinity (PSU)',
                hovertemplate='<b>%{y:.2f} PSU</b><br>%{x|%b %d, %Y}<extra></extra>',
                mode='lines'
            ))
            
            # Add gradient fill
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['salinity'],
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.05)',
                line=dict(color='rgba(0,0,0,0)', width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            chart_title = "🧂 Ocean Salinity Patterns"
            y_title = "Salinity (PSU)"
            
        elif selected_param == 'pressure':
            # Ultra-optimized depth visualization with dive profile styling
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['pressure'],
                fill='tozeroy',
                fillcolor='rgba(16, 185, 129, 0.2)',
                line=dict(
                    color='rgba(16, 185, 129, 1)', 
                    width=4, 
                    shape='spline',
                    smoothing=1.3  # Extra smoothing for depth
                ),
                name='🌊 Ocean Depth (dbar)',
                hovertemplate='<b>%{y:.0f} dbar deep</b><br>%{x|%b %d, %Y}<br><i>~%{customdata:.0f}m depth</i><extra></extra>',
                customdata=plot_data['pressure'],  # Approximate meters (1 dbar ≈ 1m)
                mode='lines'
            ))
            
            # Add depth zones visualization
            max_depth = plot_data['pressure'].max()
            
            # Surface zone (0-100m) - light blue
            fig.add_hrect(y0=0, y1=min(100, max_depth), 
                         fillcolor="rgba(147, 197, 253, 0.1)", 
                         layer="below", line_width=0)
            
            # Shallow zone (100-500m) - medium blue  
            if max_depth > 100:
                fig.add_hrect(y0=100, y1=min(500, max_depth), 
                             fillcolor="rgba(59, 130, 246, 0.1)", 
                             layer="below", line_width=0)
            
            # Deep zone (500-1000m) - dark blue
            if max_depth > 500:
                fig.add_hrect(y0=500, y1=min(1000, max_depth), 
                             fillcolor="rgba(30, 64, 175, 0.1)", 
                             layer="below", line_width=0)
            
            # Very deep zone (>1000m) - navy
            if max_depth > 1000:
                fig.add_hrect(y0=1000, y1=max_depth, 
                             fillcolor="rgba(15, 23, 42, 0.1)", 
                             layer="below", line_width=0)
            
            chart_title = "🌊 Ocean Depth Profile"
            y_title = "Depth (dbar)"
            
        else:  # combined
            # Optimized combined view with dual-axis
            temp_color = 'rgba(239, 68, 68, 1)'
            sal_color = 'rgba(59, 130, 246, 1)'
            
            # Temperature trace
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['temperature'],
                yaxis='y',
                fill='tozeroy',
                fillcolor='rgba(239, 68, 68, 0.1)',
                line=dict(color=temp_color, width=2.5, shape='spline', smoothing=1.0),
                name='🌡️ Temperature',
                hovertemplate='<b>Temp: %{y:.1f}°C</b><extra></extra>',
                mode='lines'
            ))
            
            # Salinity trace on secondary axis
            fig.add_trace(go.Scatter(
                x=x_axis,
                y=plot_data['salinity'],
                yaxis='y2',
                line=dict(color=sal_color, width=2.5, shape='spline', smoothing=1.0),
                name='🧂 Salinity',
                hovertemplate='<b>Salinity: %{y:.2f} PSU</b><extra></extra>',
                mode='lines'
            ))
            
            chart_title = "🌊 Temperature & Salinity Analysis"
            y_title = "Temperature (°C)"
        
        # Ultra-optimized layout for professional visualization
        layout_config = {
            'template': 'plotly_white',
            'height': 380,
            'margin': dict(l=70, r=70, t=70, b=60),
            'showlegend': selected_param == 'combined',  # Only show legend for combined view
            'title': {
                'text': chart_title,
                'x': 0.5,
                'font': {'size': 18, 'color': '#1f2937', 'family': 'Arial, sans-serif'}
            },
            'xaxis': dict(
                title=dict(text=x_title, font=dict(size=13, color='#6b7280')),
                showgrid=True, 
                gridwidth=1, 
                gridcolor='rgba(0,0,0,0.08)',
                color='#6b7280',
                tickfont=dict(size=11),
                showline=True,
                linecolor='rgba(0,0,0,0.1)',
                mirror=True
            ),
            'yaxis': dict(
                title=dict(text=y_title, font=dict(size=13, color='#6b7280')),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.08)',
                color='#6b7280',
                tickfont=dict(size=11),
                showline=True,
                linecolor='rgba(0,0,0,0.1)',
                mirror=True,
                zeroline=True,
                zerolinecolor='rgba(0,0,0,0.15)',
                zerolinewidth=1
            ),
            'plot_bgcolor': 'rgba(248, 250, 252, 0.3)',
            'paper_bgcolor': 'white',
            'font': dict(color='#374151', family='Arial, sans-serif'),
            'hovermode': 'x unified',
            'hoverlabel': dict(
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor='rgba(0, 0, 0, 0.1)',
                font=dict(size=12, color='#1f2937')
            )
        }
        
        # Add secondary y-axis for combined view
        if selected_param == 'combined':
            layout_config['yaxis2'] = dict(
                title=dict(text="Salinity (PSU)", font=dict(size=13, color='#3b82f6')),
                overlaying='y',
                side='right',
                showgrid=False,
                color='#3b82f6',
                tickfont=dict(size=11),
                showline=True,
                linecolor='rgba(59, 130, 246, 0.3)',
                mirror=False
            )
            layout_config['legend'] = dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.1)",
                borderwidth=1,
                font=dict(size=12)
            )
        
        fig.update_layout(**layout_config)
        
        return fig
    
    def create_pressure_donut(self):
        """Create pressure distribution donut chart."""
        if self.data is None or self.data.empty:
            return go.Figure()
        
        # Create pressure ranges
        pressure_ranges = pd.cut(self.data['pressure'], 
                               bins=[0, 100, 500, 1000, float('inf')], 
                               labels=['Shallow<br>(0-100)', 'Medium<br>(100-500)', 
                                      'Deep<br>(500-1000)', 'Very Deep<br>(>1000)'])
        
        range_counts = pressure_ranges.value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=range_counts.index,
            values=range_counts.values,
            hole=0.6,
            marker_colors=['#10b981', '#f59e0b', '#ef4444', '#8b5cf6'],
            textinfo='percent',
            textposition='outside',
            showlegend=False
        )])
        
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def get_level_summary(self, data=None):
        """Get summary of different depth levels in the data."""
        if data is None:
            data = self.data
        
        if data is None or data.empty:
            return {}
        
        levels = {
            'Surface (0-100m)': len(data[data['pressure'] <= 100]),
            'Shallow (100-500m)': len(data[(data['pressure'] > 100) & (data['pressure'] <= 500)]),
            'Deep (500-1000m)': len(data[(data['pressure'] > 500) & (data['pressure'] <= 1000)]),
            'Very Deep (>1000m)': len(data[data['pressure'] > 1000])
        }
        
        return levels
    
    def get_recent_activities(self):
        """Get recent data activities for the activity panel."""
        if self.data is None or self.data.empty:
            return []
        
        recent_data = self.data.tail(5)
        activities = []
        
        for _, row in recent_data.iterrows():
            temp_status = "High" if row['temperature'] > self.data['temperature'].median() else "Normal"
            sal_status = "High" if row['salinity'] > self.data['salinity'].median() else "Normal"
            
            activities.append({
                'time': row['datetime'].strftime('%H:%M') if pd.notna(row['datetime']) else 'Recent',
                'temp': f"{row['temperature']:.1f}°C",
                'sal': f"{row['salinity']:.1f} PSU",
                'pressure': f"{row['pressure']:.0f} dbar",
                'temp_status': temp_status,
                'sal_status': sal_status
            })
        
        return activities


    def get_detailed_sample_data(self, limit=15):
        """Get detailed sample data for the data table."""
        if self.data is None or self.data.empty:
            return []
        
        # Get sample data with all important columns
        sample_data = self.data.head(limit)
        detailed_data = []
        
        for idx, row in sample_data.iterrows():
            detailed_data.append({
                'profile_id': row.get('profile_id', f'Profile_{idx}'),
                'temperature': f"{row['temperature']:.2f}°C",
                'salinity': f"{row['salinity']:.2f} PSU", 
                'pressure': f"{row['pressure']:.1f} dbar",
                'depth_level': self.categorize_depth_level(row['pressure']),
                'temp_category': self.categorize_temp_level(row['temperature']),
                'datetime': row['datetime'].strftime('%Y-%m-%d %H:%M') if pd.notna(row.get('datetime')) else 'N/A'
            })
        
        return detailed_data
    
    def categorize_depth_level(self, pressure):
        """Categorize depth level for display."""
        if pressure <= 100:
            return "Surface"
        elif pressure <= 500:
            return "Shallow"
        elif pressure <= 1000:
            return "Deep"
        else:
            return "Very Deep"
    
    def categorize_temp_level(self, temperature):
        """Categorize temperature level for display."""
        if temperature < 10:
            return "Cold"
        elif temperature < 20:
            return "Moderate"
        else:
            return "Warm"
    
    def search_ocean_overview(self, query=""):
        """Search function specifically for ocean overview insights."""
        if self.data is None or self.data.empty:
            return "No data available"
        
        query = query.lower().strip()
        
        # Default overview if no query
        if not query:
            return self.get_default_overview()
        
        # Temperature queries
        if any(word in query for word in ['temperature', 'temp', 'hot', 'cold', 'warm']):
            temp_stats = self.data['temperature'].agg(['min', 'max', 'mean', 'std'])
            return f"🌡️ Temperature Overview: Range {temp_stats['min']:.1f}°C to {temp_stats['max']:.1f}°C, Average {temp_stats['mean']:.1f}°C, Variation ±{temp_stats['std']:.1f}°C. {len(self.data[self.data['temperature'] > 20])} warm locations, {len(self.data[self.data['temperature'] < 10])} cold locations."
        
        # Depth/Pressure queries
        elif any(word in query for word in ['depth', 'deep', 'pressure', 'shallow', 'surface']):
            pressure_stats = self.data['pressure'].agg(['min', 'max', 'mean'])
            levels = self.get_level_summary()
            return f"🌊 Depth Overview: Range {pressure_stats['min']:.0f} to {pressure_stats['max']:.0f} dbar, Average depth {pressure_stats['mean']:.0f} dbar. Distribution: {levels.get('Surface (0-100m)', 0)} surface, {levels.get('Shallow (100-500m)', 0)} shallow, {levels.get('Deep (500-1000m)', 0)} deep, {levels.get('Very Deep (>1000m)', 0)} very deep locations."
        
        # Salinity queries
        elif any(word in query for word in ['salinity', 'salt', 'salty']):
            sal_stats = self.data['salinity'].agg(['min', 'max', 'mean', 'std'])
            return f"🧂 Salinity Overview: Range {sal_stats['min']:.2f} to {sal_stats['max']:.2f} PSU, Average {sal_stats['mean']:.2f} PSU, Variation ±{sal_stats['std']:.2f} PSU. Ocean salinity shows {'high' if sal_stats['mean'] > 35 else 'normal'} salt content."
        
        # Data quality queries
        elif any(word in query for word in ['quality', 'data', 'records', 'measurements']):
            return f"📊 Data Quality Overview: {len(self.data)} total measurements analyzed. Temperature readings: {len(self.data.dropna(subset=['temperature']))} valid, Salinity readings: {len(self.data.dropna(subset=['salinity']))} valid, Pressure readings: {len(self.data.dropna(subset=['pressure']))} valid. Data completeness: {(len(self.data.dropna()) / len(self.data) * 100):.1f}%"
        
        # Trend queries
        elif any(word in query for word in ['trend', 'change', 'pattern']):
            temp_trend = self.data['temperature'].iloc[-10:].mean() - self.data['temperature'].iloc[:10].mean()
            sal_trend = self.data['salinity'].iloc[-10:].mean() - self.data['salinity'].iloc[:10].mean()
            return f"📈 Trend Overview: Temperature {'increasing' if temp_trend > 0 else 'decreasing'} by {abs(temp_trend):.2f}°C, Salinity {'increasing' if sal_trend > 0 else 'decreasing'} by {abs(sal_trend):.2f} PSU over the measurement period."
        
        # Range queries
        elif any(word in query for word in ['range', 'variation', 'difference']):
            temp_range = self.data['temperature'].max() - self.data['temperature'].min()
            sal_range = self.data['salinity'].max() - self.data['salinity'].min()
            pressure_range = self.data['pressure'].max() - self.data['pressure'].min()
            return f"📏 Range Overview: Temperature varies by {temp_range:.1f}°C, Salinity varies by {sal_range:.2f} PSU, Depth varies by {pressure_range:.0f} dbar across all measurement locations."
        
        else:
            return self.get_default_overview()
    
    def get_default_overview(self):
        """Get default ocean overview."""
        if self.data is None or self.data.empty:
            return "No data available"
        
        temp_avg = self.data['temperature'].mean()
        sal_avg = self.data['salinity'].mean()
        pressure_avg = self.data['pressure'].mean()
        levels = self.get_level_summary()
        
        return f"🌊 Ocean Overview: Analyzing {len(self.data)} measurements. Average conditions: {temp_avg:.1f}°C temperature, {sal_avg:.1f} PSU salinity at {pressure_avg:.0f} dbar average depth. Data spans {levels.get('Surface (0-100m)', 0)} surface + {levels.get('Shallow (100-500m)', 0)} shallow + {levels.get('Deep (500-1000m)', 0)} deep + {levels.get('Very Deep (>1000m)', 0)} very deep locations."


# Initialize app with modern theme
app = dash.Dash(__name__, 
               external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
               suppress_callback_exceptions=True)

app.title = "OceanCrystal Dashboard"

# Initialize processor
processor = OceanDashboardProcessor()
processor.load_data()

# Modern dashboard layout
app.layout = html.Div([
    # Top Navigation Bar with Search
    dbc.Navbar([
        dbc.Container([
            dbc.NavbarBrand([
                html.I(className="fas fa-water me-2", style={"color": "#3b82f6"}),
                "OceanCrystal"
            ], className="fw-bold", style={"fontSize": "1.4rem"}),
            
            # Search Section
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id="search-input",
                            placeholder="Search ocean data...",
                            style={"borderRadius": "20px 0 0 20px"}
                        ),
                        dbc.Button(
                            html.I(className="fas fa-search"),
                            id="search-btn",
                            color="primary",
                            style={"borderRadius": "0 20px 20px 0"}
                        )
                    ], size="sm")
                ], width=8),
                dbc.Col([
                    dbc.Select(
                        id="depth-filter",
                        options=[
                            {"label": "🌊 All Ocean Depths", "value": "all"},
                            {"label": "🏊 Surface (0-100m) - Swimming zone", "value": "surface"},
                            {"label": "🐠 Shallow (100-500m) - Colorful fish zone", "value": "shallow"},
                            {"label": "🐙 Deep (500-1000m) - Dark ocean zone", "value": "deep"},
                            {"label": "🦑 Very Deep (1000m+) - Deep sea creatures", "value": "very_deep"}
                        ],
                        value="all",
                        size="sm"
                    )
                ], width=4)
            ], className="g-2", align="center"),
            
            dbc.Nav([
                dbc.NavItem([
                    html.I(className="fas fa-user-circle", style={"fontSize": "1.8rem", "color": "#6b7280"})
                ])
            ], className="ms-auto")
        ], fluid=True)
    ], color="white", dark=False, className="shadow-sm mb-4"),
    
    # Main Dashboard Content
    dbc.Container([
        # Search Results Info Row
        dbc.Row([
            dbc.Col([
                dbc.Alert(
                    id="search-results-info",
                    children="Showing all ocean data",
                    color="info",
                    className="mb-3",
                    style={"borderLeft": "4px solid #3b82f6"}
                )
            ])
        ]),
        
        # Level Summary Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-swimmer me-2", style={"color": "#17a2b8"}),
                                "Surface Level"
                            ], className="mb-2"),
                            html.H4(id="surface-count", className="text-info mb-0"),
                            html.Small("0-100m depth", className="text-muted d-block"),
                            html.Small("🏊 Swimming & diving zone", className="text-info")
                        ])
                    ])
                ], className="border-0 shadow-sm text-center", style={"backgroundColor": "#e8f4fd"})
            ], md=3, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-fish me-2", style={"color": "#28a745"}),
                                "Shallow Level"
                            ], className="mb-2"),
                            html.H4(id="shallow-count", className="text-success mb-0"),
                            html.Small("100-500m depth", className="text-muted d-block"),
                            html.Small("🐠 Colorful fish zone", className="text-success")
                        ])
                    ])
                ], className="border-0 shadow-sm text-center", style={"backgroundColor": "#e8f5e8"})
            ], md=3, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-water me-2", style={"color": "#ffc107"}),
                                "Deep Level"
                            ], className="mb-2"),
                            html.H4(id="deep-count", className="text-warning mb-0"),
                            html.Small("500-1000m depth", className="text-muted d-block"),
                            html.Small("🐙 Dark ocean zone", className="text-warning")
                        ])
                    ])
                ], className="border-0 shadow-sm text-center", style={"backgroundColor": "#fff8e1"})
            ], md=3, className="mb-3"),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-water me-2", style={"color": "#dc3545"}),
                                "Very Deep Level"
                            ], className="mb-2"),
                            html.H4(id="very-deep-count", className="text-danger mb-0"),
                            html.Small(">1000m depth", className="text-muted d-block"),
                            html.Small("🦑 Deep sea creatures", className="text-danger")
                        ])
                    ])
                ], className="border-0 shadow-sm text-center", style={"backgroundColor": "#fde2e4"})
            ], md=3, className="mb-3"),
        ]),
        
        # Depth Explanation Card
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H6([
                                html.I(className="fas fa-info-circle me-2", style={"color": "#17a2b8"}),
                                "What do these depth levels mean?"
                            ], className="mb-3", style={"color": "#2c3e50"}),
                            
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Strong("🏊 Surface (0-100m):"),
                                        html.P("Like a swimming pool! This is where people swim, snorkel, and see beautiful fish. Sunlight makes it bright and warm.", 
                                              className="mb-2 small")
                                    ])
                                ], md=6),
                                dbc.Col([
                                    html.Div([
                                        html.Strong("🐠 Shallow (100-500m):"),
                                        html.P("Like a coral reef! Colorful fish live here. Some sunlight still reaches down, but it's getting darker.", 
                                              className="mb-2 small")
                                    ])
                                ], md=6),
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Strong("🐙 Deep (500-1000m):"),
                                        html.P("Like a dark cave! No sunlight reaches here. Water is cold and only special fish with big eyes can see.", 
                                              className="mb-2 small")
                                    ])
                                ], md=6),
                                dbc.Col([
                                    html.Div([
                                        html.Strong("🦑 Very Deep (1000m+):"),
                                        html.P("Like outer space underwater! Completely dark, very cold, high pressure. Only amazing deep-sea creatures live here!", 
                                              className="mb-2 small")
                                    ])
                                ], md=6),
                            ])
                        ])
                    ])
                ], className="border-0 shadow-sm", style={"backgroundColor": "#f8f9fa", "borderLeft": "4px solid #17a2b8"})
            ])
        ], className="mb-4"),
        
        # Top KPI Cards Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.H3(id="avg-temp", className="mb-0 fw-bold"),
                                html.P("Avg Temperature", className="text-muted mb-0 small"),
                            ]),
                            html.Div([
                                html.I(className="fas fa-thermometer-half", 
                                      style={"fontSize": "2rem", "color": "#ef4444"})
                            ], className="text-end")
                        ], className="d-flex justify-content-between align-items-center")
                    ])
                ], className="border-0 shadow-sm h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.H3(id="max-pressure", className="mb-0 fw-bold"),
                                html.P("Max Pressure", className="text-muted mb-0 small"),
                            ]),
                            html.Div([
                                html.I(className="fas fa-arrows-alt-v", 
                                      style={"fontSize": "2rem", "color": "#10b981"})
                            ], className="text-end")
                        ], className="d-flex justify-content-between align-items-center")
                    ])
                ], className="border-0 shadow-sm h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.H3(id="avg-salinity", className="mb-0 fw-bold"),
                                html.P("Avg Salinity", className="text-muted mb-0 small"),
                            ]),
                            html.Div([
                                html.I(className="fas fa-tint", 
                                      style={"fontSize": "2rem", "color": "#3b82f6"})
                            ], className="text-end")
                        ], className="d-flex justify-content-between align-items-center")
                    ])
                ], className="border-0 shadow-sm h-100")
            ], md=3, className="mb-3"),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.Div([
                                html.H3(id="total-records", className="mb-0 fw-bold"),
                                html.P("Data Points", className="text-muted mb-0 small"),
                            ]),
                            html.Div([
                                html.I(className="fas fa-database", 
                                      style={"fontSize": "2rem", "color": "#8b5cf6"})
                            ], className="text-end")
                        ], className="d-flex justify-content-between align-items-center")
                    ])
                ], className="border-0 shadow-sm h-100")
            ], md=3, className="mb-3"),
        ]),
        
        # Ocean Overview Query Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("🔍 Ocean Overview Query", className="mb-0"),
                            html.Small("Ask about temperature, depth, salinity, trends, quality, or range", className="text-muted")
                        ])
                    ]),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(
                                id="overview-query-input",
                                placeholder="e.g., 'temperature trends', 'depth analysis', 'data quality'...",
                                style={"borderRadius": "25px 0 0 25px"}
                            ),
                            dbc.Button(
                                "Analyze",
                                id="overview-query-btn",
                                color="primary",
                                style={"borderRadius": "0 25px 25px 0"}
                            )
                        ]),
                        html.Div(
                            id="overview-results",
                            className="mt-3",
                            style={"minHeight": "60px", "backgroundColor": "#f8f9fa", "padding": "15px", "borderRadius": "8px", "border": "1px solid #dee2e6"}
                        )
                    ])
                ], className="border-0 shadow-sm")
            ])
        ], className="mb-4"),
        
        # Main Content Row
        dbc.Row([
            # Left Column - Main Chart
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("Ocean Overview", className="mb-0"),
                            html.Div([
                                html.Label("Parameter:", className="me-2 text-muted", style={'fontSize': '12px'}),
                                dcc.Dropdown(
                                    id="chart-parameter-selector",
                                    options=[
                                        {'label': '🌡️ Temperature', 'value': 'temperature'},
                                        {'label': '🧂 Salinity', 'value': 'salinity'}, 
                                        {'label': '🌊 Depth', 'value': 'pressure'},
                                        {'label': '📊 Combined', 'value': 'combined'}
                                    ],
                                    value='temperature',
                                    style={'width': '130px', 'fontSize': '12px'},
                                    clearable=False
                                )
                            ], className="d-flex align-items-center")
                        ], className="d-flex justify-content-between align-items-center")
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="main-chart", config={'displayModeBar': False})
                    ], className="p-0")
                ], className="border-0 shadow-sm")
            ], md=8, className="mb-3"),
            
            # Right Column - Pressure Distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div([
                            html.H5("🌊 Ocean Depth Distribution", className="mb-0"),
                            html.Small("How many measurements at each depth level", className="text-muted")
                        ])
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="pressure-chart", config={'displayModeBar': False})
                    ], className="p-2")
                ], className="border-0 shadow-sm")
            ], md=4, className="mb-3"),
        ]),
        
        # Bottom Row - Detailed Data Table
        dbc.Row([
            # Detailed Sample Data
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("📊 Detailed Ocean Data Sample", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="detailed-data-table")
                    ])
                ], className="border-0 shadow-sm")
            ], md=12),
        ]),
        
    ], fluid=True),
    
    # Auto-refresh interval
    dcc.Interval(id='interval-component', interval=30*1000, n_intervals=0)
    
], style={"backgroundColor": "#f8fafc", "minHeight": "100vh"})


# Main callback to update all dashboard components
@app.callback(
    [Output("avg-temp", "children"),
     Output("max-pressure", "children"), 
     Output("avg-salinity", "children"),
     Output("total-records", "children"),
     Output("main-chart", "figure"),
     Output("pressure-chart", "figure"),
     Output("detailed-data-table", "children"),
     Output("search-results-info", "children"),
     Output("surface-count", "children"),
     Output("shallow-count", "children"),
     Output("deep-count", "children"),
     Output("very-deep-count", "children")],
    [Input('interval-component', 'n_intervals'),
     Input('search-btn', 'n_clicks'),
     Input('depth-filter', 'value'),
     Input('chart-parameter-selector', 'value')],
    [State('search-input', 'value')]
)
def update_dashboard(n, search_clicks, depth_level, selected_param, search_query):
    try:
        # Always store original data first
        original_data = processor.data.copy() if processor.data is not None else pd.DataFrame()
        
        # Determine what data to use
        if search_query and search_query.strip():
            # Text search provided
            filtered_data = processor.search_data(
                query_text=search_query.strip(),
                depth_level=depth_level if depth_level != "all" else "all"
            )
            processor.data = filtered_data
        elif depth_level and depth_level != "all":
            # Only depth filter applied
            filtered_data = processor.search_data(
                query_text="",
                depth_level=depth_level
            )
            processor.data = filtered_data
        else:
            # No filters - use all data
            filtered_data = original_data
        
        # Check if we have data
        if filtered_data is None or filtered_data.empty:
            return "N/A", "N/A", "N/A", "N/A", go.Figure(), go.Figure(), "No data available", "No data found", "0", "0", "0", "0"
        
        # Get KPI metrics (ensure processor.data is set correctly for metrics)
        if search_query or (depth_level != "all"):
            processor.data = filtered_data
        
        metrics = processor.get_kpi_metrics()
        
        if not metrics:
            return "N/A", "N/A", "N/A", "N/A", go.Figure(), go.Figure(), "No metrics", "No metrics found", "0", "0", "0", "0"
        
        # Get level summary
        level_summary = processor.get_level_summary(filtered_data)
        
        # KPI values
        avg_temp = f"{metrics['avg_temp']:.1f}°C"
        max_pressure = f"{metrics['max_pressure']:.0f} dbar"
        avg_salinity = f"{metrics['avg_salinity']:.1f} PSU"
        total_records = f"{metrics['total_records']:,}"
        
        # Search results info
        search_info = f"Showing {len(filtered_data)} results"
        if search_query:
            search_info += f" for '{search_query}'"
        if depth_level != "all":
            depth_names = {
                "surface": "Surface Level (0-100m)",
                "shallow": "Shallow Level (100-500m)", 
                "deep": "Deep Level (500-1000m)",
                "very_deep": "Very Deep Level (>1000m)"
            }
            search_info += f" in {depth_names.get(depth_level, depth_level)}"
        
        # Level counts
        surface_count = level_summary.get('Surface (0-100m)', 0)
        shallow_count = level_summary.get('Shallow (100-500m)', 0)
        deep_count = level_summary.get('Deep (500-1000m)', 0)
        very_deep_count = level_summary.get('Very Deep (>1000m)', 0)
        
        # Main chart (context-aware based on search query and dropdown selection)
        chart_context = search_query if search_query and search_query.strip() else ""
        main_fig = processor.create_main_chart(chart_type="overview", query_context=chart_context, selected_param=selected_param or 'temperature')
        
        # Pressure donut chart
        pressure_fig = processor.create_pressure_donut()
        
        # Detailed data table
        detailed_data = processor.get_detailed_sample_data(15)
        data_table = create_detailed_data_table(detailed_data)
        
        # Restore original data if it was temporarily changed
        if search_query or (depth_level != "all"):
            processor.data = original_data
        
        return (avg_temp, max_pressure, avg_salinity, total_records, main_fig, pressure_fig, 
                data_table, search_info, surface_count, shallow_count, deep_count, very_deep_count)
        
    except Exception as e:
        print(f"Dashboard error: {e}")
        import traceback
        traceback.print_exc()
        return "Error", "Error", "Error", "Error", go.Figure(), go.Figure(), "Error", "Error occurred", "0", "0", "0", "0"


# Ocean Overview Query Callback
@app.callback(
    Output("overview-results", "children"),
    [Input("overview-query-btn", "n_clicks"),
     Input("overview-query-input", "n_submit")],
    [State("overview-query-input", "value")]
)
def update_ocean_overview(btn_clicks, input_submit, query):
    """Update ocean overview based on user query."""
    if not btn_clicks and not input_submit:
        return html.Div([
            html.P("🌊 Enter a query above to get detailed ocean insights!", className="text-muted mb-0"),
            html.Small("Try: 'temperature analysis', 'depth distribution', 'salinity patterns', 'data trends'", className="text-info")
        ])
    
    if not query:
        query = ""
    
    try:
        overview_text = processor.search_ocean_overview(query)
        return html.Div([
            html.P(overview_text, className="mb-0", style={"fontSize": "14px", "lineHeight": "1.5"})
        ])
    except Exception as e:
        return html.Div([
            html.P(f"Error: {str(e)}", className="text-danger mb-0")
        ])


def create_detailed_data_table(detailed_data):
    """Create a detailed data table component."""
    if not detailed_data:
        return html.P("No data available", className="text-muted")
    
    # Create table header
    header = html.Thead([
        html.Tr([
            html.Th("Profile ID", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("DateTime", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("Temperature", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("Salinity", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("Pressure", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("Depth Level", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
            html.Th("Temp Category", style={"backgroundColor": "#f8f9fa", "fontSize": "12px", "padding": "8px"}),
        ])
    ])
    
    # Create table body
    rows = []
    for data in detailed_data:
        # Color coding for depth levels
        depth_color = {
            "Surface": "#e3f2fd",
            "Shallow": "#e8f5e8", 
            "Deep": "#fff3e0",
            "Very Deep": "#fce4ec"
        }.get(data['depth_level'], "#ffffff")
        
        rows.append(html.Tr([
            html.Td(data['profile_id'], style={"fontSize": "11px", "padding": "6px"}),
            html.Td(data['datetime'], style={"fontSize": "11px", "padding": "6px"}),
            html.Td(data['temperature'], style={"fontSize": "11px", "padding": "6px", "fontWeight": "bold", "color": "#d32f2f"}),
            html.Td(data['salinity'], style={"fontSize": "11px", "padding": "6px", "fontWeight": "bold", "color": "#1976d2"}),
            html.Td(data['pressure'], style={"fontSize": "11px", "padding": "6px", "fontWeight": "bold", "color": "#388e3c"}),
            html.Td([
                dbc.Badge(data['depth_level'], color="secondary", className="w-100")
            ], style={"fontSize": "11px", "padding": "6px", "backgroundColor": depth_color}),
            html.Td([
                dbc.Badge(data['temp_category'], 
                         color="danger" if data['temp_category'] == "Warm" else ("warning" if data['temp_category'] == "Moderate" else "info"),
                         className="w-100")
            ], style={"fontSize": "11px", "padding": "6px"}),
        ], style={"backgroundColor": depth_color if depth_color != "#ffffff" else "white"}))
    
    body = html.Tbody(rows)
    
    return html.Div([
        html.Div([
            html.P(f"📊 Showing {len(detailed_data)} detailed ocean measurements", 
                   className="mb-2", style={"fontSize": "14px", "fontWeight": "bold"})
        ]),
        dbc.Table([header, body], 
                  striped=True, bordered=True, hover=True, size="sm",
                  style={"fontSize": "11px"})
    ])


# Additional callback for search input enter key
@app.callback(
    Output('search-btn', 'n_clicks'),
    [Input('search-input', 'n_submit')],
    [State('search-btn', 'n_clicks')]
)
def search_on_enter(n_submit, current_clicks):
    """Trigger search when user presses Enter in search box."""
    if n_submit:
        return (current_clicks or 0) + 1
    return current_clicks or 0


# Ocean Overview Chart Update Callback
@app.callback(
    Output("main-chart", "figure", allow_duplicate=True),
    [Input("overview-query-btn", "n_clicks"),
     Input("overview-query-input", "n_submit"),
     Input("chart-parameter-selector", "value")],
    [State("overview-query-input", "value")],
    prevent_initial_call=True
)
def update_overview_chart(btn_clicks, input_submit, selected_param, query):
    """Update main chart based on ocean overview query and parameter selection."""
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # If dropdown changed, use that parameter
    if trigger_id == "chart-parameter-selector":
        chart_context = ""
        param = selected_param or 'temperature'
    # If query was made, extract parameter from query
    elif (trigger_id in ["overview-query-btn", "overview-query-input"]) and query:
        chart_context = query.strip()
        param = selected_param or 'temperature'
    else:
        return dash.no_update
    
    try:
        main_fig = processor.create_main_chart(chart_type="overview", query_context=chart_context, selected_param=param)
        return main_fig
    except Exception as e:
        print(f"Overview chart error: {e}")
        return dash.no_update


if __name__ == "__main__":
    print("🌊 OceanCrystal Dashboard - Modern Analytics Interface")
    print("📊 Financial-style dashboard for ocean data")
    print("🚀 Starting on http://localhost:8050")
    
    app.run(debug=True, port=8050)