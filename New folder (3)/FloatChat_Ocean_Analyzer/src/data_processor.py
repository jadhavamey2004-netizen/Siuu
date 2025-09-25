"""
Data processing module for FloatChat Ocean Analyzer
Handles JULD conversion and data preparation
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
        
        Args:
            start_month: Starting month (1-12)
            end_month: Ending month (1-12)
            years: List of years to include, or None for all years
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
        """
        Get data for a specific quarter.
        
        Args:
            quarter: Quarter (1, 2, 3, or 4)
            years: List of years to include, or None for all years
        """
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
        
        # Create subplots with secondary y-axis for pressure
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
        
        # Pressure plot (inverted y-axis to show depth-like behavior)
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
        """
        Process search queries and return relevant data subset.
        
        Args:
            query: Natural language search query
            filtered_data: Data to search within (optional)
        """
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
            # Get most recent 100 records
            result_data = filtered_data.nlargest(100, 'datetime')
            response_text = f"Showing {len(result_data)} most recent measurements"
        
        # Default: return summary
        else:
            result_data = filtered_data.head(50)  # Limit to 50 records
            response_text = f"Showing overview of {len(result_data)} records from the dataset"
        
        # Limit result size for performance
        if len(result_data) > 200:
            result_data = result_data.head(200)
            response_text += f" (limited to 200 records for display)"
        
        return result_data, response_text