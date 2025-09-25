# FloatChat Ocean Analyzer

A specialized time-series visualization tool for ocean float data with JULD (Julian Date) conversion and quarterly time scaling.

## Features

- **JULD Conversion**: Automatically converts Julian Date format to readable timestamps
- **Time-Series Visualization**: Interactive plots for Temperature, Salinity, and Pressure over time
- **Quarterly Analysis**: Built-in quarterly time scaling (Q1-Q4) and custom month ranges
- **Natural Language Search**: Query data with simple text like "high temperature" or "recent data"
- **Two-Panel Layout**: Search controls on the left, visualizations on the right
- **Real-time Filtering**: Filter data by time periods and years

## Installation

1. Install required packages:
```bash
pip install dash plotly pandas dash-bootstrap-components pyarrow numpy
```

2. Place your ocean data files in the expected directory structure:
```
C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533\
├── full_profile_data.parquet
├── measurements.parquet
└── trajectory.parquet
```

## Usage

1. Run the application:
```bash
python app.py
```

2. Open your browser to `http://localhost:8050`

3. Use the search panel to:
   - Enter natural language queries
   - Select quarterly or custom time ranges
   - Filter by specific years
   - View interactive time series plots

## Data Format

The application expects Parquet files with:
- **JULD**: Julian Date timestamps (days since 1950-01-01)
- **PRES/TEMP/PSAL**: Pressure, Temperature, Salinity measurements
- **Adjusted values**: Uses adjusted measurements when available

## Architecture

- `app.py`: Main Dash application with UI layout
- `src/data_processor.py`: Core data processing and JULD conversion
- `explore_data.py`: Data structure analysis utility

## Search Queries

Try these example queries:
- "high temperature"
- "low salinity" 
- "deep water"
- "recent measurements"
- "pressure data"

The system will automatically filter and visualize relevant data based on your query.