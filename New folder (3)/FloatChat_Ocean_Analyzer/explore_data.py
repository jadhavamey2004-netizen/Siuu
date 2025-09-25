"""
Data explorer script to examine the Parquet file structure
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Path to the data
data_path = r"C:\Users\jadha\Downloads\drive-download-20250924T143019Z-1-001\2900533"

# Find parquet files
parquet_files = list(Path(data_path).glob("*.parquet"))

print("📊 Exploring Ocean Data Files")
print("=" * 50)

for i, file_path in enumerate(parquet_files):
    print(f"\n📁 File {i+1}: {file_path.name}")
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Check for time-related columns
        time_cols = [col for col in df.columns if 'juld' in col.lower() or 'time' in col.lower() or 'date' in col.lower()]
        if time_cols:
            print(f"   Time columns: {time_cols}")
            
        # Check for ocean variables
        ocean_vars = [col for col in df.columns if any(var in col.lower() for var in ['psal', 'temp', 'pres'])]
        if ocean_vars:
            print(f"   Ocean variables: {ocean_vars}")
            
        # Show first few rows
        print(f"   Sample data:")
        print(df.head(3).to_string())
        
        print("-" * 30)
        
    except Exception as e:
        print(f"   Error reading file: {e}")

print("\n✅ Data exploration complete!")