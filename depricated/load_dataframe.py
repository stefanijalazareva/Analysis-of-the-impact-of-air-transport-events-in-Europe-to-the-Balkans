"""
Air Transport Delay Data Processing Module

This module handles the loading, processing, and cleaning of air transport delay data.
It processes raw .npy files containing flight delay information for various airports
and converts them into a structured pandas DataFrame with proper data types and
timezone information.

The module provides functions to:
- Load and process raw delay data files
- Clean and standardize the data
- Save processed data for future use
- Load pre-processed data for analysis

Output files:
- cleaned_delays.parquet: Contains the full processed dataset
- data_summary.json: Contains summary statistics and data quality metrics
"""

import os
import glob
import numpy as np
import pandas as pd
import json

def get_raw_data_path():
    """
    Get the path to the directory containing raw delay data files.

    Returns:
        str: Absolute path to the RawData directory
    """
    return os.path.join(os.getcwd(), "data", "RawData")

def process_delay_file(filepath):
    """
    Process a single delay data file and convert it to a pandas DataFrame.

    Args:
        filepath (str): Path to the .npy file containing delay data

    Returns:
        pandas.DataFrame: Processed DataFrame with standardized columns and data types
    """
    arr = np.load(filepath, allow_pickle=True)
    rows = [list(r) for r in arr]

    df = pd.DataFrame(rows, columns=["dep", "arr", "sched_ts", "delay_s"])

    df["sched_ts"] = pd.to_numeric(df["sched_ts"], errors="coerce").astype("float64")
    df["delay_s"] = pd.to_numeric(df["delay_s"], errors="coerce").astype("float64")
    df["sched_dt_utc"] = pd.to_datetime(df["sched_ts"], unit="s", utc=True)
    df["sched_dt"] = df["sched_dt_utc"].dt.tz_convert("Europe/Skopje")
    df["delay_min"] = df["delay_s"] / 60.0
    df["file_arr"] = os.path.basename(filepath).split("_")[-1].split(".")[0]

    return df

def load_cleaned_data():
    """
    Load the pre-processed and cleaned delay dataset.

    Returns:
        pandas.DataFrame: The cleaned delay dataset with all proper data types
                        and timezone information
    """
    output_dir = os.path.join(os.getcwd(), "data", "ProcessedData")
    return pd.read_parquet(os.path.join(output_dir, "cleaned_delays.parquet"))

# Main processing logic
data_path = get_raw_data_path()
files = glob.glob(os.path.join(data_path, "Delays_*.npy"))
print(f"Found {len(files)} delay data files")

# Process all files
dfs = []
for f in files:
    df = process_delay_file(f)
    print(f"Processed {os.path.basename(f)}, shape: {df.shape}")
    dfs.append(df)

# Combine all data
all_df = pd.concat(dfs, ignore_index=True)

# Quality checks and statistics
print("\nData Quality Checks:")
print("-------------------")
print(f"Total shape: {all_df.shape}")
print("\nMissing values:")
print(all_df.isnull().sum())

mismatches = all_df[all_df["arr"] != all_df["file_arr"]]
print("\nVerifying destinations match filename:")
print(f"Number of mismatches: {mismatches.shape[0]}")

print("\nDate range:")
print(f"Start: {all_df['sched_dt'].min()}")
print(f"End: {all_df['sched_dt'].max()}")

print("\nDelay statistics (minutes):")
print(all_df["delay_min"].describe())

# Save processed data
output_dir = os.path.join(os.getcwd(), "data", "ProcessedData")
os.makedirs(output_dir, exist_ok=True)

cleaned_data_path = os.path.join(output_dir, "cleaned_delays.parquet")
all_df.to_parquet(cleaned_data_path, compression='gzip')
print(f"\nSaved cleaned dataset to: {cleaned_data_path}")

# Save summary statistics
summary_stats = {
    'total_flights': all_df.shape[0],
    'date_range': {
        'start': all_df['sched_dt'].min().isoformat(),
        'end': all_df['sched_dt'].max().isoformat()
    },
    'delays_summary': all_df["delay_min"].describe().to_dict(),
    'flights_per_airport': all_df.groupby('file_arr').size().to_dict(),
    'missing_values': all_df.isnull().sum().to_dict(),
    'destination_mismatches': int(mismatches.shape[0])
}

summary_path = os.path.join(output_dir, "data_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"Saved summary statistics to: {summary_path}")

if __name__ == "__main__":
    print("\nTo use the cleaned data in other scripts, import the load_cleaned_data function:")
    print("from load_dataframe import load_cleaned_data")
    print("df = load_cleaned_data()")
