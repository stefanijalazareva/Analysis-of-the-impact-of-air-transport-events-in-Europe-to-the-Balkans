"""
Time Series Builder for Air Transport Delays

This module creates regularly-sampled time series of delays for each airport,
which is required for network analysis and connectivity metrics.

The module performs the following operations:
- Resampling to hourly frequency
- Handling of missing values through forward/backward filling and interpolation
- Time alignment across all airports
- Data validation and quality checks

Output files:
- hourly_delays.csv: Contains the hourly resampled delay data
- timeseries_quality.json: Contains data quality metrics and validation results
"""

import pandas as pd
import json
from pathlib import Path
from load_dataframe import load_cleaned_data

def create_hourly_timeseries(df, output_dir):
    """
    Create hourly time series of delays for each airport.

    This function performs the following steps:
    1. Creates a pivot table with airports as columns
    2. Resamples the data to hourly frequency
    3. Handles missing values through forward/backward filling and interpolation
    4. Saves the processed data and generates a quality report

    Args:
        df (pandas.DataFrame): Input DataFrame with delay data
        output_dir (pathlib.Path): Directory to save output files

    Returns:
        pandas.DataFrame: Processed hourly time series with airports as columns
    """
    print("Creating hourly time series...")

    pivot = (df.pivot_table(index='sched_dt',
                           columns='arr',
                           values='delay_min',
                           aggfunc='mean')
            .resample('1h').mean())

    # Handle missing values with a 24-hour window for each method
    pivot = (pivot
             .ffill(limit=24)
             .bfill(limit=24)
             .interpolate(method='time', limit=24))

    pivot.to_csv(output_dir / 'hourly_delays.csv')

    quality_report = {
        'start_time': pivot.index.min().isoformat(),
        'end_time': pivot.index.max().isoformat(),
        'total_hours': len(pivot),
        'missing_values_before_interpolation': pivot.isnull().sum().to_dict(),
        'airports': pivot.columns.tolist(),
        'mean_delays': pivot.mean().to_dict(),
        'std_delays': pivot.std().to_dict()
    }

    with open(output_dir / 'timeseries_quality.json', 'w') as f:
        json.dump(quality_report, f, indent=2)

    return pivot

def validate_timeseries(ts):
    """
    Validate the time series data for compatibility with delaynet analysis.

    Performs checks for:
    - Time range coverage
    - Sampling frequency consistency
    - Missing values
    - Time index gaps

    Args:
        ts (pandas.DataFrame): Time series data to validate
    """
    print("\nValidating time series...")
    print(f"Time range: {ts.index.min()} to {ts.index.max()}")
    print(f"Frequency: {pd.infer_freq(ts.index)}")
    print("\nMissing values per airport:")
    print(ts.isnull().sum())

    # Check for gaps in time index
    time_diffs = ts.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta('1h')]
    if not gaps.empty:
        print("\nWarning: Found gaps in time series:")
        print(gaps)

def main():
    """
    Main execution function that orchestrates the time series creation process.

    The function performs the following steps:
    1. Loads the pre-processed delay data
    2. Creates and configures the output directory
    3. Generates hourly time series from the data
    4. Validates the generated time series
    5. Saves results to disk

    Returns:
        pandas.DataFrame: The processed hourly time series with airports as columns
    """
    print("Loading cleaned delay data...")
    df = load_cleaned_data()

    output_dir = Path("data/TimeSeries")
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = create_hourly_timeseries(df, output_dir)
    validate_timeseries(ts)

    print(f"\nTime series data saved to {output_dir / 'hourly_delays.csv'}")
    print(f"Quality report saved to {output_dir / 'timeseries_quality.json'}")

    return ts


if __name__ == "__main__":
    """
    Script execution entry point.
    
    When run as a script, this will:
    1. Execute the main processing pipeline
    2. Display summary statistics for the generated time series
    """
    ts = main()

    print("\nSummary Statistics:")
    print("-------------------")
    print("\nMean delays by airport:")
    print(ts.mean().sort_values(ascending=False))
