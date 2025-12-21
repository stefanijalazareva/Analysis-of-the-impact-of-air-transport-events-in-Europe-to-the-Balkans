"""
Time Series Builder for Air Transport Delays

This module creates regularly-sampled time series of delays for each airport,
which is required for network analysis and connectivity metrics.

The module performs the following operations:
- Resampling to hourly frequency
- Handling of ALL missing values through forward/backward filling and time-based interpolation
- Time alignment across all airports (creates complete continuous time grid)
- Data validation and quality checks
- Imputation tracking to distinguish original vs. filled values

Output files:
- hourly_delays.csv: Contains the hourly resampled delay data (all gaps filled)
- timeseries_quality.json: Contains data quality metrics, validation results, and imputation statistics
"""

import pandas as pd
import json
from pathlib import Path
from data_processing.data_loader import DataLoader

def create_hourly_timeseries(df, output_dir):
    """
    Create hourly time series of delays for each airport.

    This function performs the following steps:
    1. Creates a pivot table with airports as columns
    2. Resamples the data to hourly frequency
    3. Handles ALL missing values through forward/backward filling and time-based interpolation
    4. Tracks imputation statistics (original vs filled values)
    5. Saves the processed data and generates a quality report

    Note: All gaps are filled to ensure compatibility with network analysis methods
    that require complete continuous time series (Granger causality, transfer entropy, etc.)

    Args:
        df (pandas.DataFrame): Input DataFrame with delay data
        output_dir (pathlib.Path): Directory to save output files

    Returns:
        pandas.DataFrame: Processed hourly time series with airports as columns (all gaps filled)
    """
    print("Creating hourly time series...")

    pivot = (df.pivot_table(index='sched_dt',
                           columns='arr',
                           values='delay_min',
                           aggfunc='mean')
            .resample('1h').mean())

    missing_before = pivot.isnull().sum().to_dict()
    total_hours = len(pivot)
    
    print(f"Total hours in time series: {total_hours}")
    print(f"Missing values before imputation: {sum(missing_before.values())} ({100*sum(missing_before.values())/(total_hours*len(pivot.columns)):.1f}%)")

    pivot = (pivot
             .ffill()
             .bfill()
             .interpolate(method='time'))

    missing_after = pivot.isnull().sum().to_dict()
    if sum(missing_after.values()) > 0:
        print(f"WARNING: {sum(missing_after.values())} missing values remain after imputation!")
    else:
        print("All missing values successfully imputed")
    
    pivot.to_csv(output_dir / 'hourly_delays.csv')

    imputation_stats = {}
    for airport in pivot.columns:
        original_count = total_hours - missing_before.get(airport, 0)
        imputed_count = missing_before.get(airport, 0)
        imputation_stats[airport] = {
            'original_values': int(original_count),
            'imputed_values': int(imputed_count),
            'imputation_percentage': round(100 * imputed_count / total_hours, 2) if total_hours > 0 else 0
        }

    quality_report = {
        'start_time': pivot.index.min().isoformat(),
        'end_time': pivot.index.max().isoformat(),
        'total_hours': int(total_hours),
        'airports': pivot.columns.tolist(),
        'imputation_method': 'forward_fill -> backward_fill -> time_interpolation (all gaps filled)',
        'missing_values_before_imputation': {k: int(v) for k, v in missing_before.items()},
        'missing_values_after_imputation': {k: int(v) for k, v in missing_after.items()},
        'imputation_statistics': imputation_stats,
        'mean_delays': {k: round(v, 4) for k, v in pivot.mean().to_dict().items()},
        'std_delays': {k: round(v, 4) for k, v in pivot.std().to_dict().items()}
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
    df = DataLoader().load_processed_data()  

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
    print("\nMean delays by airport:")
    print(ts.mean().sort_values(ascending=False))
