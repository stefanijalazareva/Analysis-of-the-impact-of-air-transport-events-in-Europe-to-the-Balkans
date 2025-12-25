"""
Time Series Builder for Air Transport Delays

This module creates regularly-sampled hourly time series of delays for each airport.
It is designed specifically for delay-based network and causal analysis.

Key principles:
- Hourly resampling on a complete continuous time grid
- Missing values represent hours with NO flights
- Missing values are filled with 0 (no flights = no delay)
- An explicit activity mask is generated to distinguish inactivity from true zero delay

Output files:
- hourly_delays.csv: Hourly mean delays (minutes), gaps filled with 0
- hourly_activity.csv: Binary activity indicator (1 = flights occurred, 0 = no flights)
- timeseries_quality.json: Data quality metrics, validation results, and imputation statistics
"""

import pandas as pd
import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.data_loader import DataLoader


def create_hourly_timeseries(df, output_dir):
    """
    Create hourly time series of delays for each airport.

    This function performs the following steps:
    1. Ensures timestamp column exists
    2. Converts delays from seconds to minutes
    3. Creates a pivot table with airports as columns
    4. Resamples the data to hourly frequency
    5. Creates an activity indicator (1 = flights exist, 0 = no flights)
    6. Fills missing values with 0 (no flights = no delay)
    7. Tracks imputation statistics
    8. Saves the processed data and generates a quality report

    Note: Missing values are filled with 0, interpreting absence of data as no flights.
    This is appropriate because:
    - No flight record = No flight occurred = No delay to measure
    - Prevents artificial delay values during night hours or low-traffic periods
    - Network analysis can properly distinguish between "no activity" vs "activity with delays"

    Args:
        df (pandas.DataFrame): Input DataFrame with delay data
        output_dir (pathlib.Path): Directory to save output files

    Returns:
        tuple: (pivot_delay, pivot_activity) - Hourly delays and activity indicator DataFrames
    """
    print("Creating hourly time series...")

    if 'timestamp' not in df.columns:
        if 'sched_ts' in df.columns:
            print("Converting sched_ts to timestamp...")
            df['timestamp'] = pd.to_datetime(df['sched_ts'], unit='s')
        else:
            raise ValueError("DataFrame must have either 'timestamp' or 'sched_ts' column")
    
    if 'delay_min' not in df.columns:
        if 'delay_s' in df.columns:
            df['delay_min'] = df['delay_s'] / 60.0
        else:
            raise ValueError("DataFrame must have either 'delay_min' or 'delay_s' column")

    pivot_delay = (
        df.pivot_table(
            index='timestamp',
            columns='arr',
            values='delay_min',
            aggfunc='mean'
        )
        .resample('1h')
        .mean()
    )

    pivot_activity = (
        df.pivot_table(
            index='timestamp',
            columns='arr',
            values='delay_min',
            aggfunc='count'
        )
        .resample('1h')
        .sum()
        .fillna(0)
    )
    pivot_activity = (pivot_activity > 0).astype(int)

    missing_before = pivot_delay.isnull().sum().to_dict()
    total_hours = len(pivot_delay)
    total_cells = total_hours * len(pivot_delay.columns)
    
    print(f"Total hours in time series: {total_hours}")
    print(f"Missing values before imputation: {sum(missing_before.values())} "
          f"({100 * sum(missing_before.values()) / total_cells:.1f}%)")
    print("Interpretation: Missing values represent hours with no flights (e.g., night hours, low-traffic periods)")
    print("Strategy: Filling with 0 (no flights = no delay)")

    pivot_delay = pivot_delay.fillna(0)

    missing_after = pivot_delay.isnull().sum().to_dict()
    if sum(missing_after.values()) > 0:
        print(f"WARNING: {sum(missing_after.values())} missing values remain after imputation!")
    else:
        print("All missing values filled with 0")
    
    try:
        pivot_delay.to_csv(output_dir / 'hourly_delays.csv')
        pivot_activity.to_csv(output_dir / 'hourly_activity.csv')
    except Exception as e:
        print(f"Error saving CSV files: {e}")
        raise

    imputation_stats = {}
    for airport in pivot_delay.columns:
        original_count = total_hours - missing_before.get(airport, 0)
        imputed_count = missing_before.get(airport, 0)
        active_hours = pivot_activity[airport].sum()
        imputation_stats[airport] = {
            'original_values': int(original_count),
            'imputed_values': int(imputed_count),
            'imputation_percentage': round(100 * imputed_count / total_hours, 2) if total_hours > 0 else 0,
            'active_hours': int(active_hours),
            'activity_percentage': round(100 * active_hours / total_hours, 2) if total_hours > 0 else 0
        }

    quality_report = {
        'start_time': pivot_delay.index.min().isoformat(),
        'end_time': pivot_delay.index.max().isoformat(),
        'total_hours': int(total_hours),
        'airports': pivot_delay.columns.tolist(),
        'imputation_method': 'zero_fill (no flights = no delay)',
        'interpretation': 'Missing values represent hours with no scheduled flights',
        'missing_values_before_imputation': {k: int(v) for k, v in missing_before.items()},
        'missing_values_after_imputation': {k: int(v) for k, v in missing_after.items()},
        'imputation_statistics': imputation_stats,
        'mean_delays_minutes': {k: round(v, 4) for k, v in pivot_delay.mean().to_dict().items()},
        'std_delays_minutes': {k: round(v, 4) for k, v in pivot_delay.std().to_dict().items()},
        'notes_for_network_analysis': {
            'delay_file': 'hourly_delays.csv - Use for Granger causality, transfer entropy',
            'activity_file': 'hourly_activity.csv - Use to filter analysis to active hours only',
            'recommendation': 'Consider conditional analysis: test causality only when both airports are active'
        }
    }

    try:
        with open(output_dir / 'timeseries_quality.json', 'w') as f:
            json.dump(quality_report, f, indent=2)
    except Exception as e:
        print(f"Error saving quality report: {e}")
        raise

    return pivot_delay, pivot_activity


def validate_timeseries(ts):
    """
    Validate the time series data for compatibility with network analysis.

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
    
    freq = pd.infer_freq(ts.index)
    print(f"Frequency: {freq if freq else 'Could not infer (irregular spacing)'}")
    
    print("\nMissing values per airport:")
    missing = ts.isnull().sum()
    if missing.sum() == 0:
        print("No missing values found (all gaps filled)")
    else:
        print(missing[missing > 0])

    time_diffs = ts.index.to_series().diff()
    gaps = time_diffs[time_diffs > pd.Timedelta('1h')]
    if not gaps.empty:
        print("\nFound gaps in time series:")
        print(gaps)
    else:
        print("\nNo gaps in time index")


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
        tuple: (delays, activity) - The processed hourly time series DataFrames
    """
    print("Loading processed delay data...")
    df = DataLoader().load_processed_data()
    
    print(f"\nLoaded data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Date range: {df['sched_ts'].min()} to {df['sched_ts'].max()}" if 'sched_ts' in df.columns else "")
    
    output_dir = Path("data/TimeSeries")
    output_dir.mkdir(parents=True, exist_ok=True)

    delays, activity = create_hourly_timeseries(df, output_dir)
    validate_timeseries(delays)

    print(f"\nTime series data saved to {output_dir / 'hourly_delays.csv'}")
    print(f"Activity data saved to {output_dir / 'hourly_activity.csv'}")
    print(f"Quality report saved to {output_dir / 'timeseries_quality.json'}")

    return delays, activity


if __name__ == "__main__":
    delays, activity = main()

    print("SUMMARY")
    print("\nTop 10 airports by mean delay (minutes):")
    print(delays.mean().sort_values(ascending=False).head(10))
    
    print("\nAirport activity summary (% of hours with flights):")
    activity_pct = (activity.sum() / len(activity) * 100).sort_values(ascending=False)
    print(activity_pct.head(10))
