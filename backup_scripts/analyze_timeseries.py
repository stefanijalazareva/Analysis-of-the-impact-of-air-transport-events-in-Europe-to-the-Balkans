"""
Time Series Analysis of Air Transport Delays

This module provides detailed analysis of the prepared hourly delay time series,
focusing on temporal patterns, correlations, and delay propagation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import calendar
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timeseries_analysis.log'),
        logging.StreamHandler()
    ]
)

def load_timeseries():
    """
    Load and validate the prepared hourly time series data.

    Returns:
        pandas.DataFrame: Hourly delay time series for all airports
    """
    data_path = Path("data/ProcessedData/cleaned_delays.parquet")
    logging.info(f"Loading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df['sched_dt'] = pd.to_datetime(df['sched_dt'])
        df = df.set_index('sched_dt')

    logging.info(f"Loaded data shape: {df.shape}")
    logging.info(f"Index type: {type(df.index)}")
    logging.info(f"Date range: {df.index.min()} to {df.index.max()}")

    return df

def analyze_temporal_patterns(df):
    """
    Analyze temporal patterns in delays across different time scales.

    Args:
        df: DataFrame with datetime index and airport delay columns

    Returns:
        dict: Collection of temporal pattern analyses
    """
    output_dir = Path("data/Analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Processing temporal patterns...")

    # First, ensure we only use numeric columns for analysis
    numeric_df = df.apply(pd.to_numeric, errors='coerce')

    """
    Convert all columns to numeric, coercing errors to NaN.
    This ensures that any non-numeric values will be replaced with NaN,
    allowing us to perform mathematical operations on the data.
    """

    # Drop columns that are entirely NaN after conversion
    numeric_df = numeric_df.dropna(axis=1, how='all')

    """
    Remove any columns that are completely non-numeric,
    as they can't be used for temporal pattern analysis.
    """

    # Log what happened during conversion
    original_cols = df.shape[1]
    numeric_cols = numeric_df.shape[1]
    logging.info(f"Converted columns from {original_cols} to {numeric_cols} numeric columns")

    if numeric_df.empty:
        logging.error("No numeric columns found in the dataset for temporal analysis.")
        return {}

    """
    Extract time components from the datetime index to analyze
    patterns by hour of day, day of week, and month of year.
    """
    patterns = {
        'hour': numeric_df.index.hour,
        'day_of_week': numeric_df.index.dayofweek,
        'month': numeric_df.index.month
    }

    """
    Calculate patterns with explicit numeric_only parameter to ensure
    we're only using numeric data in our calculations.
    """
    results = {}
    for time_unit, values in patterns.items():
        df_temp = numeric_df.copy()
        df_temp[time_unit] = values
        # Use numeric_only=True to avoid issues with non-numeric data
        results[time_unit] = df_temp.groupby(time_unit).mean(numeric_only=True)

        if time_unit == 'day_of_week':
            results[time_unit].index = [calendar.day_name[x] for x in results[time_unit].index]
        elif time_unit == 'month':
            results[time_unit].index = [calendar.month_name[x] for x in results[time_unit].index if x > 0]

    """
    Generate visualizations using default matplotlib style
    instead of seaborn to avoid compatibility issues.
    """
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    """
    Create hourly plot showing average delay by hour of day.
    This can help identify times of day with higher delays.
    """
    results['hour'].mean(axis=1).plot(ax=axes[0], marker='o')
    axes[0].set_title('Average Delay by Hour')
    axes[0].set_xlabel('Hour of Day')
    axes[0].set_ylabel('Mean Delay (minutes)')

    """
    Create daily plot showing average delay by day of week.
    This helps identify which days typically have higher delays.
    """
    results['day_of_week'].mean(axis=1).plot(kind='bar', ax=axes[1])
    axes[1].set_title('Average Delay by Day')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
    axes[1].set_ylabel('Mean Delay (minutes)')

    """
    Create monthly plot showing average delay by month of year.
    This helps identify seasonal patterns in flight delays.
    """
    results['month'].mean(axis=1).plot(kind='bar', ax=axes[2])
    axes[2].set_title('Average Delay by Month')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45)
    axes[2].set_ylabel('Mean Delay (minutes)')

    plt.tight_layout()
    fig.savefig(output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

    """
    Save numerical results to CSV files for further analysis.
    """
    for name, data in results.items():
        data.mean(axis=1).to_csv(output_dir / f'{name}_patterns.csv')

    logging.info("Temporal analysis complete!")
    return results

def analyze_correlations(df):
    """
    Analyze delay correlations between airports.

    Args:
        df: DataFrame with airport delay columns

    Returns:
        pandas.DataFrame: Sorted correlations between airport pairs
    """
    logging.info("Analyzing airport correlations...")
    output_dir = Path("data/Analysis")

    """
    Ensure we use only numeric data for correlation analysis by
    converting all columns to numeric values and handling errors.
    """
    numeric_df = pd.DataFrame()

    """
    Explicitly convert each column to numeric if possible,
    skipping columns that cannot be converted.
    """
    for col in df.columns:
        try:
            numeric_df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            logging.warning(f"Could not convert column '{col}' to numeric. Skipping.")

    """
    Drop columns that contain only NaN values after conversion,
    as they can't be used for correlation analysis.
    """
    numeric_df = numeric_df.dropna(axis=1, how='all')

    logging.info(f"Using {numeric_df.shape[1]} numeric columns for correlation analysis")

    if numeric_df.empty:
        logging.error("No numeric columns found in the dataset for correlation analysis.")
        return pd.DataFrame()

    """
    Calculate the correlation matrix to identify relationships
    between different airports' delay patterns.
    """
    corr_matrix = numeric_df.corr()

    """
    Generate and save a heatmap visualization of the correlations
    to help identify strongly correlated airport pairs.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
    plt.title('Delay Correlations Between Airports')
    plt.tight_layout()
    plt.savefig(output_dir / 'delay_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()

    """
    Create a list of airport pairs with their correlation values
    for easier analysis of the most correlated pairs.
    """
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            pairs.append({
                'airport1': corr_matrix.columns[i],
                'airport2': corr_matrix.columns[j],
                'correlation': corr_matrix.iloc[i,j]
            })

    """
    Sort the correlation data to identify the most strongly
    correlated airport pairs for further analysis.
    """
    correlations = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
    correlations.to_csv(output_dir / 'airport_correlations.csv', index=False)

    return correlations

def main():
    """
    Execute the main analysis pipeline.

    This function coordinates the overall workflow:
    1. Load the time series data
    2. Analyze temporal patterns
    3. Analyze correlations between airports
    4. Log the most significant results

    Any exceptions are caught, logged, and re-raised for proper handling.
    """
    try:
        # Load data
        df = load_timeseries()

        # Analyze temporal patterns
        patterns = analyze_temporal_patterns(df)

        # Analyze correlations
        correlations = analyze_correlations(df)
        logging.info("\nTop 5 most correlated airport pairs:")
        logging.info("\n" + str(correlations.head()))

        logging.info("Analysis completed successfully!")

    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
