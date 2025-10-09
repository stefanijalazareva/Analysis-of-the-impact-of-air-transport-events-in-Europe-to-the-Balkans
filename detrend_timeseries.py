"""
Detrend Time Series for Air Transport Delays

This script processes the prepared delay time series data and applies
various detrending methods to make the data stationary by removing
daily patterns and other trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import seaborn as sns
import warnings
import os
import traceback

# Import our delaynet module
import delaynet as dn

"""
Suppress warnings to make output cleaner
"""
warnings.filterwarnings('ignore')

"""
Configure logging with both file and console output
"""
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timeseries_analysis.log'),  # Use existing log file
        logging.StreamHandler()
    ]
)

def load_data():
    """
    Load the processed delay time series data.

    Returns:
        pandas.DataFrame: DataFrame containing the delay time series
    """
    data_path = Path("data/ProcessedData/cleaned_delays.parquet")
    logging.info(f"Loading data from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    # Ensure the index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'sched_dt' in df.columns:
            df['sched_dt'] = pd.to_datetime(df['sched_dt'])
            df = df.set_index('sched_dt')
        else:
            # If no datetime column is found, use the existing index
            logging.warning("No datetime column found. Using existing index.")

    logging.info(f"Loaded data shape: {df.shape}")
    logging.info(f"Data timespan: {df.index.min()} to {df.index.max()}")
    logging.info(f"Columns: {df.columns.tolist()}")

    return df

def prepare_data_for_detrending(df):
    """
    Prepare data for detrending by selecting only numeric columns.

    Args:
        df: DataFrame with delay time series

    Returns:
        numpy.ndarray: Array with shape (n_nodes, n_times)
        list: Column names corresponding to nodes
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    logging.info(f"Found {len(numeric_cols)} numeric columns: {numeric_cols[:5]}..." if len(numeric_cols) > 5 else numeric_cols)

    if not numeric_cols:
        logging.warning("No numeric columns found. Attempting to convert columns.")
        numeric_df = df.apply(pd.to_numeric, errors='coerce')
        numeric_cols = numeric_df.columns[~numeric_df.isna().all()].tolist()
    else:
        numeric_df = df[numeric_cols]

    # Drop columns with all NaN values
    numeric_df = numeric_df.dropna(axis=1, how='all')

    # Get column names (nodes) and transpose data to nodes × times format
    nodes = numeric_df.columns.tolist()
    ts_array = numeric_df.values.T  # Transpose to get (n_nodes, n_times)

    logging.info(f"Prepared array shape: {ts_array.shape}")

    # Handle NaN values in the array
    nan_percentage = np.isnan(ts_array).mean() * 100
    logging.info(f"Array contains {nan_percentage:.2f}% NaN values")

    return ts_array, nodes

def visualize_detrending_comparison(original, detrended_dict, nodes, output_dir):
    """
    Visualize the effect of different detrending methods on a sample of time series.

    Args:
        original: Original time series array (n_nodes, n_times)
        detrended_dict: Dict with detrended arrays for different methods
        nodes: List of node names corresponding to array rows
        output_dir: Directory to save visualization outputs
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select a sample node to visualize (use the first node with non-NaN data)
        sample_idx = 0
        found_valid = False

        for i in range(len(nodes)):
            if not np.isnan(original[i]).all():
                sample_idx = i
                found_valid = True
                break

        if not found_valid:
            logging.warning("No valid node found with non-NaN data. Using first node for visualization.")

        sample_node = nodes[sample_idx]
        logging.info(f"Visualizing detrending methods for sample node: {sample_node}")

        # Select a sample time window (24 * 7 hours = 1 week)
        window_size = 24 * 7
        if original.shape[1] > window_size:
            # Find a window with minimal NaN values
            best_start = 0
            min_nans = float('inf')

            for i in range(0, original.shape[1] - window_size, window_size):
                n_nans = np.isnan(original[sample_idx, i:i+window_size]).sum()
                if n_nans < min_nans:
                    min_nans = n_nans
                    best_start = i

            start_idx = best_start
            end_idx = start_idx + window_size
        else:
            start_idx = 0
            end_idx = original.shape[1]

        # Create a figure with subplots for original and each detrending method
        n_methods = len(detrended_dict) + 1  # +1 for original
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods), sharex=True)

        # Handle case with only one subplot
        if n_methods == 1:
            axes = [axes]

        # Plot original data
        valid_mask = ~np.isnan(original[sample_idx, start_idx:end_idx])
        time_indices = np.arange(start_idx, end_idx)[valid_mask]
        valid_data = original[sample_idx, start_idx:end_idx][valid_mask]

        axes[0].plot(time_indices, valid_data, 'b-')
        axes[0].set_title(f"Original - {sample_node}")
        axes[0].set_ylabel("Delay")

        # Plot each detrending method
        for i, (method, data) in enumerate(detrended_dict.items(), start=1):
            valid_mask = ~np.isnan(data[sample_idx, start_idx:end_idx])
            time_indices = np.arange(start_idx, end_idx)[valid_mask]
            valid_data = data[sample_idx, start_idx:end_idx][valid_mask]

            axes[i].plot(time_indices, valid_data, 'g-')
            axes[i].set_title(f"Detrended ({method}) - {sample_node}")
            axes[i].set_ylabel("Detrended Value")

        axes[-1].set_xlabel("Time Steps")
        plt.tight_layout()
        plt.savefig(output_dir / f"detrending_comparison_{sample_node}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create distribution plots to compare detrending methods
        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))

        # Handle case with only one subplot
        if n_methods == 1:
            axes = [axes]

        # Original distribution - filter out NaNs
        valid_data = original[sample_idx, :][~np.isnan(original[sample_idx, :])]
        if len(valid_data) > 0:
            sns.histplot(valid_data, kde=True, ax=axes[0])
            axes[0].set_title(f"Original Distribution\n{sample_node}")

        # Detrended distributions
        for i, (method, data) in enumerate(detrended_dict.items(), start=1):
            valid_data = data[sample_idx, :][~np.isnan(data[sample_idx, :])]
            if len(valid_data) > 0:
                sns.histplot(valid_data, kde=True, ax=axes[i])
                axes[i].set_title(f"Detrended ({method})\n{sample_node}")

        plt.tight_layout()
        plt.savefig(output_dir / f"detrending_distribution_{sample_node}.png", dpi=300, bbox_inches='tight')
        plt.close()

        logging.info("Visualization completed successfully")

    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")
        # Continue execution even if visualization fails

def check_and_visualize_stationarity(original, detrended_dict, nodes, output_dir):
    """
    Check stationarity for original and detrended time series.

    Args:
        original: Original time series array (n_nodes, n_times)
        detrended_dict: Dict with detrended arrays for different methods
        nodes: List of node names corresponding to array rows
        output_dir: Directory to save visualization outputs
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Checking stationarity of time series...")

        # Check stationarity for original data
        orig_stationary, orig_pvals = dn.check_stationarity(original)

        # Check stationarity for each detrending method
        stationary_results = {
            'original': orig_stationary,
        }

        pvalue_results = {
            'original': orig_pvals,
        }

        for method, data in detrended_dict.items():
            try:
                stat, pval = dn.check_stationarity(data)
                stationary_results[method] = stat
                pvalue_results[method] = pval
            except Exception as e:
                logging.error(f"Error checking stationarity for {method}: {str(e)}")
                # Create placeholder arrays with all False and p-value of 1.0
                stationary_results[method] = np.zeros_like(orig_stationary, dtype=bool)
                pvalue_results[method] = np.ones_like(orig_pvals)

        # Create summary DataFrame
        summary_data = {
            'node': nodes,
            'original_stationary': orig_stationary,
            'original_pvalue': orig_pvals
        }

        for method, stationary in stationary_results.items():
            if method != 'original':
                summary_data[f'{method}_stationary'] = stationary_results[method]
                summary_data[f'{method}_pvalue'] = pvalue_results[method]

        summary_df = pd.DataFrame(summary_data)

        # Calculate percentage of stationary series for each method
        stationary_percentages = {
            method: np.mean(results) * 100 for method, results in stationary_results.items()
        }

        logging.info("Stationarity test results:")
        for method, percentage in stationary_percentages.items():
            logging.info(f"  {method}: {percentage:.1f}% stationary")

        # Create visualization of stationarity improvement
        plt.figure(figsize=(10, 6))
        plt.bar(stationary_percentages.keys(), stationary_percentages.values())
        plt.title('Percentage of Stationary Time Series by Detrending Method')
        plt.ylabel('Percentage of Stationary Series (%)')
        plt.ylim([0, 100])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add percentage labels on bars
        for i, (method, percentage) in enumerate(stationary_percentages.items()):
            plt.text(i, percentage + 2, f"{percentage:.1f}%", ha='center')

        plt.tight_layout()
        plt.savefig(output_dir / "stationarity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Save summary to CSV
        summary_df.to_csv(output_dir / "stationarity_summary.csv", index=False)

        return summary_df

    except Exception as e:
        logging.error(f"Error during stationarity analysis: {str(e)}")
        # Continue execution even if stationarity check fails
        return pd.DataFrame({'error': ['Stationarity check failed']})

def save_detrended_data(detrended_dict, nodes, index, output_dir):
    """
    Save the detrended data for further analysis.

    Args:
        detrended_dict: Dict with detrended arrays for different methods
        nodes: List of node names corresponding to array rows
        index: DatetimeIndex for the original data
        output_dir: Directory to save the detrended data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for method, data in detrended_dict.items():
        try:
            # Create DataFrame from detrended array
            df = pd.DataFrame(data.T, columns=nodes, index=index[:data.shape[1]])

            # Save as parquet file
            output_file = output_dir / f"detrended_{method}.parquet"
            df.to_parquet(output_file)
            logging.info(f"Saved detrended data ({method}) to {output_file}")
        except Exception as e:
            logging.error(f"Error saving {method} detrended data: {str(e)}")

def main():
    """
    Main function to execute the detrending workflow.
    """
    try:
        logging.info("Starting detrending process...")

        # Load delay time series data
        df = load_data()

        # Prepare data for detrending
        ts_array, nodes = prepare_data_for_detrending(df)

        # Save the original index for later use
        original_index = df.index

        # Apply different detrending methods
        methods = ['delta', 'delta2', 'zs', 'linear']
        logging.info(f"Applying detrending methods: {', '.join(methods)}")

        # Apply detrending methods with comparison
        detrended_results = dn.compare_detrending_methods(
            ts_array,
            methods=methods,
            periodicity=24,  # Assuming hourly data with daily patterns
            axis=1  # Detrend along time axis
        )

        # Create output directory
        output_dir = Path("data/DetrendedData")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Visualize the results
        visualize_detrending_comparison(ts_array, detrended_results, nodes, output_dir)

        # Check stationarity
        stationarity_summary = check_and_visualize_stationarity(
            ts_array, detrended_results, nodes, output_dir
        )

        # Save detrended data for further analysis
        save_detrended_data(detrended_results, nodes, original_index, output_dir)

        logging.info("Detrending process completed successfully!")
        logging.info(f"Results saved to {os.path.abspath(output_dir)}")

    except Exception as e:
        logging.error(f"Error during detrending: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
