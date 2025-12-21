"""
Detrend Time Series for Air Transport Delays

This script processes the prepared delay time series data and applies
various detrending methods to make the data stationary by removing
daily patterns and other trends.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
import warnings
import os
import traceback

sys.path.insert(0, str(Path(__file__).parent.parent))
import delaynet as dn

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timeseries_analysis.log'),
        logging.StreamHandler()
    ]
)

def load_data():
    """
    Load the filled hourly time series data prepared for network analysis.

    Returns:
        pandas.DataFrame: DataFrame containing the hourly delay time series (all gaps filled)
    """
    data_path = Path("data/TimeSeries/hourly_delays.csv")
    logging.info(f"Loading filled hourly time series from {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run 'python -m data_processing.build_timeseries' first to create the filled time series."
        )

    df = pd.read_csv(data_path, index_col=0)
    
    # Convert index to datetime (handle timezone-aware strings from CSV)
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)

    logging.info(f"Loaded data shape: {df.shape}")
    logging.info(f"Data timespan: {df.index.min()} to {df.index.max()}")
    logging.info(f"Airports (columns): {df.columns.tolist()}")
    
    missing = df.isnull().sum().sum()
    if missing > 0:
        logging.warning(f"Found {missing} missing values in filled time series!")
    else:
        logging.info("No missing values - data is ready for network analysis")

    return df

def prepare_data_for_detrending(df):
    """
    Prepare filled hourly time series data for detrending.
    
    The data should already be in the correct format from build_timeseries.py:
    - Columns are airports
    - Rows are hourly timestamps
    - All missing values filled

    Args:
        df: DataFrame with hourly delay time series (airports as columns)

    Returns:
        numpy.ndarray: Array with shape (n_nodes, n_times) where nodes are airports
        list: Airport codes (column names corresponding to nodes)
    """
    nodes = df.columns.tolist()
    ts_array = df.values.T

    logging.info(f"Prepared array shape: {ts_array.shape} ({len(nodes)} airports × {ts_array.shape[1]} time steps)")

    nan_count = np.isnan(ts_array).sum()
    if nan_count > 0:
        logging.warning(f"Array contains {nan_count} NaN values ({np.isnan(ts_array).mean() * 100:.2f}%)")
        logging.warning("This should not happen - time series should be fully filled!")
    else:
        logging.info("Array contains no NaN values - ready for detrending")

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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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

        window_size = 24 * 7
        if original.shape[1] > window_size:
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

        n_methods = len(detrended_dict) + 1  # +1 for original
        fig, axes = plt.subplots(n_methods, 1, figsize=(12, 3 * n_methods), sharex=True)

        if n_methods == 1:
            axes = [axes]

        valid_mask = ~np.isnan(original[sample_idx, start_idx:end_idx])
        time_indices = np.arange(start_idx, end_idx)[valid_mask]
        valid_data = original[sample_idx, start_idx:end_idx][valid_mask]

        axes[0].plot(time_indices, valid_data, 'b-')
        axes[0].set_title(f"Original - {sample_node}")
        axes[0].set_ylabel("Delay")

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

        fig, axes = plt.subplots(1, n_methods, figsize=(4 * n_methods, 4))

        if n_methods == 1:
            axes = [axes]

        valid_data = original[sample_idx, :][~np.isnan(original[sample_idx, :])]
        if len(valid_data) > 0:
            sns.histplot(valid_data, kde=True, ax=axes[0])
            axes[0].set_title(f"Original Distribution\n{sample_node}")

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

        orig_stationary, orig_pvals = dn.check_stationarity(original)

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
                stationary_results[method] = np.zeros_like(orig_stationary, dtype=bool)
                pvalue_results[method] = np.ones_like(orig_pvals)

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

        stationary_percentages = {
            method: np.mean(results) * 100 for method, results in stationary_results.items()
        }

        logging.info("Stationarity test results:")
        for method, percentage in stationary_percentages.items():
            logging.info(f"  {method}: {percentage:.1f}% stationary")

        plt.figure(figsize=(10, 6))
        plt.bar(stationary_percentages.keys(), stationary_percentages.values())
        plt.title('Percentage of Stationary Time Series by Detrending Method')
        plt.ylabel('Percentage of Stationary Series (%)')
        plt.ylim([0, 100])
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        for i, (method, percentage) in enumerate(stationary_percentages.items()):
            plt.text(i, percentage + 2, f"{percentage:.1f}%", ha='center')

        plt.tight_layout()
        plt.savefig(output_dir / "stationarity_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        summary_df.to_csv(output_dir / "stationarity_summary.csv", index=False)

        return summary_df

    except Exception as e:
        logging.error(f"Error during stationarity analysis: {str(e)}")
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
            df = pd.DataFrame(data.T, columns=nodes, index=index[:data.shape[1]])

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

        df = load_data()

        ts_array, nodes = prepare_data_for_detrending(df)

        original_index = df.index

        methods = ['delta', 'delta2', 'zs', 'linear']
        logging.info(f"Applying detrending methods: {', '.join(methods)}")
        logging.info("  - delta: First-order differencing (local mean subtraction)")
        logging.info("  - delta2: Second-order differencing")
        logging.info("  - zs: Z-score normalization with daily periodicity (24 hours)")
        logging.info("  - linear: Linear detrending using scipy.signal.detrend")

        detrended_results = dn.compare_detrending_methods(
            ts_array,
            methods=methods,
            periodicity=24,
            axis=1
        )

        output_dir = Path("data/DetrendedData")
        output_dir.mkdir(parents=True, exist_ok=True)

        visualize_detrending_comparison(ts_array, detrended_results, nodes, output_dir)

        stationarity_summary = check_and_visualize_stationarity(
            ts_array, detrended_results, nodes, output_dir
        )

        save_detrended_data(detrended_results, nodes, original_index, output_dir)

        logging.info("Detrending process completed successfully!")
        logging.info(f"Results saved to {os.path.abspath(output_dir)}")

    except Exception as e:
        logging.error(f"Error during detrending: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
