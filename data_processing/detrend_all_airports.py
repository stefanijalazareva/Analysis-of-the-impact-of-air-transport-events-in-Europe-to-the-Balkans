"""
Detrend all airports from the TimeSeries data.
Processes hourly_delays.csv and saves results for each airport.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import logging

try:
    from .detrending import detrend_and_save
    from .detrend_timeseries import Detrender
except ImportError:
    from detrending import detrend_and_save
    from detrend_timeseries import Detrender

logging.basicConfig(
    format="%(asctime)s | %(levelname)8s | %(message)s",
    level=logging.INFO,
)


def detrend_all_airports(
    input_file: str = "data/TimeSeries/hourly_delays.csv",
    output_base_dir: str = "data/Detrended/airports",
    methods: List[str] = None,
    save_csv: bool = True,
    save_summary: bool = True
):
    """
    Detrend time series for all airports.
    
    :param input_file: Path to CSV file with time series data
    :param output_base_dir: Base directory for saving results
    :param methods: List of detrending methods to apply (default: ['z_score', 'delta'])
    :param save_csv: Whether to save detrended series as CSV
    :param save_summary: Whether to save summary statistics
    """
    if methods is None:
        methods = ['z_score', 'delta']
    
    logging.info(f"Loading time series from {input_file}...")
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    airports = df.columns.tolist()
    logging.info(f"Found {len(airports)} airports: {airports}")
    logging.info(f"Time series length: {len(df)} observations")
    logging.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    total_airports = len(airports)
    total_methods = len(methods)
    total_tasks = total_airports * total_methods
    current_task = 0
    
    results_summary = {
        'airports': [],
        'methods': [],
        'status': [],
        'mean_original': [],
        'std_original': [],
        'mean_detrended': [],
        'std_detrended': []
    }
    
    for airport in airports:
        logging.info(f"\n{'='*70}")
        logging.info(f"Processing airport: {airport}")
        
        ts_data = df[airport].values
        
        nan_count = np.isnan(ts_data).sum()
        if nan_count > 0:
            logging.warning(f"  {airport} has {nan_count} NaN values - skipping")
            for method in methods:
                results_summary['airports'].append(airport)
                results_summary['methods'].append(method)
                results_summary['status'].append('SKIPPED_NAN')
                results_summary['mean_original'].append(np.nan)
                results_summary['std_original'].append(np.nan)
                results_summary['mean_detrended'].append(np.nan)
                results_summary['std_detrended'].append(np.nan)
            continue
        
        logging.info(f"  Original - Mean: {np.mean(ts_data):.2f}, Std: {np.std(ts_data):.2f}")
        
        for method in methods:
            current_task += 1
            progress = (current_task / total_tasks) * 100
            
            logging.info(f"  [{current_task}/{total_tasks} - {progress:.1f}%] Applying {method}...")
            
            try:
                output_dir = f"{output_base_dir}/{airport}/{method}"
                ts_detrended = detrend_and_save(
                    ts_data,
                    method=method,
                    output_dir=output_dir,
                    save_csv=save_csv,
                    save_summary=save_summary
                )
                
                mean_det = np.mean(ts_detrended)
                std_det = np.std(ts_detrended)
                
                logging.info(f"Success - Mean: {mean_det:.2f}, Std: {std_det:.2f}")
                
                results_summary['airports'].append(airport)
                results_summary['methods'].append(method)
                results_summary['status'].append('SUCCESS')
                results_summary['mean_original'].append(float(np.mean(ts_data)))
                results_summary['std_original'].append(float(np.std(ts_data)))
                results_summary['mean_detrended'].append(float(mean_det))
                results_summary['std_detrended'].append(float(std_det))
                
            except Exception as e:
                logging.error(f"Error: {e}")
                
                results_summary['airports'].append(airport)
                results_summary['methods'].append(method)
                results_summary['status'].append('ERROR')
                results_summary['mean_original'].append(float(np.mean(ts_data)))
                results_summary['std_original'].append(float(np.std(ts_data)))
                results_summary['mean_detrended'].append(np.nan)
                results_summary['std_detrended'].append(np.nan)
    

    logging.info("Saving overall summary...")
    
    summary_df = pd.DataFrame(results_summary)
    summary_path = Path(output_base_dir) / "detrending_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    logging.info(f"Summary saved to: {summary_path}")
    
    success_count = (summary_df['status'] == 'SUCCESS').sum()
    error_count = (summary_df['status'] == 'ERROR').sum()
    skipped_count = (summary_df['status'] == 'SKIPPED_NAN').sum()
    
    logging.info(f"\n{'='*70}")
    logging.info("FINAL STATISTICS:")
    logging.info(f"  Total tasks: {total_tasks}")
    logging.info(f"  Successful: {success_count}")
    logging.info(f"  Errors: {error_count}")
    logging.info(f"  Skipped (NaN): {skipped_count}")
    logging.info(f"  Success rate: {(success_count/total_tasks)*100:.1f}%")
    
    return summary_df


if __name__ == "__main__":
    print("Detrending all airports from TimeSeries data...\n")
    
    Detrender.list_methods()
    
    summary = detrend_all_airports(
        input_file="data/TimeSeries/hourly_delays.csv",
        output_base_dir="data/Detrended/airports",
        methods=['z_score', 'delta', 'second_difference'],
        save_csv=True,
        save_summary=True
    )
    
    print("Check data/Detrended/airports/ for results.")
    print("\nSummary preview:")
    print(summary.head(10))
