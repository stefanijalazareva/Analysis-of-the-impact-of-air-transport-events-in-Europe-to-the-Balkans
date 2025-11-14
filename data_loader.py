"""
Air Transport Delay Data Loading and Processing Module
"""

import os
import glob
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path

class DataLoader:
    def __init__(self):
        # Use absolute paths based on the script location
        self.base_path = Path("C:/Stefanija/MANU/Statistical and information theory analysis of the impact of air transport events in Europe to the Balkans/AirTransportEvents")
        self.raw_data_path = self.base_path / "data" / "RawData"
        self.processed_data_path = self.base_path / "data" / "ProcessedData"

        print(f"Initialized DataLoader with:")
        print(f"Base path: {self.base_path}")
        print(f"Raw data path: {self.raw_data_path}")
        print(f"Processed data path: {self.processed_data_path}")

    def get_available_airports(self) -> List[str]:
        """List all available airport codes from raw data files."""
        if not self.raw_data_path.exists():
            raise ValueError(f"Raw data directory not found at: {self.raw_data_path}")

        files = list(self.raw_data_path.glob("Delays_*.npy"))
        if not files:
            raise ValueError(f"No delay data files found in: {self.raw_data_path}")

        return [f.stem[7:] for f in files]  # Remove 'Delays_' prefix

    def load_raw_delays(self, airport_code: str) -> np.ndarray:
        """Load raw delay data for a specific airport."""
        filepath = self.raw_data_path / f'Delays_{airport_code}.npy'
        if not filepath.exists():
            raise FileNotFoundError(f"No data file found for airport {airport_code}")
        return np.load(filepath, allow_pickle=True)

    def process_delay_file(self, filepath: Path) -> pd.DataFrame:
        """Process a single delay data file into a standardized DataFrame."""
        print(f"Processing file: {filepath}")
        arr = np.load(filepath, allow_pickle=True)
        rows = [list(r) for r in arr]
        df = pd.DataFrame(rows, columns=["dep", "arr", "sched_ts", "delay_s"])

        # Convert and validate data types
        df["sched_ts"] = pd.to_numeric(df["sched_ts"], errors="coerce")
        df["delay_s"] = pd.to_numeric(df["delay_s"], errors="coerce")

        # Add computed columns
        df["timestamp"] = pd.to_datetime(df["sched_ts"], unit='s')

        return df

    def load_processed_data(self) -> pd.DataFrame:
        """Load processed data from parquet or create if not exists."""
        print("Loading data...")
        parquet_path = self.processed_data_path / "cleaned_delays.parquet"

        if parquet_path.exists():
            print(f"Loading cached data from: {parquet_path}")
            return pd.read_parquet(parquet_path)

        print("Processing raw data files...")
        airports = self.get_available_airports()
        print(f"Found airports: {airports}")

        dfs = []
        for airport in airports:
            try:
                filepath = self.raw_data_path / f'Delays_{airport}.npy'
                df = self.process_delay_file(filepath)
                dfs.append(df)
                print(f"Processed {airport}")
            except Exception as e:
                print(f"Error processing {airport}: {e}")
                continue

        if not dfs:
            raise ValueError("No valid data files could be processed")

        combined_df = pd.concat(dfs, ignore_index=True)

        # Save processed data
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        combined_df.to_parquet(parquet_path, index=False)

        return combined_df

    def get_data_summary(self) -> Dict:
        """
        Generate summary statistics for the dataset.

        Returns:
            dict: Summary statistics including counts, missing values, etc.
        """
        df = self.load_processed_data()
        summary = {
            "total_records": len(df),
            "airports": df["airport_code"].unique().tolist(),
            "date_range": {
                "start": df["timestamp"].min().strftime("%Y-%m-%d"),
                "end": df["timestamp"].max().strftime("%Y-%m-%d")
            },
            "missing_values": df.isnull().sum().to_dict(),
            "delay_stats": {
                "mean": df["delay_s"].mean(),
                "median": df["delay_s"].median(),
                "std": df["delay_s"].std()
            }
        }

        # Save summary
        summary_path = self.processed_data_path / "data_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        return summary

# Example usage
if __name__ == "__main__":
    loader = DataLoader()
    print("Available airports:", loader.get_available_airports())

    # Load data for specific airports
    df = loader.load_processed_data(["LWSK", "LBSF"])
    print("\nData shape:", df.shape)
    print("\nSample data:")
    print(df.head())

    # Get and display summary
    summary = loader.get_data_summary()
    print("\nData summary:", json.dumps(summary, indent=2))
