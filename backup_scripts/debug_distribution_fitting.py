"""
Debugged version of the distribution fitting script to identify where it's failing.
"""
print("Starting script...")

import sys
print("Python version:", sys.version)

try:
    print("Importing libraries...")
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from scipy.optimize import dual_annealing, minimize
    from scipy.stats import kstest, ks_2samp
    from pathlib import Path
    import logging
    import time
    from tqdm import tqdm
    import warnings
    import argparse
    import json
    import datetime
    print("All libraries imported successfully")
except Exception as e:
    print(f"Error importing libraries: {e}")
    sys.exit(1)

print("Setting up logging...")
try:
    # Configure logging to console only initially
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    print("Logging configured")
except Exception as e:
    print(f"Error setting up logging: {e}")
    sys.exit(1)

print("Checking data paths...")
try:
    data_dir = Path("data/ProcessedData")
    output_dir = Path("data/DistributionFitting")

    # Check if input directory exists
    if data_dir.exists():
        print(f"Input directory exists: {data_dir}")
    else:
        print(f"Input directory does not exist: {data_dir}")

    # Check if output directory exists, create if not
    if not output_dir.exists():
        print(f"Creating output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Output directory exists: {output_dir}")

    # Check for the input data file
    input_file = data_dir / 'cleaned_delays.parquet'
    if input_file.exists():
        print(f"Input file exists: {input_file}")
    else:
        print(f"Input file does not exist: {input_file}")
except Exception as e:
    print(f"Error checking data paths: {e}")
    sys.exit(1)

print("Trying to load a small sample of data...")
try:
    df = pd.read_parquet(input_file, engine='pyarrow')
    print(f"Successfully loaded data with shape: {df.shape}")
    print(f"Sample of data columns: {df.columns.tolist()[:5]}")
    print(f"First few rows of delay_min column: {df['delay_min'].head().tolist()}")
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

print("Creating a test output file...")
try:
    with open(output_dir / 'debug_test.json', 'w') as f:
        json.dump({"test": "successful", "timestamp": str(datetime.datetime.now())}, f)
    print("Successfully wrote test output file")
except Exception as e:
    print(f"Error writing test output file: {e}")
    sys.exit(1)

print("Debug script completed successfully")
