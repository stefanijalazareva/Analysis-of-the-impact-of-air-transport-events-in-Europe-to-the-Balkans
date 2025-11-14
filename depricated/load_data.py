"""
This module handles loading and initial processing of air transport delay data.
It provides functionality to access raw delay data stored in numpy files for various airports.
"""

import os
import numpy as np

def get_raw_data_path():
    """Returns the path to the raw data directory containing delay data files."""
    return os.path.join(os.getcwd(), "data", "RawData")

outdir = get_raw_data_path()

print("Available raw data files (showing first 10):", os.listdir(outdir)[:10])

def load_airport_delays(airport_code="LWSK"):
    """
    Loads delay data for a specific airport.

    Args:
        airport_code (str): ICAO code of the airport (default: LWSK for Skopje)

    Returns:
        numpy.ndarray: Array containing delay data for the specified airport
    """
    return np.load(os.path.join(outdir, f'Delays_{airport_code}.npy'), allow_pickle=True)

arr = load_airport_delays()
print(type(arr), arr.shape)
print(arr[:5])
