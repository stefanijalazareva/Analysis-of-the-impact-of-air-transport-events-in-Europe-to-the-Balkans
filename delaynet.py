"""
DelayNet: Network Analysis for Air Transport Delays

This module provides specialized functions for preprocessing and analyzing
time series data related to air transport delays, with a focus on
detrending methods and network connectivity tests.
"""

import numpy as np
import pandas as pd
from scipy import signal
import logging

def detrend(data, method='delta', periodicity=None, axis=1):
    """
    Detrend time series data using various methods.

    Args:
        data: numpy array of shape (n_nodes, n_times) or (n_times, n_nodes)
        method: detrending method to use
            'delta': local mean subtraction (first difference)
            'delta2': second difference
            'zs': z-score with optional periodicity
            'linear': linear detrending
        periodicity: period length for z-score detrending (e.g., 24 for hourly data with daily patterns)
        axis: axis along which to perform detrending (0 for rows, 1 for columns)

    Returns:
        numpy array with detrended data of same shape as input
    """
    # Make a copy to avoid modifying the original data
    result = np.copy(data)

    # Replace NaN with zeros to avoid propagation issues during detrending
    result = np.nan_to_num(result, nan=0.0)

    if method == 'delta':
        # First difference (local mean subtraction)
        result = np.diff(result, n=1, axis=axis)

        # Pad with a zero at the beginning to maintain shape
        pad_shape = list(result.shape)
        pad_shape[axis] = 1
        padding = np.zeros(pad_shape)

        if axis == 0:
            result = np.vstack([padding, result])
        else:
            result = np.hstack([padding, result])

    elif method == 'delta2':
        # Second difference
        result = np.diff(result, n=2, axis=axis)

        # Pad with zeros at the beginning to maintain shape
        pad_shape = list(result.shape)
        pad_shape[axis] = 2
        padding = np.zeros(pad_shape)

        if axis == 0:
            result = np.vstack([padding, result])
        else:
            result = np.hstack([padding, result])

    elif method == 'zs':
        # Z-score detrending
        if periodicity is None:
            # Standard z-score across the whole time series
            mean = np.mean(result, axis=axis, keepdims=True)
            std = np.std(result, axis=axis, keepdims=True)
            # Avoid division by zero
            std = np.where(std == 0, 1.0, std)
            result = (result - mean) / std
        else:
            # Periodic z-score (e.g., remove daily patterns)
            shape = result.shape

            if axis == 1:
                # For each node, calculate z-score with respect to the same hour across days
                for i in range(shape[0]):  # For each node
                    series = result[i, :]
                    n_periods = len(series) // periodicity

                    if n_periods > 0:  # Only proceed if we have at least one complete period
                        # Reshape to [n_periods, periodicity]
                        periodic_view = series[:n_periods * periodicity].reshape(n_periods, periodicity)

                        # Calculate mean and std for each position in the period
                        periodic_mean = np.mean(periodic_view, axis=0)
                        periodic_std = np.std(periodic_view, axis=0)

                        # Replace zeros in std with 1 to avoid division by zero
                        periodic_std = np.where(periodic_std == 0, 1.0, periodic_std)

                        # Apply z-score for each position in the period
                        for j in range(n_periods):
                            start_idx = j * periodicity
                            end_idx = start_idx + periodicity
                            result[i, start_idx:end_idx] = ((series[start_idx:end_idx] - periodic_mean) /
                                                           periodic_std)

                        # Handle remaining values if series length is not a multiple of periodicity
                        if len(series) > n_periods * periodicity:
                            remainder_start = n_periods * periodicity
                            remainder = len(series) - remainder_start
                            for j in range(remainder):
                                period_pos = j % periodicity
                                result[i, remainder_start + j] = ((series[remainder_start + j] - periodic_mean[period_pos]) /
                                                                periodic_std[period_pos])
            else:
                # For each time point, calculate z-score across nodes
                for i in range(shape[1]):  # For each time point
                    series = result[:, i]
                    n_periods = len(series) // periodicity

                    if n_periods > 0:
                        # Reshape to [n_periods, periodicity]
                        periodic_view = series[:n_periods * periodicity].reshape(n_periods, periodicity)

                        # Calculate mean and std for each position in the period
                        periodic_mean = np.mean(periodic_view, axis=0)
                        periodic_std = np.std(periodic_view, axis=0)

                        # Replace zeros in std with 1 to avoid division by zero
                        periodic_std = np.where(periodic_std == 0, 1.0, periodic_std)

                        # Apply z-score for each position in the period
                        for j in range(n_periods):
                            start_idx = j * periodicity
                            end_idx = start_idx + periodicity
                            result[start_idx:end_idx, i] = ((series[start_idx:end_idx] - periodic_mean) /
                                                           periodic_std)

                        # Handle remaining values
                        if len(series) > n_periods * periodicity:
                            remainder_start = n_periods * periodicity
                            remainder = len(series) - remainder_start
                            for j in range(remainder):
                                period_pos = j % periodicity
                                result[remainder_start + j, i] = ((series[remainder_start + j] - periodic_mean[period_pos]) /
                                                                periodic_std[period_pos])

    elif method == 'linear':
        # Linear detrending
        if axis == 1:
            for i in range(result.shape[0]):
                result[i, :] = signal.detrend(result[i, :])
        else:
            for i in range(result.shape[1]):
                result[:, i] = signal.detrend(result[:, i])

    else:
        raise ValueError(f"Unknown detrending method: {method}. Valid options are: 'delta', 'delta2', 'zs', 'linear'.")

    return result


def compare_detrending_methods(data, methods=['delta', 'delta2', 'zs', 'linear'], periodicity=24, axis=1):
    """
    Compare different detrending methods on the same dataset.

    Args:
        data: numpy array of shape (n_nodes, n_times) or (n_times, n_nodes)
        methods: list of detrending methods to compare
        periodicity: period length for z-score detrending
        axis: axis along which to perform detrending

    Returns:
        dict: Dictionary with each method's name as key and detrended data as value
    """
    results = {}

    for method in methods:
        if method == 'zs':
            results[method] = detrend(data, method=method, periodicity=periodicity, axis=axis)
        else:
            results[method] = detrend(data, method=method, axis=axis)

    return results


def check_stationarity(data, alpha=0.05, axis=1):
    """
    Check stationarity of time series using Augmented Dickey-Fuller test.
    If statsmodels is not available, falls back to a simpler heuristic method.

    Args:
        data: numpy array of shape (n_nodes, n_times) or (n_times, n_nodes)
        alpha: significance level for ADF test
        axis: axis along which to check stationarity

    Returns:
        numpy array: Boolean array indicating stationarity for each series
        numpy array: p-values from ADF test for each series (or NaN if using fallback)
    """
    try:
        # Try to import statsmodels
        from statsmodels.tsa.stattools import adfuller

        stationary = []
        p_values = []

        if axis == 1:
            # Check stationarity for each row (node)
            for i in range(data.shape[0]):
                try:
                    # Handle NaN values and ensure enough data points
                    series = data[i, :]
                    if np.isnan(series).all() or len(series) < 10:
                        p_values.append(1.0)  # Not stationary
                        stationary.append(False)
                        continue

                    # Replace remaining NaNs with forward fill and then backward fill
                    series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values

                    result = adfuller(series, maxlag=1)
                    p_value = result[1]
                    p_values.append(p_value)
                    stationary.append(p_value < alpha)
                except Exception as e:
                    # If ADF test fails, assume not stationary
                    logging.warning(f"ADF test failed for series {i}: {str(e)}")
                    p_values.append(1.0)
                    stationary.append(False)
        else:
            # Check stationarity for each column (time point)
            for i in range(data.shape[1]):
                try:
                    series = data[:, i]
                    if np.isnan(series).all() or len(series) < 10:
                        p_values.append(1.0)
                        stationary.append(False)
                        continue

                    series = pd.Series(series).fillna(method='ffill').fillna(method='bfill').values

                    result = adfuller(series, maxlag=1)
                    p_value = result[1]
                    p_values.append(p_value)
                    stationary.append(p_value < alpha)
                except Exception as e:
                    logging.warning(f"ADF test failed for series {i}: {str(e)}")
                    p_values.append(1.0)
                    stationary.append(False)

        return np.array(stationary), np.array(p_values)

    except ImportError:
        # Fallback method if statsmodels is not available
        logging.warning("statsmodels not available, using simple heuristic for stationarity check")

        def simple_stationarity_check(series):
            """Simple heuristic check for stationarity"""
            # Remove NaN values
            series = series[~np.isnan(series)]
            if len(series) < 10:
                return False, 1.0

            # Split series into two halves
            half = len(series) // 2
            first_half, second_half = series[:half], series[half:]

            # Check if mean and variance are similar in both halves
            mean_diff = abs(np.mean(first_half) - np.mean(second_half))
            var_diff = abs(np.var(first_half) - np.var(second_half))

            # Normalize by the overall mean and variance
            overall_mean = np.mean(series)
            overall_var = np.var(series)

            if overall_mean == 0:
                mean_ratio = 0 if mean_diff == 0 else 1
            else:
                mean_ratio = mean_diff / (abs(overall_mean) + 1e-10)

            if overall_var == 0:
                var_ratio = 0 if var_diff == 0 else 1
            else:
                var_ratio = var_diff / (overall_var + 1e-10)

            # Heuristic p-value and stationarity check
            p_value = (mean_ratio + var_ratio) / 2
            return p_value < alpha, p_value

        stationary = []
        p_values = []

        if axis == 1:
            for i in range(data.shape[0]):
                is_stationary, p_value = simple_stationarity_check(data[i, :])
                stationary.append(is_stationary)
                p_values.append(p_value)
        else:
            for i in range(data.shape[1]):
                is_stationary, p_value = simple_stationarity_check(data[:, i])
                stationary.append(is_stationary)
                p_values.append(p_value)

        return np.array(stationary), np.array(p_values)
