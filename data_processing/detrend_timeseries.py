"""
Detrending methods for time series analysis.
Includes delta, identity, second_difference and z-score.
"""

from typing import Union, Callable, Optional
import logging
import numpy as np
from numpy import ndarray
from numba import prange

logging.getLogger("delaynet").addHandler(logging.NullHandler())
logging.basicConfig(
    format="%(asctime)s | %(levelname)8s | %(filename)s:%(lineno)d | %(message)s",
    level=logging.INFO,
)


class Detrender:
    """Class-based interface for time series detrending methods."""
    
    AVAILABLE_METHODS = {
        'delta': ['delta', 'Delta', 'DELTA'],
        'identity': ['identity', 'Identity', 'IDENTITY', 'none', 'None'],
        'second_difference': ['second_difference', 'second_diff', 'diff2', '2nd_diff'],
        'z_score': ['z_score', 'zscore', 'z', 'Z'],
    }
    
    def __init__(self, ts: ndarray, validate: bool = True):
        """Initialize detrender with time series.
        
        :param ts: Input time series (1D or 2D numpy array)
        :param validate: If True, validate input data
        """
        if validate:
            self._validate_input(ts)
        self.ts = ts
        self.original_shape = ts.shape
    
    @staticmethod
    def _validate_input(ts: ndarray):
        """Validate input time series."""
        if not isinstance(ts, ndarray):
            raise TypeError(f"ts must be ndarray, got {type(ts)}")
        if ts.size == 0:
            raise ValueError("ts must not be empty")
        if np.isnan(ts).any():
            raise ValueError("Input ts contains NaNs")
        if np.isinf(ts).any():
            raise ValueError("Input ts contains Infs")
    
    @staticmethod
    def _validate_output(ts_detrended: ndarray, original_shape: tuple, check_shape: bool = True):
        """Validate output time series."""
        if check_shape and ts_detrended.shape != original_shape:
            raise ValueError(f"Shape mismatch: {ts_detrended.shape} vs {original_shape}")
        if np.isnan(ts_detrended).any():
            raise ValueError("Detrended ts contains NaNs")
        if np.isinf(ts_detrended).any():
            raise ValueError("Detrended ts contains Infs")
    
    def delta(self, window_size: int = 10) -> ndarray:
        """Remove local mean using sliding window.
        
        :param window_size: Size of sliding window
        :return: Detrended time series
        """
        if not isinstance(window_size, (int, np.integer)) or window_size <= 0:
            raise ValueError("window_size must be positive integer")
        
        ts2 = np.copy(self.ts)
        for k in range(len(self.ts)):
            sub_ts = self.ts[max(0, k - window_size):k + window_size]
            ts2[k] = self.ts[k] - np.mean(sub_ts)
        
        self._validate_output(ts2, self.original_shape)
        return ts2
    
    def identity(self) -> ndarray:
        """Return original time series (no detrending).
        
        :return: Original time series
        """
        logging.warning("Identity used: no detrending applied")
        return self.ts
    
    def second_difference(self) -> ndarray:
        """Compute second-order difference.
        
        :return: Second difference (length reduced by 2)
        """
        t_ts = np.copy(self.ts)
        t_ts = t_ts[1:] - t_ts[:-1]
        t_ts = t_ts[1:] - t_ts[:-1]
        
        self._validate_output(t_ts, t_ts.shape, check_shape=False)
        return t_ts
    
    def z_score(self, periodicity: int = 1, max_periods: int = -1) -> ndarray:
        """Standardize using periodic mean and std.
        
        :param periodicity: Period for seasonal detrending (1 = no seasonality)
        :param max_periods: Max periods to use (-1 = all)
        :return: Z-score normalized time series
        """
        if periodicity <= 0:
            raise ValueError("periodicity must be positive")
        if max_periods < -1:
            raise ValueError("max_periods must be >= -1")
        if 2 * periodicity + 1 > self.ts.size:
            raise ValueError("Periodicity too large for ts size")
        
        if max_periods * periodicity + 1 >= self.ts.size:
            max_periods = -1
        
        # Simple case: standard z-score
        if periodicity == 1 and max_periods == -1:
            ts_std = np.std(self.ts)
            result = np.zeros(self.ts.size) if ts_std == 0 else (self.ts - np.mean(self.ts)) / ts_std
        else:
            # Periodic z-score
            result = (self._z_score_loop_all(periodicity) if max_periods == -1
                     else self._z_score_loop_partial(periodicity, max_periods))
        
        self._validate_output(result, self.original_shape)
        return result
    
    def _z_score_loop_all(self, periodicity: int) -> ndarray:
        """Z-score using all periods."""
        ts2 = np.copy(self.ts)
        for k in prange(self.ts.size):
            sub_ts = self._get_sub_ts_all_periods(k, periodicity)
            st_dev = np.std(sub_ts)
            ts2[k] = 0.0 if st_dev == 0 else (self.ts[k] - np.mean(sub_ts)) / st_dev
        return ts2
    
    def _z_score_loop_partial(self, periodicity: int, max_periods: int) -> ndarray:
        """Z-score using limited periods."""
        ts2 = np.copy(self.ts)
        for k in prange(self.ts.size):
            sub_ts = self._get_sub_ts_partial(k, periodicity, max_periods)
            st_dev = np.std(sub_ts)
            ts2[k] = 0.0 if st_dev == 0 else (self.ts[k] - np.mean(sub_ts)) / st_dev
        return ts2
    
    def _get_sub_ts_all_periods(self, k: int, periodicity: int) -> ndarray:
        """Get all values at same phase."""
        phase = k % periodicity
        indices = np.arange(phase, len(self.ts), periodicity)
        return self.ts[indices[indices != k]]
    
    def _get_sub_ts_partial(self, k: int, periodicity: int, max_periods: int) -> ndarray:
        """Get nearby values at same phase."""
        remainder = k % periodicity
        start_index = max(remainder, k - max_periods * periodicity)
        end_index = min(len(self.ts) - remainder, k + (max_periods + 1) * periodicity)
        indices = np.arange(start_index, end_index, periodicity)
        return self.ts[indices[indices != k]]
    
    def detrend(self, method: Union[str, Callable], *args, **kwargs) -> ndarray:
        """Apply detrending method by name or callable.
        
        :param method: Method name or callable
        :param args: Positional arguments for method
        :param kwargs: Keyword arguments for method
        :return: Detrended time series
        """
        if callable(method):
            return method(self.ts, *args, **kwargs)
        
        # Find method by name
        method_lower = method.lower()
        for method_name, aliases in self.AVAILABLE_METHODS.items():
            if method_lower in [a.lower() for a in aliases]:
                method_func = getattr(self, method_name)
                return method_func(*args, **kwargs)
        
        raise ValueError(f"Unknown detrending method: {method}")
    
    @classmethod
    def list_methods(cls):
        """Print available detrending methods."""
        print("Available detrending methods:")
        for method, aliases in cls.AVAILABLE_METHODS.items():
            print(f"\n{method}:")
            for alias in aliases:
                print(f"  - {alias}")
        print()


# Backward compatibility: standalone function interface
def detrend(ts: ndarray, method: Union[str, Callable], *args, **kwargs) -> ndarray:
    """Apply detrending method to time series.
    
    :param ts: Input time series
    :param method: Detrending method name or callable
    :param args: Positional arguments for method
    :param kwargs: Keyword arguments for method
    :return: Detrended time series
    """
    detrender = Detrender(ts, validate=True)
    return detrender.detrend(method, *args, **kwargs)


# Export available methods for backward compatibility
__all_detrending_names_simple__ = Detrender.AVAILABLE_METHODS
