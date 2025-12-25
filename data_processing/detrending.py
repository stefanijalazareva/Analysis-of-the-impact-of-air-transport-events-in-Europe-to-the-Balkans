"""Module to provide unified interface for all detrending methods and save results."""

from typing import Union, Callable
from pathlib import Path
import json
import pandas as pd
import numpy as np
from numpy import ndarray

try:
    from .detrend_timeseries import Detrender, detrend
except ImportError:
    from detrend_timeseries import Detrender, detrend

def detrend_and_save(
    ts: ndarray,
    /,
    method: Union[str, Callable[[ndarray, ...], ndarray]],
    *args,
    axis: int = None,
    output_dir: str = "data/Detrended",
    save_csv: bool = True,
    save_summary: bool = True,
    **kwargs,
) -> ndarray:
    """
    Detrend a time series and optionally save the results and summary.

    :param ts: Input time series (1D or 2D)
    :param method: Detrending method (str or callable)
    :param axis: Axis to apply detrending for multidimensional arrays
    :param output_dir: Directory to save results
    :param save_csv: If True, save detrended series as CSV
    :param save_summary: If True, save summary statistics as JSON
    :param args: Positional args for detrending function
    :param kwargs: Keyword args for detrending function
    :return: Detrended time series
    """
    if ts.ndim > 1 and axis is not None:
        ts_detrended = detrend(ts, method, *args, axis=axis, **kwargs)
    else:
        ts_detrended = detrend(ts, method, *args, **kwargs)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_csv:
        if ts_detrended.ndim == 1:
            df = pd.DataFrame(ts_detrended, columns=['detrended'])
        else:
            df = pd.DataFrame(ts_detrended)
        df.to_csv(out_path / "detrended_series.csv", index=False)

    if save_summary:
        summary = {
            "mean_original": float(np.mean(ts)),
            "mean_detrended": float(np.mean(ts_detrended)),
            "std_original": float(np.std(ts)),
            "std_detrended": float(np.std(ts_detrended)),
            "shape_original": ts.shape,
            "shape_detrended": ts_detrended.shape
        }
        with open(out_path / "detrended_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return ts_detrended

def show_detrending_methods():
    """Display all available detrending methods."""
    Detrender.list_methods()

if __name__ == "__main__":
    print(f"Original series: mean={ts_with_trend.mean():.2f}, std={ts_with_trend.std():.2f}")
    
    methods_to_test = ['z_score', 'delta', 'identity', 'second_difference']
    
    for method in methods_to_test:
        print(f"\nTesting method: {method}")
        try:
            output_dir = f"data/Detrended/{method}"
            ts_detrended = detrend_and_save(
                ts_with_trend,
                method=method,
                output_dir=output_dir,
                save_csv=True,
                save_summary=True
            )
            print(f"  Detrended series: mean={ts_detrended.mean():.2f}, std={ts_detrended.std():.2f}")
            print(f"  Results saved to: {output_dir}")
        except Exception as e:
            print(f"  Error: {e}")
    
    show_detrending_methods()
