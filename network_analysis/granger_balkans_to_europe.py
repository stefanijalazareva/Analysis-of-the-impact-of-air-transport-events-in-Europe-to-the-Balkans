import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import logging
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# CONFIGURATION
# -----------------------------
DETREND_METHOD = "zs"
MAX_LAG = 24
ALPHA = 0.05

EUROPE_AIRPORTS = [
    'EGLL', 'LFPG', 'EHAM', 'EDDF', 'LEMD',
    'LEBL', 'EDDM', 'EGKK', 'LIRF', 'EIDW'
]

BALKANS_AIRPORTS = [
    'LATI', 'LQSA', 'LBSF', 'LBBG', 'LDZA',
    'LDSP', 'LDDU', 'BKPR', 'LYTV', 'LWSK'
]

OUTPUT_DIR = Path("results/granger_europe_balkans")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# -----------------------------
# LOAD DATA
# -----------------------------
def load_detrended_data(method):
    path = Path(f"data/DetrendedData/detrended_{method}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Missing detrended data: {path}")
    return pd.read_parquet(path)

# -----------------------------
# BUILD REGIONAL SERIES
# -----------------------------
def build_regional_series(df):
    return pd.DataFrame({
        "Europe": df[EUROPE_AIRPORTS].mean(axis=1),
        "Balkans": df[BALKANS_AIRPORTS].mean(axis=1)
    }).dropna()

# -----------------------------
# GRANGER TEST
# -----------------------------
def granger_test_balkans_to_europe(regional_df, max_lag):
    p_vals, f_stats = [], []

    for lag in range(1, max_lag + 1):
        result = grangercausalitytests(
            regional_df[["Europe", "Balkans"]],
            maxlag=lag,
            verbose=False
        )

        f_stat, p_val = result[lag][0]["ssr_ftest"][:2]
        p_vals.append(p_val)
        f_stats.append(f_stat)

        logging.info(
            f"Lag {lag:2d} | F = {f_stat:8.3f} | p-value = {p_val:.4e}"
        )

    return np.array(p_vals), np.array(f_stats)

# -----------------------------
# PLOT
# -----------------------------
def plot_pvalue_vs_lag(p_values, alpha, path):
    lags = np.arange(1, len(p_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(lags, p_values, marker="o", linewidth=2)
    plt.axhline(alpha, color="red", linestyle="--", label=f"α = {alpha}")
    plt.yscale("log")

    plt.xlabel("Lag (hours)")
    plt.ylabel("p-value (log scale)")
    plt.title("Granger Causality: Balkans → Europe\np-value vs Lag")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
def main():
    logging.info("Starting Granger causality: Balkans → Europe")

    df = load_detrended_data(DETREND_METHOD)
    regional_df = build_regional_series(df)

    p_vals, f_stats = granger_test_balkans_to_europe(
        regional_df, MAX_LAG
    )

    results = pd.DataFrame({
        "lag_hours": np.arange(1, MAX_LAG + 1),
        "p_value": p_vals,
        "f_statistic": f_stats
    })

    results.to_csv(
        OUTPUT_DIR / "granger_balkans_to_europe.csv",
        index=False
    )

    plot_pvalue_vs_lag(
        p_vals,
        ALPHA,
        OUTPUT_DIR / "pvalue_vs_lag_balkans_europe.png"
    )

    best = results.loc[results["p_value"].idxmin()]
    logging.info(
        f"Minimum p-value at lag {int(best.lag_hours)} "
        f"(p = {best.p_value:.4e})"
    )

    logging.info(f"Results saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
