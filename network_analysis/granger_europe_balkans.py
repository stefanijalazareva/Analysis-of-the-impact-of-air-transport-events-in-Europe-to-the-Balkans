import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import logging

# -----------------------------
# CONFIGURATION
# -----------------------------
DETREND_METHOD = "zs"        # 'zs' or 'delta'
MAX_LAG = 24                # hours
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
# LOAD DETRENDED DATA
# -----------------------------
def load_detrended_data(method):
    path = Path(f"data/DetrendedData/detrended_{method}.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Detrended data not found: {path}")
    df = pd.read_parquet(path)
    logging.info(f"Loaded detrended data: {df.shape}")
    return df

# -----------------------------
# BUILD REGIONAL TIME SERIES
# -----------------------------
def build_regional_series(df):
    europe_ts = df[EUROPE_AIRPORTS].mean(axis=1)
    balkans_ts = df[BALKANS_AIRPORTS].mean(axis=1)

    regional_df = pd.DataFrame({
        "Balkans": balkans_ts,
        "Europe": europe_ts
    }).dropna()

    logging.info(f"Regional series length: {len(regional_df)}")
    return regional_df

# -----------------------------
# GRANGER CAUSALITY ANALYSIS
# -----------------------------
def granger_test_europe_to_balkans(regional_df, max_lag):
    p_values = []
    f_stats = []

    for lag in range(1, max_lag + 1):
        result = grangercausalitytests(
            regional_df[["Balkans", "Europe"]],
            maxlag=lag,
            verbose=False
        )

        # Extract F-test statistics for this lag
        f_test = result[lag][0]["ssr_ftest"]
        f_stat, p_val = f_test[0], f_test[1]

        p_values.append(p_val)
        f_stats.append(f_stat)

        logging.info(
            f"Lag {lag:2d} | F = {f_stat:8.3f} | p-value = {p_val:.4e}"
        )

    return np.array(p_values), np.array(f_stats)

# -----------------------------
# VISUALIZATION (Fig.2-style)
# -----------------------------
def plot_pvalue_vs_lag(p_values, alpha, output_path):
    lags = np.arange(1, len(p_values) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(lags, p_values, marker="o", linewidth=2, label="p-value")
    plt.axhline(alpha, color="red", linestyle="--", label=f"α = {alpha}")

    plt.yscale("log")
    plt.xlabel("Lag (hours)")
    plt.ylabel("p-value (log scale)")
    plt.title("Granger Causality: Europe → Balkans\np-value vs Lag")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    logging.info("Starting Granger causality analysis: Europe → Balkans")

    df = load_detrended_data(DETREND_METHOD)
    regional_df = build_regional_series(df)

    p_values, f_stats = granger_test_europe_to_balkans(
        regional_df, MAX_LAG
    )

    results_df = pd.DataFrame({
        "lag_hours": np.arange(1, MAX_LAG + 1),
        "p_value": p_values,
        "f_statistic": f_stats
    })

    results_df.to_csv(
        OUTPUT_DIR / "granger_europe_to_balkans.csv",
        index=False
    )

    plot_pvalue_vs_lag(
        p_values,
        ALPHA,
        OUTPUT_DIR / "pvalue_vs_lag_europe_balkans.png"
    )

    best_lag = results_df.loc[results_df["p_value"].idxmin()]

    logging.info("=== SUMMARY ===")
    logging.info(
        f"Minimum p-value at lag {int(best_lag.lag_hours)} hours "
        f"(p = {best_lag.p_value:.4e})"
    )

    logging.info(f"Results saved to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
