"""
Advanced Tail & Stability Analysis for Air Transport Delays
===========================================================

This script assumes the following files exist:

- results/burr_analysis/burr_analysis_summary.csv
    Contains Burr XII fits + model/data percentiles for each airport and delay type.

- results/burr_analysis/burr_ci_summary.csv
    Contains bootstrap CI for Burr XII parameters (c, d, loc, scale).

- results/NCT_confidence_intervals_results/bootstrap_CI_all_airports.csv
    Contains bootstrap CI for NCT parameters.

- data/NonCentralT/noncentral_t_parameters.csv
    Contains fitted NCT parameters (df, nc, loc, scale) for each airport.

- data/RawData/Delays_<AIRPORT>.npy
    Raw delay arrays for each airport.

Outputs will be written to: results/advanced_tail_analysis/
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# CONFIG & PATHS

BASE_DIR = Path(".")
Burr_summary_path = BASE_DIR / "results" / "burr_analysis" / "burr_analysis_summary.csv"
Burr_CI_path = BASE_DIR / "results" / "burr_analysis" / "burr_ci_summary.csv"
NCT_CI_path = BASE_DIR / "results" / "NCT_confidence_intervals_results" / "bootstrap_CI_all_airports.csv"
NCT_param_path = BASE_DIR / "data" / "NonCentralT" / "noncentral_t_parameters.csv"
RAW_DATA_DIR = BASE_DIR / "data" / "RawData"

OUTDIR = BASE_DIR / "results" / "advanced_tail_analysis"
OUTDIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid")

EUROPE = ["EGLL", "LFPG", "EHAM", "EDDF", "LEMD", "LEBL", "EDDM", "EGKK", "LIRF", "EIDW"]
BALKANS = ["LATI", "LQSA", "LBSF", "LBBG", "LDZA", "LDSP", "LDDU", "BKPR", "LYTV", "LWSK"]

def get_region(icao: str) -> str:
    if icao in EUROPE:
        return "Europe"
    if icao in BALKANS:
        return "Balkans"
    return "Unknown"

# LOAD DATA

burr_df = pd.read_csv(Burr_summary_path)
burr_ci_df = pd.read_csv(Burr_CI_path)
nct_ci_df = pd.read_csv(NCT_CI_path)
nct_params_df = pd.read_csv(NCT_param_path)

# Normalize column names just in case
burr_df.columns = [c.strip() for c in burr_df.columns]
burr_ci_df.columns = [c.strip() for c in burr_ci_df.columns]
nct_ci_df.columns = [c.strip() for c in nct_ci_df.columns]
nct_params_df.columns = [c.strip() for c in nct_params_df.columns]

# 1) TAIL RISK INDEX (P95 - P90)/P90  FOR DATA + BURR

# Positive delays only
burr_pos = burr_df[burr_df["Delay_Type"] == "positive"].copy()

for col in ["Airport", "Data_P90", "Data_P95", "P90", "P95"]:
    if col not in burr_pos.columns:
        raise ValueError(f"Expected column '{col}' not found in burr_analysis_summary.csv.")

burr_pos["Region"] = burr_pos["Airport"].apply(get_region)

# Tail risk for data and Burr model
burr_pos["TailRisk_Data"] = (burr_pos["Data_P95"] - burr_pos["Data_P90"]) / burr_pos["Data_P90"]
burr_pos["TailRisk_Burr"] = (burr_pos["P95"] - burr_pos["P90"]) / burr_pos["P90"]

tail_risk_long = burr_pos.melt(
    id_vars=["Airport", "Region"],
    value_vars=["TailRisk_Data", "TailRisk_Burr"],
    var_name="Source",
    value_name="TailRisk",
)

tail_risk_long["Source"] = tail_risk_long["Source"].map(
    {"TailRisk_Data": "Data", "TailRisk_Burr": "Burr XII"}
)

# Boxplot by Source
plt.figure(figsize=(8, 5))
sns.boxplot(data=tail_risk_long, x="Source", y="TailRisk", hue="Region")
plt.title("Tail Risk Index (P95 - P90)/P90 – Data vs Burr XII")
plt.ylabel("Tail Risk")
plt.xlabel("")
plt.tight_layout()
plt.savefig(OUTDIR / "tail_risk_data_vs_burr_by_region.png", dpi=300)
plt.close()

# Barplot per airport (Burr XII only)
plt.figure(figsize=(12, 5))
order = burr_pos.sort_values("TailRisk_Burr")["Airport"]
sns.barplot(
    data=burr_pos,
    x="Airport",
    y="TailRisk_Burr",
    hue="Region",
    order=order,
)
plt.title("Tail Risk Index per Airport – Burr XII (Positive Delays)")
plt.ylabel("Tail Risk (P95 - P90)/P90")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTDIR / "tail_risk_burr_per_airport.png", dpi=300)
plt.close()


# 2) CI STABILITY HEATMAPS (Burr XII)

for col in ["Airport", "Delay_Type", "Param", "CI_lower", "CI_upper"]:
    if col not in burr_ci_df.columns:
        raise ValueError(f"Expected column '{col}' in burr_ci_summary.csv.")

burr_ci_df["CI_width"] = burr_ci_df["CI_upper"] - burr_ci_df["CI_lower"]
burr_ci_df["Region"] = burr_ci_df["Airport"].apply(get_region)

# Heatmap – positive delays
for delay_type in ["positive", "negative"]:
    sub = burr_ci_df[burr_ci_df["Delay_Type"] == delay_type].copy()
    if sub.empty:
        continue

    pivot = sub.pivot_table(
        index="Airport", columns="Param", values="CI_width", aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "CI Width"},
    )
    plt.title(f"Burr XII – CI Width Heatmap ({delay_type.capitalize()} Delays)")
    plt.ylabel("Airport")
    plt.xlabel("Parameter")
    plt.tight_layout()
    plt.savefig(OUTDIR / f"burr_ci_heatmap_{delay_type}.png", dpi=300)
    plt.close()


# 3) POSITIVE vs NEGATIVE Burr PARAMETERS

# Compare c, d, scale, loc by Delay_Type
expected_params_cols = ["Shape_c", "Shape_d", "Location", "Scale"]
for col in expected_params_cols:
    if col not in burr_df.columns:
        raise ValueError(f"Expected column '{col}' in burr_analysis_summary.csv.")

burr_df["Region"] = burr_df["Airport"].apply(get_region)

params_long = burr_df.melt(
    id_vars=["Airport", "Region", "Delay_Type"],
    value_vars=expected_params_cols,
    var_name="Parameter",
    value_name="Value",
)

plt.figure(figsize=(12, 6))
sns.boxplot(
    data=params_long,
    x="Parameter",
    y="Value",
    hue="Delay_Type",
)
plt.title("Burr XII Parameters – Positive vs Negative Delays (All Airports)")
plt.xlabel("Parameter")
plt.ylabel("Estimated Value")
plt.legend(title="Delay Type")
plt.tight_layout()
plt.savefig(OUTDIR / "burr_params_positive_vs_negative_boxplot.png", dpi=300)
plt.close()

# By region (optional)
plt.figure(figsize=(12, 6))
sns.boxplot(
    data=params_long,
    x="Parameter",
    y="Value",
    hue="Region",
)
plt.title("Burr XII Parameters – Europe vs Balkans (Positive & Negative)")
plt.xlabel("Parameter")
plt.ylabel("Estimated Value")
plt.tight_layout()
plt.savefig(OUTDIR / "burr_params_region_boxplot.png", dpi=300)
plt.close()

# 4) CDF COMPARISON: DATA vs BURR vs NCT (few representative airports)

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of {candidates} found in columns {df.columns}.")


airport_col_nct = find_col(nct_params_df, ["airport", "Airport", "ICAO"])
df_col = find_col(nct_params_df, ["df", "DF"])
nc_col = find_col(nct_params_df, ["nc", "NC"])
loc_col = find_col(nct_params_df, ["loc", "Location", "loc_param", "loc (mean)", "loc(mean)"])
scale_col = find_col(nct_params_df, ["scale", "Scale", "scale_param", "scale (std)", "scale(std)"])


# Pick 4 representative airport
representative_airports = ["EGLL", "LFPG", "LWSK", "BKPR"]
cdf_outdir = OUTDIR / "cdf_comparisons"
cdf_outdir.mkdir(exist_ok=True)

for icao in representative_airports:
    # Load raw delays
    npy_path = RAW_DATA_DIR / f"Delays_{icao}.npy"
    if not npy_path.exists():
        print(f"[CDF] Raw data not found for {icao}, skipping.")
        continue

    data_arr = np.load(npy_path, allow_pickle=True)
    # assuming delay in 4-та колона
    delays = data_arr[:, 3].astype(float)
    pos_delays = delays[delays > 0]

    if len(pos_delays) < 100:
        print(f"[CDF] Not enough positive delays for {icao}, skipping.")
        continue

    # Empirical CDF
    x_emp = np.sort(pos_delays)
    y_emp = np.arange(1, len(x_emp) + 1) / len(x_emp)

    # Burr params for positive delays
    row_burr = burr_df[(burr_df["Airport"] == icao) & (burr_df["Delay_Type"] == "positive")]
    if row_burr.empty:
        print(f"[CDF] No Burr parameters for {icao}, skipping.")
        continue

    c_burr = float(row_burr["Shape_c"].iloc[0])
    d_burr = float(row_burr["Shape_d"].iloc[0])
    loc_burr = float(row_burr["Location"].iloc[0])
    scale_burr = float(row_burr["Scale"].iloc[0])

    # NCT params
    row_nct = nct_params_df[nct_params_df[airport_col_nct] == icao]
    if row_nct.empty:
        print(f"[CDF] No NCT parameters for {icao}, skipping.")
        continue

    df_nct = float(row_nct[df_col].iloc[0])
    nc_nct = float(row_nct[nc_col].iloc[0])
    loc_nct = float(row_nct[loc_col].iloc[0])
    scale_nct = float(row_nct[scale_col].iloc[0])

    # Define grid (quantile-based for robustness)
    x_grid = np.linspace(np.percentile(pos_delays, 1), np.percentile(pos_delays, 99), 300)

    # Model CDFs
    burr_cdf = stats.burr12(c_burr, d_burr, loc=loc_burr, scale=scale_burr).cdf(x_grid)
    nct_cdf = stats.nct(df_nct, nc_nct, loc=loc_nct, scale=scale_nct).cdf(x_grid)

    # Plot
    plt.figure(figsize=(7, 5))
    plt.plot(x_emp, y_emp, label="Empirical", color="black", linewidth=2)
    plt.plot(x_grid, burr_cdf, label="Burr XII", linestyle="--")
    plt.plot(x_grid, nct_cdf, label="NCT", linestyle=":")
    plt.title(f"CDF Comparison – {icao} (Positive Delays)")
    plt.xlabel("Delay (minutes)")
    plt.ylabel("CDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(cdf_outdir / f"cdf_comparison_{icao}.png", dpi=300)
    plt.close()


print("\nAll advanced analyses finished.")
print(f"Outputs saved under: {OUTDIR.resolve()}\n")
