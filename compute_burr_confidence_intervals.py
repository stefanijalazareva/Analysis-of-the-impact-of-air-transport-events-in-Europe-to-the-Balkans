"""
Compute bootstrap confidence intervals for Burr XII parameters (c, d, loc, scale)
for BOTH positive and negative delays.

Uses already fitted parameters from:
results/burr_analysis/burr_analysis_summary.csv

Outputs:
- burr_ci_summary.csv
- burr_bootstrap_params.csv
- Burr_CI_boxplot_<Region>_<DelayType>.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import burr

# ---------------- CONFIG ----------------
N_BOOTSTRAP = 500
MAX_SAMPLE_SIZE = 40000
RANDOM_SEED = 42

EUROPE = ["EGLL","LFPG","EHAM","EDDF","LEMD","LEBL","EDDM","EGKK","LIRF","EIDW"]
BALKANS = ["LATI","LQSA","LBSF","LBBG","LDZA","LDSP","LDDU","BKPR","LYTV","LWSK"]

RESULTS_DIR = Path("results/burr_analysis")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Bootstrap fitting function

def bootstrap_burr(row, rng):
    """
    Returns matrix: boot_params [n_success x 4] with columns (c,d,loc,scale)
    """
    airport = row["Airport"]
    delay_type = row["Delay_Type"]

    c = row["Shape_c"]
    d = row["Shape_d"]
    loc = row["Location"]
    scale = row["Scale"]
    n = int(row["Sample_Size"])

    n_sample = min(n, MAX_SAMPLE_SIZE)

    frozen = burr(c, d, loc=loc, scale=scale)
    boot_params = []

    for _ in range(N_BOOTSTRAP):
        sample = frozen.rvs(size=n_sample, random_state=rng)

        try:
            # Re-fit Burr XII
            c_hat, d_hat, loc_hat, scale_hat = burr.fit(sample)
            boot_params.append([c_hat, d_hat, loc_hat, scale_hat])
        except:
            continue

    if len(boot_params) < 10:
        print(f"WARNING: only {len(boot_params)} successful fits for {airport}-{delay_type}")

    return np.array(boot_params)



# Compute CI for all airports

def compute_ci(df):
    rng = np.random.default_rng(RANDOM_SEED)
    summary_records = []
    boot_records = []

    for _, row in df.iterrows():
        airport = row["Airport"]
        d_type = row["Delay_Type"]

        print(f"Bootstrapping Burr XII for {airport} ({d_type}) ...")

        boot_params = bootstrap_burr(row, rng)
        if boot_params is None or len(boot_params) < 10:
            continue

        param_names = ["c", "d", "loc", "scale"]
        original_cols = ["Shape_c", "Shape_d", "Location", "Scale"]

        # Record bootstrap params (long format)
        for col_index, param in enumerate(param_names):
            for val in boot_params[:, col_index]:
                boot_records.append({
                    "Airport": airport,
                    "Delay_Type": d_type,
                    "Param": param,
                    "Value": val
                })

        # Compute CI per parameter
        for col_index, param in enumerate(param_names):
            vals = boot_params[:, col_index]
            summary_records.append({
                "Airport": airport,
                "Delay_Type": d_type,
                "Param": param,
                "Mean": np.mean(vals),
                "CI_lower": np.percentile(vals, 2.5),
                "CI_upper": np.percentile(vals, 97.5),
                "N_bootstrap": len(vals)
            })

    return pd.DataFrame(summary_records), pd.DataFrame(boot_records)


# Plot CI boxplots by region
def plot_region(df_boot, df_params, region, airports, delay_type, out_file):

    region_boot = df_boot[(df_boot["Airport"].isin(airports)) &
                          (df_boot["Delay_Type"] == delay_type)]

    region_params = df_params[(df_params["Airport"].isin(airports)) &
                              (df_params["Delay_Type"] == delay_type)]

    if region_boot.empty:
        print(f"No data for region {region} ({delay_type}) - skipping.")
        return

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    fig.suptitle(f"Burr XII Parameter Uncertainty — {region} ({delay_type})", fontsize=18)

    param_info = [
        ("c", "Shape_c", "Shape parameter c"),
        ("d", "Shape_d", "Shape parameter d"),
        ("loc", "Location", "Location parameter"),
        ("scale", "Scale", "Scale parameter"),
    ]

    for ax, (param, orig_col, title) in zip(axes, param_info):
        data = []
        orig_vals = []
        sorted_airports = [a for a in airports if a in region_boot["Airport"].unique()]

        for ap in sorted_airports:
            vals = region_boot[(region_boot["Airport"] == ap) &
                               (region_boot["Param"] == param)]["Value"]
            if len(vals) == 0:
                continue

            data.append(vals)

            orig_val = float(region_params.loc[region_params["Airport"] == ap, orig_col].iloc[0])
            orig_vals.append(orig_val)

        if not data:
            ax.set_visible(False)
            continue

        positions = np.arange(1, len(data) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.5,
            whis=(2.5, 97.5),
            showmeans=True
        )

        ax.scatter(positions, orig_vals, color="black", marker="x", s=60)

        ax.set_title(title, fontsize=13)
        ax.set_xticks(positions)
        ax.set_xticklabels(sorted_airports, rotation=45, fontsize=10)
        ax.set_ylabel("Parameter value")

    plt.tight_layout(rect=[0, 0, 0.95, 0.92])
    plt.savefig(out_file, dpi=300)
    plt.close(fig)

def main():
    summary_path = RESULTS_DIR / "burr_analysis_summary.csv"
    df = pd.read_csv(summary_path)

    df = df[df["Distribution"] == "Burr XII"].copy()

    # Compute CI
    ci_df, boot_df = compute_ci(df)

    ci_df.to_csv(RESULTS_DIR / "burr_ci_summary.csv", index=False)
    boot_df.to_csv(RESULTS_DIR / "burr_bootstrap_params.csv", index=False)

    print("\nSaved burr_ci_summary.csv")
    print("Saved burr_bootstrap_params.csv")

    # Visualizations (4 panels × 4 groups)
    plot_region(boot_df, df, "Europe", EUROPE, "positive",
                RESULTS_DIR / "Burr_CI_Europe_Positive.png")

    plot_region(boot_df, df, "Balkans", BALKANS, "positive",
                RESULTS_DIR / "Burr_CI_Balkans_Positive.png")

    plot_region(boot_df, df, "Europe", EUROPE, "negative",
                RESULTS_DIR / "Burr_CI_Europe_Negative.png")

    plot_region(boot_df, df, "Balkans", BALKANS, "negative",
                RESULTS_DIR / "Burr_CI_Balkans_Negative.png")

    print("Saved regional CI plots.")


if __name__ == "__main__":
    main()
